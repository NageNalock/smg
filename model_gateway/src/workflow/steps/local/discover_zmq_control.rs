//! Discover control metadata after direct generation sockets have handshaken.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    create_worker::{build_model_card, resolve_model_id, take_kv_transfer_metadata},
    discover_metadata::normalize_grpc_keys,
    SubmitTokenizerJobStep,
};
use crate::{
    routers::grpc::backend_client::{
        zmq_connect_timeout, ZMQ_CONNECT_TIMEOUT_LABEL, ZMQ_CONTROL_URL_LABEL,
    },
    worker::{BasicWorker, BasicWorkerBuilder, ConnectionMode, Worker},
    workflow::{data::WorkerWorkflowData, steps::UpdatePoliciesStep},
};

pub struct DiscoverZmqControlMetadataStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverZmqControlMetadataStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.connection_mode != Some(ConnectionMode::Zmq)
            || !context
                .data
                .config
                .labels
                .contains_key(ZMQ_CONTROL_URL_LABEL)
        {
            return Ok(StepResult::Skip);
        }
        let app = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let mut workers = context
            .data
            .actual_workers
            .clone()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("actual_workers".to_string()))?;
        // A metadata-only property update preserves the same generation and
        // control association. Follow its current registry handle instead of
        // abandoning discovery when the old immutable worker is replaced.
        for worker in &mut workers {
            let Some(current) = app.worker_registry.get_by_url(worker.url()) else {
                continue;
            };
            if !Arc::ptr_eq(&current, worker)
                && current.connection_mode() == worker.connection_mode()
                && current.metadata().spec.runtime_type == worker.metadata().spec.runtime_type
                && current.worker_type() == worker.worker_type()
                && serde_json::to_value(&current.metadata().spec.models).is_ok_and(
                    |current_models| {
                        serde_json::to_value(&worker.metadata().spec.models)
                            .is_ok_and(|previous_models| current_models == previous_models)
                    },
                )
                && current.metadata().spec.labels.get(ZMQ_CONTROL_URL_LABEL)
                    == worker.metadata().spec.labels.get(ZMQ_CONTROL_URL_LABEL)
            {
                for (key, value) in &current.metadata().spec.labels {
                    if worker.metadata().spec.labels.get(key) != Some(value) {
                        context
                            .data
                            .config
                            .labels
                            .insert(key.clone(), value.clone());
                    }
                }
                *worker = current;
            }
        }
        context.data.actual_workers = Some(workers.clone());
        let timeout = if context
            .data
            .config
            .labels
            .contains_key(ZMQ_CONNECT_TIMEOUT_LABEL)
        {
            zmq_connect_timeout(&context.data.config.labels).map_err(|message| {
                WorkflowError::StepFailed {
                    step_id: StepId::new("discover_zmq_control_metadata"),
                    message,
                }
            })?
        } else {
            Duration::from_secs(app.router_config.worker_startup_timeout_secs)
        };
        let started = *context
            .data
            .control_metadata_started
            .get_or_insert_with(Instant::now);
        let expired = started.elapsed() >= timeout;
        for index in 0..workers.len() {
            let worker = Arc::clone(&workers[index]);
            let Some(id) = app.worker_registry.get_id_by_url(worker.url()) else {
                continue;
            };
            if !app
                .worker_registry
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(&current, &worker))
            {
                continue; // A concurrent registration owns this URL now.
            }
            let observed_revision = worker.revision();
            let fail = |message: String| {
                if !app
                    .worker_registry
                    .get(&id)
                    .is_some_and(|current| Arc::ptr_eq(&current, &worker))
                    || worker.revision() != observed_revision
                {
                    return WorkflowError::StepFailed { step_id: StepId::new("discover_zmq_control_metadata"),
                        message: "worker changed during control discovery; retrying current property state".to_string() };
                }

                if expired {
                    app.worker_registry.transition_status_if_revision(
                        &id,
                        observed_revision,
                        WorkerStatus::Failed,
                    );
                }
                WorkflowError::StepFailed {
                    step_id: StepId::new("discover_zmq_control_metadata"),
                    message: if expired {
                        format!("Control metadata startup deadline elapsed: {message}")
                    } else {
                        message
                    },
                }
            };
            // Acquisition starts the detached handshake and returns unavailable
            // until it lands. Step retries never cancel that owning driver.
            let acquisition = worker.get_backend_client().await;
            let acquisition = if expired && acquisition.is_err() {
                // READY and storing the completed client are separate async
                // ticks. Give the final attempt a short, bounded handoff window
                // so it can read metadata from an engine that just became ready.
                tokio::time::timeout(Duration::from_secs(1), async {
                    loop {
                        let result = worker.get_backend_client().await;
                        if result.is_ok() {
                            break result;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .unwrap_or(acquisition)
            } else {
                acquisition
            };
            let client = acquisition
                .map_err(|e| fail(e.to_string()))?
                .ok_or_else(|| fail("ZMQ backend client unavailable".to_string()))?;
            let model_info = tokio::time::timeout(Duration::from_secs(3), client.get_model_info())
                .await
                .map_err(|_| fail("control model metadata RPC timed out".to_string()))?
                .map_err(|e| fail(e.to_string()))?;
            let server_info =
                tokio::time::timeout(Duration::from_secs(3), client.get_server_info())
                    .await
                    .map_err(|_| fail("control server metadata RPC timed out".to_string()))?
                    .map_err(|e| fail(e.to_string()))?;
            if !app
                .worker_registry
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(&current, &worker))
                || worker.revision() != observed_revision
            {
                return Err(fail("worker changed while reading metadata".to_string()));
            }
            let mut labels = worker.metadata().spec.labels.clone();
            labels.extend(model_info.to_labels());
            labels.extend(server_info.to_labels());
            normalize_grpc_keys(&mut labels);
            labels.extend(context.data.config.labels.clone()); // actual operator overrides win
            let mut spec = (*worker.metadata().spec).clone();
            let (connector, role, engine_id) = take_kv_transfer_metadata(&spec, &mut labels);
            spec.kv_connector = connector;
            spec.kv_role = role;
            spec.kv_engine_id = engine_id;
            if spec.bootstrap_port.is_none() {
                spec.bootstrap_port = labels
                    .get("disaggregation_bootstrap_port")
                    .and_then(|port| port.parse().ok());
            }
            let card = build_model_card(
                resolve_model_id(&context.data.config, &labels),
                &context.data.config,
                &labels,
                &app.router_config.model_aliases,
            );
            context.data.final_labels.clone_from(&labels);
            let previous = Arc::clone(&worker);
            let previous_revision = previous.revision();
            let mut builder = BasicWorkerBuilder::from_spec(spec)
                .model(card)
                .labels(labels)
                .health_config(worker.metadata().health_config.clone())
                .health_endpoint(worker.metadata().health_endpoint.clone())
                .resilience(worker.resilience().clone())
                .status(worker.status())
                .overload_defaults(worker.metadata().overload)
                .backend_client((*client).clone())
                .connect_signal_tx(app.worker_registry.connect_signal_sender());
            if let Some(basic) = worker.as_any().downcast_ref::<BasicWorker>() {
                builder = builder.circuit_breaker_config(basic.circuit_breaker_config());
            }
            let replacement: Arc<dyn Worker> = Arc::new(builder.build());
            let completion_sink = previous.completion_sink();
            replacement.set_completion_sink(completion_sink.clone());
            // Preserve live runtime state and the generation connection; this
            // metadata read must not bind another socket set or reset health.
            if app
                .worker_registry
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(&current, &previous))
                && app.worker_registry.replace_claiming_local(
                    &id,
                    Arc::clone(&replacement),
                    previous_revision,
                )
            {
                if previous.model_id() != replacement.model_id() {
                    app.policy_registry
                        .remove_worker_from_cache_aware(previous.model_id(), previous.url());
                }
                // Reconcile exactly one registration: removing then adding
                // keeps model counts stable for unchanged identities and retries.
                app.policy_registry.on_worker_removed(previous.model_id());
                workers[index] = Arc::clone(&replacement);
                context.data.actual_workers = Some(workers.clone());
                let mut policy_context = context.clone();
                policy_context.data.actual_workers = Some(vec![Arc::clone(&replacement)]);
                UpdatePoliciesStep.execute(&mut policy_context).await?;
                if completion_sink.is_some() {
                    replacement.set_completion_sink(completion_sink);
                }
            } else {
                return Err(fail(
                    "worker replaced before metadata could be published".to_string(),
                ));
            }
        }
        // The first tokenizer job may have had no source. Newly discovered
        // tokenizer/model paths can now load through the normal deduped flow.
        SubmitTokenizerJobStep.execute(context).await?;
        if let Some(app) = &context.data.app_context {
            for worker in context.data.actual_workers.as_deref().unwrap_or_default() {
                let Some(id) = app.worker_registry.get_id_by_url(worker.url()) else {
                    continue;
                };
                if !app
                    .worker_registry
                    .get(&id)
                    .is_some_and(|current| Arc::ptr_eq(&current, worker))
                {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("discover_zmq_control_metadata"),
                        message:
                            "worker changed before control metadata readiness publication; retrying"
                                .to_string(),
                    });
                }
                let revision = worker.revision();
                worker.mark_control_metadata_ready();
                app.worker_registry.transition_status_if_revision(
                    &id,
                    revision,
                    WorkerStatus::Ready,
                );
                if !app
                    .worker_registry
                    .get(&id)
                    .is_some_and(|current| Arc::ptr_eq(&current, worker))
                {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("discover_zmq_control_metadata"),
                        message: "worker changed during readiness publication; retrying"
                            .to_string(),
                    });
                }
            }
        }
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        !matches!(error, WorkflowError::StepFailed { message, .. }
            if message.starts_with("Control metadata startup deadline elapsed:"))
    }
}
