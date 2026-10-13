//! Step to update worker properties.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::create_worker::labels_for_rank;
use crate::{
    worker::{
        overload::OverloadThresholds, service::MAX_WORKER_LABELS, BasicWorker, BasicWorkerBuilder,
        Worker,
    },
    workflow::data::WorkerUpdateWorkflowData,
};

/// Step to update worker properties.
///
/// This step creates new worker instances with updated properties and
/// re-registers them to replace the old workers in the registry.
pub struct UpdateWorkerPropertiesStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateWorkerPropertiesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        // A label refresh replaces what discovery read before with what the
        // engine reports now (an operator's own labels keep their values) and
        // rebuilds a worker only when that changes something; an API update
        // merges the request's labels into the existing ones.
        let refresh = context.data.refresh_discovered_labels;
        let fresh_engine_labels = context.data.fresh_engine_labels.clone();
        let snapshot = context.data.snapshot.clone();
        if refresh && fresh_engine_labels.is_none() {
            debug!("No engine labels were re-read; nothing to update");
            context.data.updated_workers = Some(Vec::new());
            return Ok(StepResult::Skip);
        }
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let workers_to_update = context
            .data
            .workers_to_update
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_update".to_string()))?
            .clone();

        debug!(
            "Updating properties for {} worker(s)",
            workers_to_update.len()
        );

        let mut updated_workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_update.len());

        // The labels each worker ends up with: the request's merged into its
        // own. The admin check held the cap against the labels the worker had
        // when the request arrived; updates queued together each passed it,
        // so it is held again here, over the labels actually merged, and only
        // when the merge adds keys: a worker over the cap through the labels
        // discovered at registration still takes a rewrite of its keys and an
        // update of its other properties. Every rank is checked before any is
        // replaced, so a refusal leaves no partial update behind.
        let merged_labels: Vec<std::collections::HashMap<String, String>> = workers_to_update
            .iter()
            .map(|worker| {
                let existing = &worker.metadata().spec.labels;
                if let (true, Some(fresh)) = (refresh, fresh_engine_labels.as_ref()) {
                    return Ok(refreshed_labels(
                        existing,
                        &worker.metadata().discovered_labels,
                        &fresh_in_rank_space(worker, fresh),
                    ));
                }
                let mut updated_labels = existing.clone();
                if let Some(ref new_labels) = request.labels {
                    for (key, value) in new_labels {
                        updated_labels.insert(key.clone(), value.clone());
                    }
                }
                if updated_labels.len() > existing.len() && updated_labels.len() > MAX_WORKER_LABELS {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("update_worker_properties"),
                        message: format!(
                            "{} labels on worker {} after this update ({} now, {} new); at most {MAX_WORKER_LABELS} are accepted",
                            updated_labels.len(),
                            worker.url(),
                            existing.len(),
                            updated_labels.len() - existing.len()
                        ),
                    });
                }
                Ok(updated_labels)
            })
            .collect::<Result<_, _>>()?;

        for (index, (worker, updated_labels)) in
            workers_to_update.iter().zip(merged_labels).enumerate()
        {
            if refresh && updated_labels == worker.metadata().spec.labels {
                debug!(
                    worker_url = %worker.url(),
                    "Engine labels unchanged; the worker is kept as it is"
                );
                continue;
            }
            // A refresh follows the labels; the model card built from them at
            // registration (context length, model type, architectures) is not
            // re-derived here (a follow-up).
            let discovered_labels = match (refresh, fresh_engine_labels.as_ref()) {
                (true, Some(fresh)) => fresh_in_rank_space(worker, fresh),
                _ => (*worker.metadata().discovered_labels).clone(),
            };
            // Resolve priority and cost: use update value if specified, otherwise keep existing
            let updated_priority = request.priority.unwrap_or_else(|| worker.priority());
            let updated_cost = request.cost.unwrap_or_else(|| worker.cost());

            // Build updated health config from resolved runtime config
            let existing_health = &worker.metadata().health_config;
            let updated_health_config = match &request.health {
                Some(update) => update.apply_to(existing_health),
                None => existing_health.clone(),
            };
            let health_endpoint = worker.metadata().health_endpoint.clone();

            // Determine API key: use new one if provided, otherwise keep existing
            let updated_api_key = request
                .api_key
                .clone()
                .or_else(|| worker.metadata().spec.api_key.clone());

            // Keep the full spec so a metadata update preserves PD transport
            // and DP configuration.
            //
            // Status: a metadata-only update is not a re-registration — the
            // worker is the same endpoint. Preserve the old status so a
            // healthy worker stays routable through the update. The one
            // exception is when the update flips `disable_health_check` to
            // `true`: the health checker skips disabled workers, so a stale
            // Pending/NotReady status would never recover. Force Ready in
            // that case.
            let next_status = if updated_health_config.disable_health_check {
                WorkerStatus::Ready
            } else {
                worker.status()
            };
            // `from_spec` starts the rebuilt worker on the built-in breaker
            // config. Carry the one resolved at registration (gateway defaults
            // plus this worker's overrides) so an update neither loosens nor
            // tightens the breaker; with matching configs the registry
            // replacement adopts the live breaker as well, so its state
            // (open, half-open, failure counts) survives the update.
            let circuit_breaker_config = worker
                .as_any()
                .downcast_ref::<BasicWorker>()
                .map(BasicWorker::circuit_breaker_config);
            let mut builder = BasicWorkerBuilder::from_spec((*worker.metadata().spec).clone())
                .http2(worker.http2())
                .labels(updated_labels)
                .discovered_labels(discovered_labels)
                .health_config(updated_health_config.clone())
                .health_endpoint(&health_endpoint)
                .resilience(worker.resilience().clone())
                .priority(updated_priority)
                .cost(updated_cost)
                // The overload block is not updatable here, but it must survive
                // the rebuild; effective thresholds re-resolve against the same
                // gateway defaults registration used.
                .overload(worker.metadata().spec.overload)
                .overload_defaults(OverloadThresholds::from_gateway_config(
                    &app_context.router_config,
                ))
                .status(next_status);
            if let Some(config) = circuit_breaker_config {
                builder = builder.circuit_breaker_config(config);
            }

            // Adopt the old worker's client only if it was materialized: a
            // never-HTTP worker (ZMQ) keeps a deferred slot instead of having
            // a client it will never use built for the replacement.
            if let Some(client) = worker.http_client_handle_if_initialized() {
                builder = builder.http_client(client);
            }

            if let Some(ref api_key) = updated_api_key {
                builder = builder.api_key(api_key.clone());
            }

            // The spec retains the ZMQ address and engine count. The connect
            // signal still has to come from the registry.
            builder =
                builder.connect_signal_tx(app_context.worker_registry.connect_signal_sender());

            let mut new_worker = builder.build();
            if let Some(previous) = worker.as_any().downcast_ref::<BasicWorker>() {
                // A health probe may finish after this update. Share its state
                // so the result reaches the replacement too.
                new_worker.share_kv_engine_state(previous);
            }
            let new_worker: Arc<dyn Worker> = Arc::new(new_worker);

            // Replace the worker in the registry (overwrite-then-diff). A
            // refresh writes back a snapshot taken before a network read,
            // against the id and revision the find step captured: a worker
            // updated or replaced meanwhile fails the guard and is left as it
            // is, one removed (or removed and registered again at the same
            // address, under a new id) is not touched; the next transition
            // into service refreshes again.
            let worker_id = if refresh {
                match snapshot.as_ref().and_then(|entries| entries.get(index)) {
                    Some((id, expected))
                        if app_context.worker_registry.replace_claiming_local(
                            id,
                            new_worker.clone(),
                            *expected,
                        ) =>
                    {
                        id.clone()
                    }
                    _ => {
                        debug!(
                            worker_url = %worker.url(),
                            "Worker changed or left during the label refresh; not rebuilt"
                        );
                        continue;
                    }
                }
            } else {
                app_context
                    .worker_registry
                    .register_or_replace(new_worker.clone())
            };

            // Same-URL replace preserves the existing shared runtime. If the
            // update disables health checks, force the lifecycle to Ready so
            // the worker does not stay stuck in a non-routable pre-update
            // status while the manager correctly skips future probes.
            if updated_health_config.disable_health_check {
                let _ = app_context
                    .worker_registry
                    .transition_status(&worker_id, WorkerStatus::Ready);
            }

            updated_workers.push(new_worker);
        }

        // Log result
        if updated_workers.len() == 1 {
            info!("Updated worker {}", updated_workers[0].url());
        } else {
            info!("Updated {} workers", updated_workers.len());
        }

        // Store updated workers for subsequent steps
        context.data.updated_workers = Some(updated_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// The engine's fresh labels as `worker` records them: on an attention-DP
/// rank worker the window label is the rank's share, the way registration
/// recorded the discovered set (with no operator labels, so the operator's
/// own window still differs from the record and wins the merge below); any
/// other worker takes them as reported.
fn fresh_in_rank_space(
    worker: &Arc<dyn Worker>,
    fresh: &std::collections::HashMap<String, String>,
) -> std::collections::HashMap<String, String> {
    labels_for_rank(
        fresh,
        worker.dp_rank().zip(worker.dp_size()),
        &std::collections::HashMap::new(),
    )
}

/// The labels a worker carries once its engine reports `fresh`: every label
/// whose value came from discovery (`discovered_before`, still unchanged
/// since) follows the engine, disappearing when the engine no longer reports
/// it; a label the operator set or overrode (its value differs from what
/// discovery read, or discovery never read it) keeps its value; a key the
/// engine reports for the first time is added unless the operator already
/// set it.
pub(crate) fn refreshed_labels(
    current: &std::collections::HashMap<String, String>,
    discovered_before: &std::collections::HashMap<String, String>,
    fresh: &std::collections::HashMap<String, String>,
) -> std::collections::HashMap<String, String> {
    let mut labels = current.clone();
    for (key, before) in discovered_before {
        if current.get(key) != Some(before) {
            // The operator overrode (or removed) it: theirs stands.
            continue;
        }
        match fresh.get(key) {
            Some(now) => {
                labels.insert(key.clone(), now.clone());
            }
            None => {
                labels.remove(key);
            }
        }
    }
    for (key, now) in fresh {
        if !discovered_before.contains_key(key) && !current.contains_key(key) {
            labels.insert(key.clone(), now.clone());
        }
    }
    labels
}

#[cfg(test)]
mod refreshed_labels_tests {
    use std::collections::HashMap;

    use super::refreshed_labels;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn engine_sourced_labels_follow_the_engine() {
        let discovered = map(&[("max_num_seqs", "256"), ("page_size", "128")]);
        let current = map(&[
            ("max_num_seqs", "256"),
            ("page_size", "128"),
            ("tier", "gold"),
        ]);
        let fresh = map(&[
            ("max_num_seqs", "8"),
            ("page_size", "1152"),
            ("supports_vision", "true"),
        ]);
        assert_eq!(
            refreshed_labels(&current, &discovered, &fresh),
            map(&[
                ("max_num_seqs", "8"),
                ("page_size", "1152"),
                ("supports_vision", "true"),
                ("tier", "gold"),
            ]),
            "engine keys follow the engine, a new engine key is added, the operator's label stays"
        );
    }

    #[test]
    fn an_operator_override_keeps_its_value() {
        let discovered = map(&[("served_model_name", "/models/x"), ("max_num_seqs", "256")]);
        let current = map(&[("served_model_name", "m1"), ("max_num_seqs", "256")]);
        let fresh = map(&[("served_model_name", "/models/y"), ("max_num_seqs", "8")]);
        assert_eq!(
            refreshed_labels(&current, &discovered, &fresh),
            map(&[("served_model_name", "m1"), ("max_num_seqs", "8")]),
            "a value the operator set over discovery's is kept; the untouched one follows"
        );
    }

    #[test]
    fn a_label_the_engine_no_longer_reports_is_dropped_unless_the_operator_set_it() {
        let discovered = map(&[("mm_processor", "inprocess"), ("supports_vision", "true")]);
        let current = map(&[
            ("mm_processor", "inprocess"),
            ("supports_vision", "true"),
            ("extra", "1"),
        ]);
        let fresh = map(&[("supports_vision", "false")]);
        assert_eq!(
            refreshed_labels(&current, &discovered, &fresh),
            map(&[("supports_vision", "false"), ("extra", "1")]),
            "a vanished engine key goes, a flipped one follows, an operator key stays"
        );
    }

    #[test]
    fn a_new_engine_key_does_not_override_an_operator_label() {
        let discovered = HashMap::new();
        let current = map(&[("weight_version", "pinned")]);
        let fresh = map(&[("weight_version", "v7"), ("vocab_size", "32000")]);
        assert_eq!(
            refreshed_labels(&current, &discovered, &fresh),
            map(&[("weight_version", "pinned"), ("vocab_size", "32000")]),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response},
        EngineId,
    };
    use openai_protocol::worker::{
        HealthCheckConfig, HealthCheckUpdate, OverloadUpdate, WorkerSpec, WorkerType,
        WorkerUpdateRequest,
    };
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        middleware::AuthConfig,
        routers::{
            common::kv_transfer::{connector_mode_for_worker, KvConnectorMode},
            grpc::{
                backend_client::BackendClient,
                zmq_client::{EosTokenIds, ZmqEngineClient},
            },
        },
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            BasicWorker, ConnectionMode, RuntimeType,
        },
    };

    fn make_app_context(workers: &[Arc<dyn Worker>]) -> Arc<AppContext> {
        use crate::{
            config::RouterConfig,
            middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::{
                common::{openai_bridge, realtime::RealtimeRegistry},
                grpc::multimodal::MultimodalConfigRegistry,
            },
            worker::{WorkerRegistry, WorkerService},
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(Arc::clone(w)).unwrap();
        }
        let job_queue = Arc::new(std::sync::OnceLock::new());

        Arc::new(AppContext {
            gateway_auth: AuthConfig::new(None),
            client: crate::http_client::client(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            rate_limit_manager: None,
            worker_registry: Arc::clone(&registry),
            prefill_admission: None,
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                router_config.policy.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            gateway: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: Arc::clone(&job_queue),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            tokenizer_load_failures: Arc::default(),
            wasm_manager: None,
            worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(
                &router_config,
            )),
            worker_service: Arc::new(WorkerService::new(registry, job_queue, router_config)),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            rl: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    fn make_context(
        app_context: Arc<AppContext>,
        worker: Arc<dyn Worker>,
        labels: HashMap<String, String>,
    ) -> WorkflowContext<WorkerUpdateWorkflowData> {
        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: Some(labels),
                api_key: None,
                health: None,
            },
            worker_url: worker.url().to_string(),
            app_context: Some(app_context),
            workers_to_update: Some(vec![worker]),
            updated_workers: None,
            refresh_discovered_labels: false,
            fresh_engine_labels: None,
            snapshot: None,
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    fn make_refresh_context(
        app_context: Arc<AppContext>,
        worker: Arc<dyn Worker>,
        fresh: HashMap<String, String>,
    ) -> WorkflowContext<WorkerUpdateWorkflowData> {
        let snapshot = vec![(
            app_context
                .worker_registry
                .get_id_by_url(worker.url())
                .expect("the worker is registered"),
            worker.revision(),
        )];
        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: worker.url().to_string(),
            app_context: Some(app_context),
            workers_to_update: Some(vec![worker]),
            updated_workers: None,
            refresh_discovered_labels: true,
            fresh_engine_labels: Some(fresh),
            snapshot: Some(snapshot),
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    fn labels_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    /// A label refresh replaces what discovery read before with what the
    /// engine reports now, keeps the operator's labels and the worker's
    /// status, records the new discovered set, and lets the decode admission
    /// window follow; the same report again rebuilds nothing.
    #[tokio::test]
    async fn a_label_refresh_follows_the_engine_and_keeps_the_operators_labels() {
        let discovered = labels_of(&[("max_num_seqs", "256"), ("page_size", "128")]);
        let mut labels = discovered.clone();
        labels.insert("tier".to_string(), "gold".to_string());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:8000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels)
                .discovered_labels(discovered)
                .status(WorkerStatus::Ready)
                .build(),
        );
        assert_eq!(worker.max_running_requests(), Some(256));
        let app_ctx = make_app_context(std::slice::from_ref(&worker));

        let fresh = labels_of(&[
            ("max_num_seqs", "8"),
            ("page_size", "1152"),
            ("supports_vision", "true"),
        ]);
        let mut refresh =
            make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh.clone());
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut refresh)
                .await
                .unwrap(),
            StepResult::Success
        );
        let rebuilt = app_ctx
            .worker_registry
            .get_by_url("grpc://decode:8000")
            .expect("the worker stays registered");
        assert_eq!(
            rebuilt.metadata().spec.labels,
            labels_of(&[
                ("max_num_seqs", "8"),
                ("page_size", "1152"),
                ("supports_vision", "true"),
                ("tier", "gold"),
            ]),
            "engine labels follow the engine, the operator's label stays"
        );
        assert_eq!(*rebuilt.metadata().discovered_labels, fresh);
        assert_eq!(
            rebuilt.max_running_requests(),
            Some(8),
            "the decode admission window follows the engine"
        );
        assert_eq!(
            rebuilt.status(),
            WorkerStatus::Ready,
            "a refresh keeps the worker routable"
        );

        // The same report again changes nothing: no rebuild, no updated worker.
        let revision = rebuilt.revision();
        let mut again = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&rebuilt), fresh);
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut again)
                .await
                .unwrap(),
            StepResult::Success
        );
        assert_eq!(
            app_ctx
                .worker_registry
                .get_by_url("grpc://decode:8000")
                .unwrap()
                .revision(),
            revision,
            "an unchanged engine report rebuilds nothing"
        );
        assert!(again.data.updated_workers.as_ref().unwrap().is_empty());
    }

    /// On an attention-DP rank worker the window label is the rank's share,
    /// recorded in rank space at registration; a refresh maps the engine's
    /// fresh labels through the same rule, so the share follows the engine's
    /// new window, and an operator's hand-set window still wins.
    #[tokio::test]
    async fn a_rank_workers_window_follows_the_engine_in_rank_space() {
        // The engine runs attention DP over 8 ranks with a window of 256: a
        // rank worker carries the share, 32, and records it in rank space.
        let discovered = labels_for_rank(
            &labels_of(&[("max_running_requests", "256"), ("attn_dp_size", "8")]),
            Some((0, 8)),
            &HashMap::new(),
        );
        assert_eq!(
            discovered.get("max_running_requests").map(String::as_str),
            Some("32")
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:8000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .dp_config(0, 8)
                .labels(discovered.clone())
                .discovered_labels(discovered.clone())
                .status(WorkerStatus::Ready)
                .build(),
        );
        assert_eq!(worker.max_running_requests(), Some(32));
        // The operator pinned another rank's window by hand: registration kept
        // only the hand-set key (`labels_for_rank` with the operator's labels).
        let pinned: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:8000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .dp_config(1, 8)
                .labels(labels_of(&[("max_num_seqs", "16"), ("attn_dp_size", "8")]))
                .discovered_labels(discovered)
                .status(WorkerStatus::Ready)
                .build(),
        );
        assert_eq!(pinned.max_running_requests(), Some(16));
        let app_ctx = make_app_context(&[Arc::clone(&worker), Arc::clone(&pinned)]);

        // The engine restarts with a window of 512: the share is 64.
        let fresh = labels_of(&[("max_running_requests", "512"), ("attn_dp_size", "8")]);
        for rank in [&worker, &pinned] {
            let mut refresh =
                make_refresh_context(Arc::clone(&app_ctx), Arc::clone(rank), fresh.clone());
            assert_eq!(
                UpdateWorkerPropertiesStep
                    .execute(&mut refresh)
                    .await
                    .unwrap(),
                StepResult::Success
            );
        }
        let rebuilt = app_ctx
            .worker_registry
            .get_by_url(worker.url())
            .expect("the rank stays registered");
        assert_eq!(
            rebuilt.max_running_requests(),
            Some(64),
            "the rank's share follows the engine's new window"
        );
        assert_eq!(
            rebuilt
                .metadata()
                .discovered_labels
                .get("max_running_requests")
                .map(String::as_str),
            Some("64"),
            "the discovered record stays in rank space"
        );
        let pinned_now = app_ctx
            .worker_registry
            .get_by_url(pinned.url())
            .expect("the pinned rank stays registered");
        assert_eq!(
            pinned_now.max_running_requests(),
            Some(16),
            "the operator's hand-set window wins over the engine's share"
        );
        assert_eq!(
            pinned_now.revision(),
            pinned.revision(),
            "nothing to follow on the pinned rank: it is not rebuilt"
        );
    }

    /// A rank registered from a provisional read (no attention-DP label yet)
    /// carries the whole window; once the engine's answer names the width,
    /// the refresh moves the rank to its share instead of the whole figure.
    #[tokio::test]
    async fn a_rank_registered_from_a_provisional_read_takes_its_share_on_refresh() {
        let discovered = labels_of(&[("max_running_requests", "256")]);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:8001")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .dp_config(3, 8)
                .labels(discovered.clone())
                .discovered_labels(discovered)
                .status(WorkerStatus::Ready)
                .build(),
        );
        assert_eq!(worker.max_running_requests(), Some(256));
        let app_ctx = make_app_context(std::slice::from_ref(&worker));

        let fresh = labels_of(&[("max_running_requests", "256"), ("attn_dp_size", "8")]);
        let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh);
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut refresh)
                .await
                .unwrap(),
            StepResult::Success
        );
        let rebuilt = app_ctx
            .worker_registry
            .get_by_url(worker.url())
            .expect("the rank stays registered");
        assert_eq!(
            rebuilt.max_running_requests(),
            Some(32),
            "the rank takes its share once the engine names its width"
        );
        assert_eq!(
            rebuilt
                .metadata()
                .spec
                .labels
                .get("attn_dp_size")
                .map(String::as_str),
            Some("8")
        );
    }

    /// A refresh writes back against the revision it read: a worker that
    /// changed during the read (an update, a replacement) is left as it is,
    /// and one that was removed is not re-added.
    #[tokio::test]
    async fn a_label_refresh_is_dropped_when_the_worker_changed_or_left_during_the_read() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode-b:8000")
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels_of(&[("max_num_seqs", "256")]))
                .discovered_labels(labels_of(&[("max_num_seqs", "256")]))
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let fresh = labels_of(&[("max_num_seqs", "8")]);

        // Changed during the read: the snapshot's revision is stale.
        let mut refresh =
            make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&worker), fresh.clone());
        // An operator update lands while the read is in flight: the registry
        // replaces the worker and bumps its revision.
        let id = app_ctx
            .worker_registry
            .get_id_by_url("grpc://decode-b:8000")
            .unwrap();
        let patched: Arc<dyn Worker> = Arc::new(
            worker
                .as_any()
                .downcast_ref::<BasicWorker>()
                .expect("a BasicWorker")
                .clone(),
        );
        assert!(app_ctx.worker_registry.replace(&id, patched));
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut refresh)
                .await
                .unwrap(),
            StepResult::Success
        );
        assert_eq!(
            app_ctx
                .worker_registry
                .get_by_url("grpc://decode-b:8000")
                .unwrap()
                .max_running_requests(),
            Some(256),
            "a stale snapshot is not written back"
        );
        assert!(refresh.data.updated_workers.as_ref().unwrap().is_empty());

        // Removed during the read, and registered again at the same address:
        // the refresh must neither bring the old worker back nor touch the new
        // one (a fresh registration starts at the revision the snapshot had).
        let current = app_ctx
            .worker_registry
            .get_by_url("grpc://decode-b:8000")
            .unwrap();
        let mut refresh = make_refresh_context(Arc::clone(&app_ctx), Arc::clone(&current), fresh);
        let id = app_ctx
            .worker_registry
            .get_id_by_url("grpc://decode-b:8000")
            .unwrap();
        app_ctx.worker_registry.remove(&id);
        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode-b:8000")
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels_of(&[("max_num_seqs", "999")]))
                .status(WorkerStatus::Ready)
                .build(),
        );
        app_ctx.worker_registry.register(replacement).unwrap();
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut refresh)
                .await
                .unwrap(),
            StepResult::Success
        );
        assert_eq!(
            app_ctx
                .worker_registry
                .get_by_url("grpc://decode-b:8000")
                .unwrap()
                .max_running_requests(),
            Some(999),
            "a worker registered again at the address keeps its own labels"
        );
        assert!(refresh.data.updated_workers.as_ref().unwrap().is_empty());
    }

    /// The cap is held where the labels are merged, not only at the admin
    /// check: a worker already at the cap refuses an update with a new key
    /// (updates queued together each passed the early check), while one that
    /// rewrites keys the worker has goes through.
    #[tokio::test]
    async fn the_label_cap_holds_over_the_labels_the_step_merges() {
        let mut builder = BasicWorkerBuilder::new("http://full:8000").status(WorkerStatus::Ready);
        for i in 0..MAX_WORKER_LABELS {
            builder = builder.label(format!("k{i}"), "v");
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut over = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("k_new".to_string(), "v".to_string())]),
        );
        let err = UpdateWorkerPropertiesStep
            .execute(&mut over)
            .await
            .expect_err("a 129th label at merge time");
        assert!(err.to_string().contains("129 labels on worker"), "{err}");
        let mut rewrite = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("k0".to_string(), "w".to_string())]),
        );
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut rewrite)
                .await
                .unwrap(),
            StepResult::Success
        );
    }

    #[tokio::test]
    async fn pd_update_preserves_connector_and_pair_descriptor() {
        for (connector, mode, protocol, transport, pair_key, expected_mode) in [
            (
                "MooncakeConnector",
                ConnectionMode::Http,
                None,
                "mooncake",
                "vllm/mooncake/page=16",
                KvConnectorMode::Mooncake {
                    host: "prefill".to_string(),
                    port: 9123,
                    engine_id: Some("engine-1_dp1".to_string()),
                },
            ),
            (
                "NixlConnector",
                ConnectionMode::Grpc,
                Some("cluster-blue"),
                "nixl",
                "cluster-blue",
                KvConnectorMode::Nixl,
            ),
        ] {
            let mut spec = WorkerSpec::new("http://prefill:8000");
            spec.worker_type = WorkerType::Prefill;
            spec.runtime_type = RuntimeType::Vllm;
            spec.connection_mode = mode;
            spec.kv_connector = Some(connector.to_string());
            spec.kv_role = Some("kv_producer".to_string());
            spec.kv_engine_id = Some("engine-1".to_string());
            spec.bootstrap_port = Some(9123);
            spec.pairing_protocol = protocol.map(str::to_string);
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::from_spec(spec)
                    .dp_config(1, 2)
                    .label("block_size", "16")
                    .label("tier", "silver")
                    .priority(2)
                    .cost(1.5)
                    .health_config(HealthCheckConfig {
                        timeout_secs: 3,
                        check_interval_secs: 11,
                        ..Default::default()
                    })
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            let app_ctx = make_app_context(std::slice::from_ref(&worker));
            let mut ctx = make_context(
                Arc::clone(&app_ctx),
                Arc::clone(&worker),
                HashMap::from([("tier".to_string(), "gold".to_string())]),
            );
            ctx.data.config.priority = Some(9);
            ctx.data.config.health = Some(HealthCheckUpdate {
                timeout_secs: Some(7),
                ..Default::default()
            });

            assert_eq!(
                UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap(),
                StepResult::Success
            );

            let updated = app_ctx
                .worker_registry
                .get_by_url("http://prefill:8000@1")
                .expect("worker stays registered");
            assert!(Arc::ptr_eq(
                &updated,
                &ctx.data.updated_workers.as_ref().unwrap()[0]
            ));
            assert_eq!(updated.base_url(), "http://prefill:8000");
            assert_eq!(updated.url(), "http://prefill:8000@1");
            assert_eq!(updated.dp_rank(), Some(1));
            assert_eq!(updated.dp_size(), Some(2));
            assert_eq!(
                updated.metadata().spec.kv_role.as_deref(),
                Some("kv_producer")
            );
            assert_eq!(connector_mode_for_worker(updated.as_ref()), expected_mode);
            assert_eq!(updated.pd_pairing().transport(), Some(transport));
            assert_eq!(updated.pd_pairing().explicit(), protocol);
            assert_eq!(updated.pd_pairing().key(), pair_key);
            assert_eq!(updated.metadata().spec.labels["tier"], "gold");
            assert_eq!(updated.metadata().spec.labels["block_size"], "16");
            assert_eq!(updated.priority(), 9);
            assert_eq!(updated.cost(), 1.5);
            assert_eq!(updated.metadata().health_config.timeout_secs, 7);
            assert_eq!(updated.metadata().health_config.check_interval_secs, 11);
            assert_eq!(updated.status(), WorkerStatus::Ready);
        }
    }

    /// A PATCH rebuilds the worker. The breaker it was registered with
    /// (gateway defaults plus overrides) and the breaker's live state must
    /// come through; a rebuild on the built-in defaults would silently
    /// re-arm a worker the breaker had taken out of rotation.
    #[tokio::test]
    async fn update_keeps_the_circuit_breaker_config_and_state() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_secs(7),
            window_duration: Duration::from_secs(9),
        };
        assert_ne!(config, CircuitBreakerConfig::default());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8000")
                .worker_type(WorkerType::Regular)
                .circuit_breaker_config(config.clone())
                .status(WorkerStatus::Ready)
                .build(),
        );
        worker.record_circuit_breaker_outcome(false);
        worker.record_circuit_breaker_outcome(false);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);

        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());
        UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

        let updated = &ctx.data.updated_workers.as_ref().unwrap()[0];
        let basic = updated.as_any().downcast_ref::<BasicWorker>().unwrap();
        assert_eq!(basic.circuit_breaker_config(), config);
        assert_eq!(updated.circuit_breaker_state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn pd_update_retains_live_kv_engine_state() {
        for current_id in [Some("engine-current"), None] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("grpc://prefill:8000")
                    .worker_type(WorkerType::Prefill)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Vllm)
                    .kv_connector("MooncakeConnector")
                    .kv_engine_id("engine-initial")
                    .bootstrap_port(Some(9123))
                    .status(WorkerStatus::Ready)
                    .build(),
            );
            worker.refresh_kv_engine_id(current_id.map(str::to_string));
            worker.set_kv_engine_id_confirmed(false);
            let app_ctx = make_app_context(std::slice::from_ref(&worker));
            let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

            UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

            let updated = &ctx.data.updated_workers.as_ref().unwrap()[0];
            assert_eq!(updated.kv_engine_id().as_deref(), current_id);
            assert!(!updated.kv_engine_id_confirmed());

            // A probe that began before the update still holds the old worker.
            worker.refresh_kv_engine_id(Some("engine-late".to_string()));
            worker.set_kv_engine_id_confirmed(true);
            assert_eq!(updated.kv_engine_id().as_deref(), Some("engine-late"));
            assert!(updated.kv_engine_id_confirmed());
            assert_eq!(
                connector_mode_for_worker(updated.as_ref()),
                KvConnectorMode::Mooncake {
                    host: "prefill".to_string(),
                    port: 9123,
                    engine_id: Some("engine-late".to_string()),
                }
            );
        }
    }

    /// A label-only PATCH must not disturb a ZMQ worker's transport: the engine
    /// handshakes once at its own startup, so losing the connected client (or
    /// the addresses and engine count that define the sockets) would black-hole
    /// every request until the engine group is restarted.
    #[tokio::test]
    async fn zmq_update_preserves_transport_state_and_the_live_client() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let _engine = engine.expect("mock engine");
        let client = Arc::new(BackendClient::Zmq(client.expect("adapter connect")));

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(ep("ts0.ipc"))
                .connection_mode(ConnectionMode::Zmq)
                .runtime_type(RuntimeType::Vllm)
                .zmq_handshake_address(handshake.clone())
                .zmq_engine_group(2)
                .health_config(HealthCheckConfig::default())
                .overload(OverloadUpdate {
                    waiting_requests: Some(8),
                    token_usage: None,
                })
                .status(WorkerStatus::Ready)
                .build(),
        );
        let old = worker
            .as_any()
            .downcast_ref::<BasicWorker>()
            .expect("BasicWorker");
        old.backend_client
            .load_full()
            .set(Arc::clone(&client))
            .ok()
            .expect("cell empty");

        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("tier".to_string(), "gold".to_string())]),
        );

        let result = UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();
        assert_eq!(result, StepResult::Success);

        let updated = ctx.data.updated_workers.as_ref().expect("updated workers");
        assert_eq!(updated.len(), 1);
        let new = updated[0]
            .as_any()
            .downcast_ref::<BasicWorker>()
            .expect("BasicWorker");

        assert_eq!(
            new.metadata().spec.labels.get("tier"),
            Some(&"gold".to_string())
        );
        assert_eq!(
            new.metadata().spec.zmq_handshake_address.as_deref(),
            Some(handshake.as_str())
        );
        assert_eq!(new.metadata().zmq_engine_count(), 2);
        // The overload block survives a metadata-only update, and so does the
        // effective threshold resolved from it.
        assert_eq!(new.metadata().spec.overload.waiting_requests, Some(8));
        assert_eq!(new.metadata().overload.waiting_requests, Some(8));
        assert!(
            new.connect_signal_tx.is_some(),
            "ZMQ promotion stays event-driven after an update"
        );

        let adopted = new
            .backend_client
            .load_full()
            .get()
            .cloned()
            .expect("replacement must inherit the connected client");
        assert!(Arc::ptr_eq(&adopted, &client));
        assert_eq!(new.status(), WorkerStatus::Ready);

        // Neither side of a ZMQ replacement may materialize an HTTP client
        // it will never use.
        assert!(old.http_client.cell_is_empty());
        assert!(new.http_client.cell_is_empty());
    }

    /// An HTTP worker's replacement must adopt the same shared client — a
    /// fresh client would silently detach it from the worker-client cache
    /// entry the old worker was keeping alive.
    #[tokio::test]
    async fn http_update_adopts_the_materialized_shared_client() {
        let client = Arc::new(crate::http_client::client());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .http_client(Arc::clone(&client))
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

        let result = UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();
        assert_eq!(result, StepResult::Success);

        let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
        let adopted = updated
            .http_client_handle_if_initialized()
            .expect("materialized client is adopted");
        assert!(Arc::ptr_eq(&adopted, &client));
    }

    /// The adopted client speaks one HTTP version; the replacement must keep
    /// describing it correctly.
    #[tokio::test]
    async fn http_update_preserves_http2() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .http_client(Arc::new(crate::http_client::client()))
                .http2(true)
                .status(WorkerStatus::Ready)
                .build(),
        );
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut ctx = make_context(app_ctx, Arc::clone(&worker), HashMap::new());

        UpdateWorkerPropertiesStep.execute(&mut ctx).await.unwrap();

        let updated = &ctx.data.updated_workers.as_ref().expect("updated workers")[0];
        assert!(updated.http2());
    }

    /// A worker over the cap through the labels discovered at registration
    /// still takes an update that adds no key: a priority alone, or a rewrite
    /// of keys it has; one that would add a key is refused.
    #[tokio::test]
    async fn a_worker_over_the_cap_takes_updates_that_add_no_key() {
        let mut builder = BasicWorkerBuilder::new("http://over:8000").status(WorkerStatus::Ready);
        for i in 0..MAX_WORKER_LABELS + 20 {
            builder = builder.label(format!("k{i}"), "v");
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        let app_ctx = make_app_context(std::slice::from_ref(&worker));
        let mut priority = make_context(Arc::clone(&app_ctx), Arc::clone(&worker), HashMap::new());
        priority.data.config.labels = None;
        priority.data.config.priority = Some(7);
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut priority)
                .await
                .unwrap(),
            StepResult::Success
        );
        let mut rewrite = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("k3".to_string(), "w".to_string())]),
        );
        assert_eq!(
            UpdateWorkerPropertiesStep
                .execute(&mut rewrite)
                .await
                .unwrap(),
            StepResult::Success
        );
        let mut adds = make_context(
            Arc::clone(&app_ctx),
            Arc::clone(&worker),
            HashMap::from([("k_new".to_string(), "v".to_string())]),
        );
        let err = UpdateWorkerPropertiesStep
            .execute(&mut adds)
            .await
            .expect_err("a new key on a worker over the cap");
        assert!(
            err.to_string().contains("149 labels on worker") && err.to_string().contains("1 new"),
            "{err}"
        );
    }

    /// Every rank is checked before any is replaced: when the second of two
    /// ranks refuses the merge, the first keeps its labels too.
    #[tokio::test]
    async fn a_refused_rank_leaves_no_partial_update_behind() {
        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://rank:8000@0")
                .status(WorkerStatus::Ready)
                .label("k0", "v")
                .build(),
        );
        let mut builder = BasicWorkerBuilder::new("http://rank:8000@1").status(WorkerStatus::Ready);
        for i in 0..MAX_WORKER_LABELS {
            builder = builder.label(format!("k{i}"), "v");
        }
        let second: Arc<dyn Worker> = Arc::new(builder.build());
        let app_ctx = make_app_context(&[Arc::clone(&first), Arc::clone(&second)]);
        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: Some(HashMap::from([("k_new".to_string(), "v".to_string())])),
                api_key: None,
                health: None,
            },
            worker_url: "http://rank:8000".to_string(),
            app_context: Some(Arc::clone(&app_ctx)),
            workers_to_update: Some(vec![Arc::clone(&first), Arc::clone(&second)]),
            updated_workers: None,
            refresh_discovered_labels: false,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        let err = UpdateWorkerPropertiesStep
            .execute(&mut context)
            .await
            .expect_err("the second rank refuses the merge");
        assert!(err.to_string().contains("http://rank:8000@1"), "{err}");
        let kept = app_ctx
            .worker_registry
            .get_by_url("http://rank:8000@0")
            .expect("the first rank is still registered");
        assert!(
            !kept.metadata().spec.labels.contains_key("k_new"),
            "the first rank was not updated before the second was checked"
        );
    }
}
