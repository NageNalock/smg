//! Loopback control/data routing checks without an engine model.

use std::{
    pin::Pin,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use axum::{body::to_bytes, response::IntoResponse};
use engine_zmq_client::{
    codec::decode_msgpack,
    mock_engine::{connect_to_frontend, default_ready_response, hello},
    protocol::tokenspeed::request::TokenizedGenerateReqInput,
    EngineId,
};
use futures::Stream;
use smg_grpc_client::tokenspeed_proto::token_speed_scheduler_server::{
    TokenSpeedScheduler, TokenSpeedSchedulerServer,
};
use tokio::{net::TcpListener, sync::Notify};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

use super::*;
use crate::worker::{
    worker::RequestCompletionSink, BasicWorkerBuilder, Worker, WorkerLoadGuard, WorkerManager,
    WorkerRegistry,
};

struct ControlStub {
    flushes: Arc<AtomicUsize>,
    generates: Arc<AtomicUsize>,
    model_reads: Arc<AtomicUsize>,
    model_failure: Option<Arc<DeferredModelFailure>>,
}
type StubStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for ControlStub {
    type GenerateStream = StubStream<tokenspeed_proto::GenerateResponse>;
    type SubscribeKvEventsStream = StubStream<common_proto::KvEventBatch>;
    type GetTokenizerStream = StubStream<common_proto::GetTokenizerChunk>;

    async fn generate(
        &self,
        _: Request<tokenspeed_proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.generates.fetch_add(1, Ordering::SeqCst);
        Err(Status::internal("generation must use direct ZMQ"))
    }
    async fn flush_cache(
        &self,
        req: Request<common_proto::FlushCacheRequest>,
    ) -> Result<Response<common_proto::FlushCacheResponse>, Status> {
        assert_eq!(req.into_inner().timeout_s, 0.0);
        self.flushes.fetch_add(1, Ordering::SeqCst);
        Ok(Response::new(common_proto::FlushCacheResponse {
            success: true,
            message: "control cache flushed".to_string(),
        }))
    }
    async fn health_check(
        &self,
        _: Request<tokenspeed_proto::HealthCheckRequest>,
    ) -> Result<Response<tokenspeed_proto::HealthCheckResponse>, Status> {
        Ok(Response::new(tokenspeed_proto::HealthCheckResponse {
            healthy: true,
            message: "ready".to_string(),
        }))
    }
    async fn get_model_info(
        &self,
        _: Request<tokenspeed_proto::GetModelInfoRequest>,
    ) -> Result<Response<tokenspeed_proto::GetModelInfoResponse>, Status> {
        self.model_reads.fetch_add(1, Ordering::SeqCst);
        if let Some(failure) = &self.model_failure {
            if failure.enabled.load(Ordering::SeqCst) {
                failure.started.notify_one();
                failure.release.notified().await;
                return Err(Status::unavailable("delayed control failure"));
            }
        }
        Ok(Response::new(tokenspeed_proto::GetModelInfoResponse {
            served_model_name: "m".to_string(),
            max_context_length: 8192,
            default_sampling_params_json: "{\"temperature\":0.2}".to_string(),
            ..Default::default()
        }))
    }
    async fn get_server_info(
        &self,
        _: Request<tokenspeed_proto::GetServerInfoRequest>,
    ) -> Result<Response<tokenspeed_proto::GetServerInfoResponse>, Status> {
        Ok(Response::new(tokenspeed_proto::GetServerInfoResponse {
            tokenspeed_version: "control-version".to_string(),
            ..Default::default()
        }))
    }
    async fn abort(
        &self,
        _: Request<tokenspeed_proto::AbortRequest>,
    ) -> Result<Response<tokenspeed_proto::AbortResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_loads(
        &self,
        _: Request<tokenspeed_proto::GetLoadsRequest>,
    ) -> Result<Response<tokenspeed_proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn subscribe_kv_events(
        &self,
        _: Request<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn start_profile(
        &self,
        _: Request<common_proto::StartProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn stop_profile(
        &self,
        _: Request<common_proto::StopProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_tokenizer(
        &self,
        _: Request<common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused"))
    }
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test owns and aborts its loopback service task"
)]
async fn tokenspeed_zmq_flush_uses_control_while_generate_uses_data_socket() {
    let dir = tempfile::tempdir().unwrap();
    let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    let (handshake, input, output) = (ep("handshake"), ep("input"), ep("output"));
    let (data, engine) = tokio::join!(
        ZmqEngineClient::connect(
            &handshake,
            &input,
            &output,
            1,
            "m".to_string(),
            zmq_client::EosTokenIds::default(),
            RuntimeType::TokenSpeed,
            Duration::from_secs(5)
        ),
        connect_to_frontend(
            &handshake,
            EngineId::from_engine_index(0),
            default_ready_response()
        ),
    );
    let mut engine = engine.unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("grpc://{}", listener.local_addr().unwrap());
    let flushes = Arc::new(AtomicUsize::new(0));
    let generates = Arc::new(AtomicUsize::new(0));
    let stub = ControlStub {
        flushes: Arc::clone(&flushes),
        generates: Arc::clone(&generates),
        model_reads: Arc::new(AtomicUsize::new(0)),
        model_failure: None,
    };
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::new(stub))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let backend =
        BackendClient::Zmq(data.unwrap().into()).with_zmq_control_endpoint(Some(endpoint.clone()));
    let worker = BasicWorkerBuilder::new(ep("worker"))
        .connection_mode(ConnectionMode::Zmq)
        .runtime_type(RuntimeType::TokenSpeed)
        .label(ZMQ_CONTROL_URL_LABEL, &endpoint)
        .backend_client(backend.clone())
        .build();
    let registry = WorkerRegistry::new();
    let worker_url = worker.url().to_string();
    registry.register(Arc::new(worker)).unwrap();
    let flush = WorkerManager::flush_cache_all(&registry).await;
    assert_eq!(flush.successful, vec![worker_url]);
    assert!(flush.failed.is_empty());
    assert_eq!(flush.zmq_workers, 1);
    assert_eq!(flushes.load(Ordering::SeqCst), 1);
    let response = flush.into_response();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["total_zmq_workers"], 1);
    assert_eq!(body["total_zmq_workers_skipped"], 0);
    assert_eq!(body["workers_flushed"], 1);
    let labels = backend.get_model_info().await.unwrap().to_labels();
    assert_eq!(labels["max_context_length"], "8192");
    let mut backend = backend;
    let request = ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
        request_id: "direct-only".to_string(),
        tokenized: Some(tokenspeed_proto::TokenizedInput {
            input_ids: vec![1],
            ..Default::default()
        }),
        sampling_params: Some(tokenspeed_proto::SamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    }));
    let (stream, frames) = tokio::join!(backend.generate(request), engine.recv_request());
    let stream = stream.unwrap();
    let frames = frames.unwrap();
    let request: TokenizedGenerateReqInput = decode_msgpack(&frames[1]).unwrap();
    assert_eq!(request.rid, "direct-only");
    assert_eq!(generates.load(Ordering::SeqCst), 0);
    drop(stream);
    server.abort();
}

#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "test owns and aborts its loopback service task"
)]
async fn control_metadata_waits_for_handshake_and_preserves_generation_connection() {
    use std::sync::OnceLock;

    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{model_card::ModelCard, worker::WorkerSpec};
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

    use crate::{
        app_context::AppContext,
        config::RouterConfig,
        policies::PolicyRegistry,
        worker::WorkerRegistry,
        workflow::{
            data::{WorkerKind, WorkerRegistrationMode},
            steps::{create_worker_workflow_data, local::DiscoverZmqControlMetadataStep},
        },
    };

    let dir = tempfile::tempdir().unwrap();
    let worker_url = format!("ipc://{}", dir.path().join("worker").display());
    let handshake_port = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let handshake = format!("tcp://127.0.0.1:{handshake_port}");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("grpc://{}", listener.local_addr().unwrap());
    let reads = Arc::new(AtomicUsize::new(0));
    let server_reads = Arc::clone(&reads);
    let failure = Arc::new(DeferredModelFailure::default());
    let server_failure = Arc::clone(&failure);
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::new(ControlStub {
                flushes: Arc::new(AtomicUsize::new(0)),
                generates: Arc::new(AtomicUsize::new(0)),
                model_reads: server_reads,
                model_failure: Some(server_failure),
            }))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let config = RouterConfig {
        disable_tokenizer_autoload: true,
        ..Default::default()
    };
    let registry = Arc::new(WorkerRegistry::new());
    let app = Arc::new(
        AppContext::builder()
            .client(crate::http_client::client())
            .rate_limiter(None)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(Arc::clone(&registry))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .router_config(config)
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_monitor(None)
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(Arc::new(OnceLock::new()))
            .build()
            .unwrap(),
    );
    let mut spec = WorkerSpec::new(&worker_url);
    spec.connection_mode = ConnectionMode::Zmq;
    spec.runtime_type = RuntimeType::TokenSpeed;
    spec.zmq_handshake_address = Some(handshake.clone());
    spec.labels
        .insert(ZMQ_CONTROL_URL_LABEL.to_string(), endpoint);
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec(spec.clone())
            .model(ModelCard::new("provisional-model-path"))
            .label("default_sampling_params_json", "{\"temperature\":0.7}")
            .build(),
    );
    let completions = Arc::new(CompletionSpy::default());
    worker.set_completion_sink(Some(
        Arc::clone(&completions) as Arc<dyn RequestCompletionSink>
    ));
    registry.register(Arc::clone(&worker)).unwrap();
    app.policy_registry
        .on_worker_added("provisional-model-path", None);
    let policy_registry = Arc::clone(&app.policy_registry);
    let mut data = create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, app);
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Zmq);
    data.actual_workers = Some(vec![Arc::clone(&worker)]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    assert!(DiscoverZmqControlMetadataStep
        .execute(&mut context)
        .await
        .is_err());
    assert_eq!(
        reads.load(Ordering::SeqCst),
        0,
        "control RPCs must wait for ZMQ READY"
    );
    assert_eq!(
        worker.status(),
        openai_protocol::worker::WorkerStatus::Pending
    );
    let engine_handshake = hello(&handshake, EngineId::from_engine_index(0))
        .await
        .unwrap();
    let updated: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec((*worker.metadata().spec).clone())
            .priority(7)
            .label("operator_update", "kept")
            .build(),
    );
    updated.set_completion_sink(worker.completion_sink());
    let id = registry.get_id_by_url(&worker_url).unwrap();
    assert!(registry.replace(&id, updated));
    // Simulate a late READY beyond the worker's control-discovery budget. The
    // final attempt must still reconcile metadata if the peer now answers.
    context
        .data
        .config
        .labels
        .insert(ZMQ_CONNECT_TIMEOUT_LABEL.to_string(), "1".to_string());
    context.data.control_metadata_started =
        Some(std::time::Instant::now() - Duration::from_secs(2));
    let mut engine = engine_handshake
        .ready(default_ready_response())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if DiscoverZmqControlMetadataStep
                .execute(&mut context)
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    // Repeating a completed metadata read must not double the registered
    // model's policy count or lose the request-completion callback.
    DiscoverZmqControlMetadataStep
        .execute(&mut context)
        .await
        .unwrap();
    let worker = registry.get_by_url(&worker_url).unwrap();
    assert!(worker.control_metadata_ready());
    assert_eq!(worker.priority(), 7);
    assert_eq!(worker.metadata().spec.labels["operator_update"], "kept");
    assert_eq!(
        worker.status(),
        openai_protocol::worker::WorkerStatus::Ready
    );
    assert_eq!(worker.metadata().spec.labels["max_context_length"], "8192");
    assert_eq!(
        worker.model_id(),
        "m",
        "discovered served model replaces only provisional startup identity"
    );
    assert_eq!(worker.context_length("m"), Some(8192));
    assert_eq!(
        worker.metadata().spec.labels["default_sampling_params_json"],
        "{\"temperature\":0.2}"
    );
    let guard = WorkerLoadGuard::with_key(Arc::clone(&worker), None);
    drop(guard);
    assert_eq!(completions.0.load(Ordering::SeqCst), 1);
    let mut client = (*worker.get_backend_client().await.unwrap().unwrap()).clone();
    let request = ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
        request_id: "after-metadata".to_string(),
        tokenized: Some(tokenspeed_proto::TokenizedInput {
            input_ids: vec![1],
            ..Default::default()
        }),
        sampling_params: Some(tokenspeed_proto::SamplingParams {
            max_new_tokens: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    }));
    let (stream, frames) = tokio::join!(client.generate(request), engine.recv_request());
    let request: TokenizedGenerateReqInput = decode_msgpack(&frames.unwrap()[1]).unwrap();
    assert_eq!(request.rid, "after-metadata");
    drop(stream.unwrap());
    failure.enabled.store(true, Ordering::SeqCst);
    let mut stale_context = context.clone();
    let discovery = tokio::spawn(async move {
        DiscoverZmqControlMetadataStep
            .execute(&mut stale_context)
            .await
    });
    tokio::time::timeout(Duration::from_secs(3), failure.started.notified())
        .await
        .unwrap();
    let current = registry.get_by_url(&worker_url).unwrap();
    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::from_spec((*current.metadata().spec).clone())
            .priority(8)
            .build(),
    );
    let id = registry.get_id_by_url(&worker_url).unwrap();
    assert!(registry.replace(&id, replacement));
    failure.release.notify_one();
    let error = discovery.await.unwrap().unwrap_err();
    assert!(
        DiscoverZmqControlMetadataStep.is_retryable(&error),
        "stale deadline failure must retry current metadata owner"
    );
    assert_eq!(
        registry.get_by_url(&worker_url).unwrap().status(),
        openai_protocol::worker::WorkerStatus::Ready
    );
    let id = registry.get_id_by_url(&worker_url).unwrap();
    registry.remove(&id);
    policy_registry.on_worker_removed("m");
    assert!(
        policy_registry.get_policy("m").is_none(),
        "removing the last worker must remove its policy"
    );
    server.abort();
}

#[derive(Debug, Default)]
struct CompletionSpy(AtomicUsize);
impl RequestCompletionSink for CompletionSpy {
    fn request_completed(&self, _worker: &dyn Worker) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Default)]
struct DeferredModelFailure {
    enabled: AtomicBool,
    started: Notify,
    release: Notify,
}
