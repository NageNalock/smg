//! A disaggregated decode leg whose engine side goes away mid-stream ends
//! every open choice with a `Complete` whose finish reason is `abort`: what
//! the servicers send for the requests their engine left behind. The gateway
//! did not ask for that abort (its own aborts drop the stream first), so the
//! stream is a failed one, not a short answer: the client reads an SSE
//! `stream_error` event and `[DONE]` instead of a finish reason outside the
//! API's vocabulary, the decode worker is charged an `engine_error` in the
//! worker-error family and the request one in the router-error family, as a
//! stream the engine ends with `error` already is. A non-streamed answer the
//! decode leg aborts is the same failure on the buffered path: a 5xx under
//! the worker-stream code, never a 200 with a truncated answer.
//!
//! The mock engines speak the vLLM proto (sequential dispatch): the prefill
//! leg completes, the decode leg streams some tokens and then aborts.

#[path = "common/mod.rs"]
mod common;

use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use proto::generate_response::Response as GenResp;
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

const MODEL: &str = "pd-decode-abort-test-model";
/// Tokens an engine that completes produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Tokens the aborting decode leg streams before its engine side goes away.
const ABORT_AFTER: u32 = 2;
/// Longer than any case: a request held past it fails the case instead of
/// hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The words the gateway ends an aborted stream with (its stream-failure
/// message), as the client reads them in the SSE error event.
const ABORT_MESSAGE: &str = "engine aborted the request before it completed";

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// What a mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// `OUTPUT_TOKENS` tokens, then a `stop` completion.
    Completes,
    /// The given tokens, then the `abort` completion a servicer sends for a
    /// request its engine side left behind.
    AbortsAfter(u32),
}

/// A mock engine counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
}

impl Engine {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn chunk(id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Chunk(proto::GenerateStreamChunk {
            token_ids: vec![id],
            prompt_tokens: 1,
            completion_tokens: 1,
            index: 0,
            ..Default::default()
        })),
    }
}

fn complete(finish_reason: &str, tokens: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=tokens).collect(),
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 1,
            completion_tokens: tokens,
            index: 0,
            ..Default::default()
        })),
    }
}

/// Feed one generate stream as `behaviour` says.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce(tx: GenerateTx, behaviour: Behaviour) {
    tokio::spawn(async move {
        let (tokens, finish_reason) = match behaviour {
            Behaviour::Completes => (OUTPUT_TOKENS, "stop"),
            Behaviour::AbortsAfter(tokens) => (tokens, "abort"),
        };
        for id in 1..=tokens {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete(finish_reason, tokens))).await;
    });
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for Engine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        produce(tx, self.behaviour);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Ok(Response::new(
            smg_grpc_client::common_proto::FlushCacheResponse::default(),
        ))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Ok(Response::new(proto::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Ok(Response::new(proto::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Ok(Response::new(proto::GetLoadsResponse::default()))
    }

    type GetTokenizerStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<smg_grpc_client::common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type SubscribeKvEventsStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg_grpc_client::common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }
}

/// Serve a mock engine in-process; returns its port.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_engine(engine: Engine) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock engine");
    let port = listener.local_addr().expect("mock engine address").port();
    tokio::spawn(
        Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(engine))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    port
}

/// A PD gRPC router over the given legs (sequential dispatch).
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(legs: &[(u16, WorkerType)]) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    for (port, worker_type) in legs {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                .worker_type(*worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        app_context.worker_registry.register(worker).unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-decode-abort-test-tenant"))
}

/// One streaming chat through `router`: its status and body.
async fn one_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String) {
    one_chat(router, true).await
}

/// One chat through `router`, streamed or not: its status and body.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_chat(router: &dyn RouterTrait, stream: bool) -> (http::StatusCode, String) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect("the request must be answered within the case cap");
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf-8 body"),
    )
}

/// Run `case` on a current-thread runtime under a thread-local recorder and
/// return the rendered `/metrics` text.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn render_after<F, Fut>(case: F) -> String
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        rt.block_on(case());
    });
    handle.render()
}

/// The sum of a counter's samples carrying every one of `labels`; zero when
/// none was emitted.
fn counter(rendered: &str, name: &str, labels: &[String]) -> f64 {
    let prefix = format!("{name}{{");
    rendered
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .filter(|line| labels.iter().all(|label| line.contains(label.as_str())))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.parse::<f64>().ok())
        .sum()
}

/// `smg_worker_errors_total` for `worker_type` and `error_type`.
fn worker_errors(rendered: &str, worker_type: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_worker_errors_total",
        &[
            format!("worker_type=\"{worker_type}\""),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// `smg_router_request_errors_total` for `error_type` on the PD backend.
fn router_errors(rendered: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_router_request_errors_total",
        &[
            "backend_type=\"pd\"".to_string(),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// The decode leg aborts after two tokens: the client reads them, then the
/// `stream_error` event and `[DONE]`, never an `abort` finish reason; the
/// decode worker is charged one `engine_error`, the prefill leg nothing.
#[test]
fn a_decode_stream_the_engine_aborts_ends_as_a_stream_error_not_a_short_completion() {
    let prefill = Engine::new(Behaviour::Completes);
    let decode = Engine::new(Behaviour::AbortsAfter(ABORT_AFTER));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (decode_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        // The headers were out before the abort: the status is the stream's.
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(
            !body.contains(r#""finish_reason":"abort""#),
            "an engine abort is not a completion the client can read as a short answer:\n{body}"
        );
        assert!(
            body.contains(r#""type":"stream_error""#),
            "the stream ends with the worker-failure event:\n{body}"
        );
        assert!(
            body.contains(ABORT_MESSAGE),
            "the event names the abort:\n{body}"
        );
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(
            body.contains(r#""finish_reason":null"#),
            "the tokens streamed before the abort still reach the client:\n{body}"
        );
    });
    assert_eq!(prefill_calls.calls(), 1, "one prefill leg");
    assert_eq!(decode_calls.calls(), 1, "one decode leg, not replayed");
    assert_eq!(
        worker_errors(&rendered, "decode", "engine_error"),
        1.0,
        "the decode worker is charged the engine's own report:\n{rendered}"
    );
    assert_eq!(
        router_errors(&rendered, "engine_error"),
        1.0,
        "the request counts as an engine error on the PD backend:\n{rendered}"
    );
    for error_type in [
        "engine_error",
        "backend_error",
        "timeout",
        "validation_error",
    ] {
        assert_eq!(
            worker_errors(&rendered, "prefill", error_type),
            0.0,
            "the prefill leg did its part ({error_type}):\n{rendered}"
        );
    }
}

/// The buffered path: a non-streamed answer whose decode leg aborts is an
/// upstream failure under the worker-stream code (a 5xx the client may
/// retry), not a 200 carrying a truncated answer; the decode worker is
/// charged the engine's own report, once per attempt the retry layer made.
#[test]
fn a_collected_answer_the_engine_aborts_is_an_upstream_failure_not_a_short_completion() {
    let prefill = Engine::new(Behaviour::Completes);
    let decode = Engine::new(Behaviour::AbortsAfter(ABORT_AFTER));
    let decode_calls = decode.clone();
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (decode_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_chat(router.as_ref(), false).await;
        assert!(
            status.is_server_error(),
            "an aborted answer is an upstream failure, not {status}:\n{body}"
        );
        assert!(body.contains("worker_stream_failed"), "{body}");
        assert!(body.contains(ABORT_MESSAGE), "{body}");
        assert!(
            !body.contains(r#""finish_reason":"abort""#),
            "never a truncated answer with an abort finish reason:\n{body}"
        );
    });
    let attempts = decode_calls.calls();
    assert!(attempts >= 1, "the decode leg was asked");
    assert_eq!(
        worker_errors(&rendered, "decode", "engine_error"),
        attempts as f64,
        "the decode worker is charged the engine's own report, once per attempt:\n{rendered}"
    );
    assert!(
        router_errors(&rendered, "backend_error") + router_errors(&rendered, "engine_error") >= 1.0,
        "the request counts in the router-error family:\n{rendered}"
    );
}

/// A decode leg that completes is untouched by the classification: a `stop`
/// finish, no error event, nothing charged.
#[test]
fn a_decode_stream_that_completes_is_a_normal_completion() {
    let prefill = Engine::new(Behaviour::Completes);
    let decode = Engine::new(Behaviour::Completes);
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (decode_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
        assert!(!body.contains(r#""error""#), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    });
    assert_eq!(worker_errors(&rendered, "decode", "engine_error"), 0.0);
    assert_eq!(router_errors(&rendered, "engine_error"), 0.0);
}
