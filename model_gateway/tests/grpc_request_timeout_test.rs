//! `--request-timeout-secs` is the deadline of every call to a gRPC worker.
//! A worker that goes silent before its first token, after its first tokens
//! or on a non-streaming request, or that never opens its stream, is timed
//! out at the configured timeout on every API and on the disaggregated path
//! too; the engine-side request is aborted, the worker is charged a timeout,
//! and the client reads the 504 the HTTP path answers for its upstream
//! client's timeout. Every case is capped: a gateway without the deadline
//! waits on the silent worker and fails its cap.

#[path = "common/mod.rs"]
mod common;

use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{body::to_bytes, http::StatusCode};
use futures::{stream, Stream, StreamExt};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{
        circuit_breaker::CircuitState, BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode,
        RuntimeType, Worker, WorkerType,
    },
};
use smg_grpc_client::{common_proto as common_pb, tokenspeed_scheduler::tokenspeed_proto as ts};
use tokio::{net::TcpListener, sync::Notify, time::timeout};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

const MODEL: &str = "request-timeout-test-model";
/// The gateway's request timeout under test.
const TIMEOUT_SECS: u64 = 1;
const TIMEOUT: Duration = Duration::from_secs(TIMEOUT_SECS);
/// Scheduling slack for a loaded machine.
const SLACK: Duration = Duration::from_secs(6);
/// A request a gateway without the deadline never answers fails here.
const CASE_CAP: Duration = Duration::from_secs(20);
/// The HTTP path's error code for its upstream client's timeout.
const TIMEOUT_CODE: &str = "call_upstream_timeout";
/// How the deadline's message begins.
const TIMEOUT_MESSAGE: &str = "request timed out after 1s";
/// Token ids in the mock tokenizer's vocabulary.
const TOKENS: [u32; 5] = [1, 2, 3, 4, 1];

/// What the scripted worker does with a generate call.
#[derive(Clone, Copy, Debug)]
enum Script {
    /// Opens the stream and never writes to it.
    SilentBeforeFirstToken,
    /// Writes this many tokens, then nothing more.
    SilentAfter(usize),
    /// Never returns from the call that opens the stream.
    NeverOpens,
    /// Answers this many tokens and a completion at once.
    Answers(usize),
}

/// A gRPC worker that follows its script, counts the aborts it receives and
/// notes whether its calls carried a deadline.
struct ScriptedWorker {
    script: Script,
    aborts: AtomicUsize,
    aborted: Notify,
    saw_deadline: AtomicBool,
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type KvStream = Pin<Box<dyn Stream<Item = Result<common_pb::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common_pb::GetTokenizerChunk, Status>> + Send>>;

impl ScriptedWorker {
    fn new(script: Script) -> Arc<Self> {
        Arc::new(Self {
            script,
            aborts: AtomicUsize::new(0),
            aborted: Notify::new(),
            saw_deadline: AtomicBool::new(false),
        })
    }

    /// Serve the worker in-process; its port.
    #[expect(
        clippy::expect_used,
        clippy::disallowed_methods,
        reason = "test helper - panicking on failure is intentional; the server task lives for the test process"
    )]
    async fn serve(self: &Arc<Self>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the scripted worker");
        let port = listener
            .local_addr()
            .expect("scripted worker address")
            .port();
        let worker = Arc::clone(self);
        tokio::spawn(
            Server::builder()
                .add_service(TokenSpeedSchedulerServer::from_arc(worker))
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        port
    }

    fn chunk(request_id: &str, token: u32) -> ts::GenerateResponse {
        ts::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                token_ids: vec![token],
                prompt_tokens: 1,
                completion_tokens: 1,
                ..Default::default()
            })),
        }
    }

    fn complete(request_id: &str, tokens: &[u32]) -> ts::GenerateResponse {
        ts::GenerateResponse {
            request_id: request_id.to_string(),
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: tokens.to_vec(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: tokens.len() as u32,
                ..Default::default()
            })),
        }
    }

    /// Whether an abort reached the worker within `within`.
    async fn aborted_within(&self, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while self.aborts.load(Ordering::SeqCst) == 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if timeout(left, self.aborted.notified()).await.is_err() {
                return self.aborts.load(Ordering::SeqCst) > 0;
            }
        }
        true
    }
}

#[tonic::async_trait]
impl TokenSpeedScheduler for ScriptedWorker {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        if request.metadata().contains_key("grpc-timeout") {
            self.saw_deadline.store(true, Ordering::SeqCst);
        }
        let request_id = request.into_inner().request_id;
        let tokens = |n: usize| -> Vec<Result<ts::GenerateResponse, Status>> {
            TOKENS
                .iter()
                .cycle()
                .take(n)
                .map(|token| Ok(Self::chunk(&request_id, *token)))
                .collect()
        };
        let stream: GenStream = match self.script {
            Script::SilentBeforeFirstToken => Box::pin(stream::pending()),
            Script::SilentAfter(n) => Box::pin(stream::iter(tokens(n)).chain(stream::pending())),
            Script::NeverOpens => std::future::pending().await,
            Script::Answers(n) => {
                let mut items = tokens(n);
                let produced: Vec<u32> = TOKENS.iter().copied().cycle().take(n).collect();
                items.push(Ok(Self::complete(&request_id, &produced)));
                Box::pin(stream::iter(items))
            }
        };
        Ok(Response::new(stream))
    }

    async fn abort(
        &self,
        _: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.aborts.fetch_add(1, Ordering::SeqCst);
        self.aborted.notify_one();
        Ok(Response::new(ts::AbortResponse::default()))
    }

    async fn health_check(
        &self,
        _: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn get_model_info(
        &self,
        _: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn get_server_info(
        &self,
        _: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn get_loads(
        &self,
        _: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn subscribe_kv_events(
        &self,
        _: Request<common_pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn flush_cache(
        &self,
        _: Request<common_pb::FlushCacheRequest>,
    ) -> Result<Response<common_pb::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn start_profile(
        &self,
        _: Request<common_pb::StartProfileRequest>,
    ) -> Result<Response<common_pb::ProfileResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn stop_profile(
        &self,
        _: Request<common_pb::StopProfileRequest>,
    ) -> Result<Response<common_pb::ProfileResponse>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }

    async fn get_tokenizer(
        &self,
        _: Request<common_pb::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("scripted worker"))
    }
}

/// A worker handle for `port`, whose breaker opens on the first failure.
fn worker_handle(port: u16, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                ..Default::default()
            })
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// A gRPC router with `--request-timeout-secs` at [`TIMEOUT_SECS`] over the
/// given workers: one regular worker, or a prefill and a decode worker.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn router_over(workers: &[Arc<dyn Worker>], mode: RoutingMode) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(TIMEOUT_SECS)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    tokenizers
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    for worker in workers {
        context
            .worker_registry
            .register(Arc::clone(worker))
            .unwrap();
    }
    RouterFactory::create_router(&context)
        .await
        .expect("gRPC router should build")
}

fn regular_mode() -> RoutingMode {
    RoutingMode::Regular {
        worker_urls: vec![],
    }
}

fn pd_mode() -> RoutingMode {
    RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    }
}

/// One regular worker following `script`, its handle, and the router over it.
async fn bed(script: Script) -> (Arc<ScriptedWorker>, Arc<dyn Worker>, Box<dyn RouterTrait>) {
    let worker = ScriptedWorker::new(script);
    let handle = worker_handle(worker.serve().await, WorkerType::Regular);
    let router = router_over(std::slice::from_ref(&handle), regular_mode()).await;
    (worker, handle, router)
}

/// A router over one prefill and one decode worker following their scripts.
async fn pd_bed(prefill: Script, decode: Script) -> Box<dyn RouterTrait> {
    let prefill = ScriptedWorker::new(prefill);
    let decode = ScriptedWorker::new(decode);
    let prefill_handle = worker_handle(prefill.serve().await, WorkerType::Prefill);
    let decode_handle = worker_handle(decode.serve().await, WorkerType::Decode);
    router_over(&[prefill_handle, decode_handle], pd_mode()).await
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("request-timeout-test-tenant"))
}

/// The APIs the deadline applies to.
#[derive(Clone, Copy, Debug)]
enum Api {
    Chat,
    Completions,
    Messages,
    Generate,
    Responses,
}

/// What the client read: the status, the body, and how long the request took.
struct Outcome {
    status: StatusCode,
    body: String,
    elapsed: Duration,
}

impl Outcome {
    /// The `data:` payloads of an SSE body, parsed.
    fn events(&self) -> Vec<Value> {
        self.body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }

    fn json(&self) -> Value {
        let parsed = serde_json::from_str(&self.body);
        assert!(parsed.is_ok(), "a JSON body: {}", self.body);
        parsed.unwrap_or_default()
    }
}

/// One request to `api` through `router`, read to its end under the case cap.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn call(router: &dyn RouterTrait, api: Api, stream: bool) -> Outcome {
    let tenant = tenant();
    let started = Instant::now();
    let response = timeout(CASE_CAP, async {
        match api {
            Api::Chat => {
                let request = serde_json::from_value(json!({
                    "model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 16, "stream": stream,
                }))
                .unwrap();
                router.route_chat(None, &tenant, request, MODEL).await
            }
            Api::Completions => {
                let request = serde_json::from_value(json!({
                    "model": MODEL, "prompt": "Hello", "max_tokens": 16, "stream": stream,
                }))
                .unwrap();
                router.route_completion(None, &tenant, request, MODEL).await
            }
            Api::Messages => {
                let request = serde_json::from_value(json!({
                    "model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 16, "stream": stream,
                }))
                .unwrap();
                router.route_messages(None, &tenant, request, MODEL).await
            }
            Api::Generate => {
                let request = serde_json::from_value(json!({
                    "model": MODEL, "text": "Hello", "stream": stream,
                    "sampling_params": { "max_new_tokens": 16 },
                }))
                .unwrap();
                router.route_generate(None, &tenant, request, MODEL).await
            }
            Api::Responses => {
                let request = serde_json::from_value(json!({
                    "model": MODEL, "input": "Hello", "stream": stream,
                    "max_output_tokens": 16, "store": false,
                }))
                .unwrap();
                router.route_responses(None, &tenant, request, MODEL).await
            }
        }
    })
    .await
    .expect("the request must be answered within the case cap: without the deadline the gateway waits on the silent worker for as long as the client does");
    let status = response.status();
    let bytes = timeout(CASE_CAP, to_bytes(response.into_body(), 1 << 20))
        .await
        .expect("the response body must end within the case cap: without the deadline the stream stays open on the silent worker")
        .expect("response body");
    Outcome {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
        elapsed: started.elapsed(),
    }
}

fn assert_at_the_deadline(outcome: &Outcome) {
    assert!(
        outcome.elapsed >= TIMEOUT && outcome.elapsed < TIMEOUT + SLACK,
        "answered {:?} after the dispatch, the deadline is {TIMEOUT:?}: {}",
        outcome.elapsed,
        outcome.body
    );
    assert!(outcome.body.contains(TIMEOUT_MESSAGE), "{}", outcome.body);
}

/// A request timed out before its response began: the HTTP path's 504.
fn assert_timeout_response(outcome: &Outcome) {
    assert_at_the_deadline(outcome);
    assert_eq!(
        outcome.status,
        StatusCode::GATEWAY_TIMEOUT,
        "{}",
        outcome.body
    );
    let body = outcome.json();
    assert_eq!(body["error"]["code"], TIMEOUT_CODE, "{body}");
    assert_eq!(body["error"]["type"], "server_error", "{body}");
}

/// A streamed request timed out after its response began: the error event
/// typed and coded as the 504, then the stream's terminator.
fn assert_timeout_event(outcome: &Outcome, api: Api) {
    assert_at_the_deadline(outcome);
    assert_eq!(outcome.status, StatusCode::OK, "{}", outcome.body);
    match api {
        Api::Chat | Api::Completions | Api::Generate => {
            let events = outcome.events();
            let error = events.last().cloned().unwrap_or_default();
            assert_eq!(error["error"]["type"], "server_error", "{}", outcome.body);
            assert_eq!(error["error"]["code"], TIMEOUT_CODE, "{}", outcome.body);
            assert!(
                outcome.body.trim_end().ends_with("data: [DONE]"),
                "{}",
                outcome.body
            );
        }
        Api::Messages => {
            assert!(outcome.body.contains("event: error"), "{}", outcome.body);
            assert!(
                outcome.body.contains(r#""type":"timeout_error""#),
                "{}",
                outcome.body
            );
        }
        Api::Responses => {
            assert!(outcome.body.contains("response.failed"), "{}", outcome.body);
            assert!(outcome.body.contains(TIMEOUT_CODE), "{}", outcome.body);
        }
    }
}

/// Run `case` on a current-thread runtime under a local recorder: its result
/// and the rendered `/metrics` text.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn with_metrics<T, F, Fut>(case: F) -> (T, String)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        rt.block_on(case())
    });
    (out, handle.render())
}

/// The rendered line of `family` whose labels contain every `label`.
fn series<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    rendered.lines().find(|line| {
        line.starts_with(&format!("{family}{{")) && labels.iter().all(|label| line.contains(label))
    })
}

/// Whether the one sample of `family` with `labels` is 1.
fn counted_once(rendered: &str, family: &str, labels: &[&str]) -> bool {
    series(rendered, family, labels).is_some_and(|line| line.ends_with(" 1"))
}

#[tokio::test]
async fn a_stream_silent_before_its_first_token_is_timed_out_and_aborted() {
    let (worker, handle, router) = bed(Script::SilentBeforeFirstToken).await;
    let outcome = call(router.as_ref(), Api::Chat, true).await;
    assert_timeout_event(&outcome, Api::Chat);
    assert!(
        worker.aborted_within(SLACK).await,
        "the engine-side request is aborted with the deadline"
    );
    assert_eq!(
        handle.circuit_breaker_state(),
        CircuitState::Open,
        "the worker's breaker sampled the 504, as the HTTP path samples its client timeout"
    );
}

#[tokio::test]
async fn a_stream_silent_after_its_first_tokens_is_timed_out_after_them() {
    let (worker, _handle, router) = bed(Script::SilentAfter(5)).await;
    let outcome = call(router.as_ref(), Api::Chat, true).await;
    assert_timeout_event(&outcome, Api::Chat);
    let deltas = outcome
        .events()
        .iter()
        .filter(|event| event["choices"][0]["delta"]["content"].is_string())
        .count();
    assert!(
        deltas >= 1,
        "what the engine produced went out first: {}",
        outcome.body
    );
    assert!(worker.aborted_within(SLACK).await);
}

#[test]
fn a_non_streaming_request_is_timed_out_counted_and_aborted() {
    let ((outcome, aborted, breaker), rendered) = with_metrics(|| async {
        let (worker, handle, router) = bed(Script::SilentBeforeFirstToken).await;
        let outcome = call(router.as_ref(), Api::Chat, false).await;
        let aborted = worker.aborted_within(SLACK).await;
        (outcome, aborted, handle.circuit_breaker_state())
    });
    assert_timeout_response(&outcome);
    assert!(
        aborted,
        "the engine-side request is aborted with the deadline"
    );
    assert_eq!(breaker, CircuitState::Open);
    assert!(
        counted_once(
            &rendered,
            "smg_router_request_errors_total",
            &[r#"endpoint="chat""#, r#"error_type="timeout""#],
        ),
        "the request's outcome is one timeout:\n{rendered}"
    );
    assert!(
        counted_once(
            &rendered,
            "smg_worker_errors_total",
            &[
                r#"worker_type="regular""#,
                r#"connection_mode="grpc""#,
                r#"error_type="timeout""#,
            ],
        ),
        "the worker is charged one timeout:\n{rendered}"
    );
    assert!(
        !rendered.contains(r#"error_type="backend_error""#),
        "and nothing else:\n{rendered}"
    );
}

#[tokio::test]
async fn a_dispatch_the_worker_never_answers_is_timed_out() {
    let (worker, handle, router) = bed(Script::NeverOpens).await;
    let outcome = call(router.as_ref(), Api::Chat, false).await;
    assert_timeout_response(&outcome);
    assert_eq!(handle.circuit_breaker_state(), CircuitState::Open);
    // No stream ever opened, so there is nothing to abort: the call is
    // cancelled, which the worker reads as the client going away.
    assert_eq!(worker.aborts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn every_api_is_timed_out_the_same_way_streamed_and_not() {
    for api in [
        Api::Chat,
        Api::Completions,
        Api::Messages,
        Api::Generate,
        Api::Responses,
    ] {
        for stream in [false, true] {
            let (worker, _handle, router) = bed(Script::SilentBeforeFirstToken).await;
            let outcome = call(router.as_ref(), api, stream).await;
            if stream {
                assert_timeout_event(&outcome, api);
            } else {
                assert_timeout_response(&outcome);
            }
            assert!(
                worker.aborted_within(SLACK).await,
                "{api:?} stream={stream}: the engine-side request is aborted"
            );
        }
    }
}

#[tokio::test]
async fn every_call_carries_its_deadline_as_the_standard_header() {
    let (worker, _handle, router) = bed(Script::Answers(1)).await;
    let outcome = call(router.as_ref(), Api::Chat, false).await;
    assert_eq!(outcome.status, StatusCode::OK, "{}", outcome.body);
    assert!(
        worker.saw_deadline.load(Ordering::SeqCst),
        "the worker read the call's deadline from grpc-timeout"
    );
}

#[tokio::test]
async fn a_worker_that_answers_in_time_is_untouched() {
    let (worker, handle, router) = bed(Script::Answers(3)).await;
    for stream in [false, true] {
        let outcome = call(router.as_ref(), Api::Chat, stream).await;
        assert_eq!(outcome.status, StatusCode::OK, "{}", outcome.body);
        assert!(outcome.elapsed < TIMEOUT, "{:?}", outcome.elapsed);
        assert!(!outcome.body.contains("timed out"), "{}", outcome.body);
    }
    assert_eq!(worker.aborts.load(Ordering::SeqCst), 0);
    assert_eq!(handle.circuit_breaker_state(), CircuitState::Closed);
}

/// Disaggregated dispatch: the decode leg's silence ends the request at the
/// deadline and is charged to the decode worker; the prefill leg, which
/// answered, is not blamed.
#[test]
fn a_silent_decode_leg_is_timed_out_on_the_disaggregated_path() {
    let (outcome, rendered) = with_metrics(|| async {
        let router = pd_bed(Script::Answers(1), Script::SilentBeforeFirstToken).await;
        call(router.as_ref(), Api::Chat, true).await
    });
    assert_timeout_event(&outcome, Api::Chat);
    assert!(
        counted_once(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#, r#"error_type="timeout""#],
        ),
        "the decode leg carries the timeout:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="prefill""#]
        )
        .is_none(),
        "the prefill leg, which answered, is not blamed:\n{rendered}"
    );
    assert!(
        counted_once(
            &rendered,
            "smg_router_request_errors_total",
            &[r#"backend_type="pd""#, r#"error_type="timeout""#],
        ),
        "the request's outcome is one timeout:\n{rendered}"
    );
}

/// The prefill leg's silence ends the request at the deadline too, charged to
/// the prefill worker; its decode partner is not blamed.
#[test]
fn a_silent_prefill_leg_is_timed_out_on_the_disaggregated_path() {
    let (outcome, rendered) = with_metrics(|| async {
        let router = pd_bed(Script::SilentBeforeFirstToken, Script::Answers(1)).await;
        call(router.as_ref(), Api::Chat, false).await
    });
    assert_timeout_response(&outcome);
    assert!(
        counted_once(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="prefill""#, r#"error_type="timeout""#],
        ),
        "the prefill leg carries the timeout:\n{rendered}"
    );
    assert!(
        series(
            &rendered,
            "smg_worker_errors_total",
            &[r#"worker_type="decode""#]
        )
        .is_none(),
        "the decode partner is not blamed:\n{rendered}"
    );
}
