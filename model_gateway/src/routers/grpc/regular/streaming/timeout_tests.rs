//! A worker stream the gateway's request deadline ends: cut at the deadline
//! whether the worker had answered or not, the engine-side stream dropped at
//! once, the worker charged a timeout, and the client told with the 504 the
//! HTTP path answers for its client timeout, as an error event.
use std::{
    future::Future,
    time::{Duration, Instant},
};

use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::worker::HealthCheckConfig;
use smg_grpc_client::vllm_engine::proto;
use tokio::{sync::oneshot, task::JoinHandle};
use tracing_test::traced_test;

use super::{
    eof_tests::{
        chat_spec, chunk, collect_events, complete, gated_stream, processor, scripted_stream,
        CharacterTokenizer,
    },
    *,
};
use crate::{
    routers::grpc::{
        context::WorkerSelection,
        utils::{
            request_deadline::{is_request_timeout, RequestDeadline, REQUEST_TIMEOUT_CODE},
            StreamFailure, StreamSource,
        },
    },
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder, Worker,
    },
};

const REQUEST_ID: &str = "chatcmpl-deadline";
const WORKER_URL: &str = "grpc://engine-under-test:30000";
const TIMEOUT: Duration = Duration::from_secs(1);
/// Scheduling slack for a loaded machine.
const SLACK: Duration = Duration::from_secs(5);

/// A worker whose breaker opens on the first failure.
fn worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(WORKER_URL)
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

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: REQUEST_ID.to_string(),
        model: "deadline-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn source(worker: &Arc<dyn Worker>) -> StreamSource {
    let selection = WorkerSelection::Single {
        worker: Arc::clone(worker),
    };
    StreamSource::new(Some(&selection), &dispatch(), metrics_labels::ENDPOINT_CHAT)
}

/// `stream`, dispatched to `worker` under the deadline, as the execution
/// stage hands it to the streaming layer.
fn under_deadline(stream: ProtoStream, worker: &Arc<dyn Worker>) -> ProtoStream {
    stream.tracked(Arc::clone(worker), 4, true).bounded(
        RequestDeadline::from_now(TIMEOUT),
        Arc::clone(worker),
        metrics_labels::WORKER_REGULAR,
        REQUEST_ID.to_string(),
    )
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a current-thread runtime")
}

/// The rendered line of `family` whose labels contain every `label`.
fn series<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    rendered.lines().find(|line| {
        line.starts_with(&format!("{family}{{")) && labels.iter().all(|label| line.contains(label))
    })
}

/// Stream a chat answer from `stream` under the deadline, ending it the way
/// the response task does: the failure the streaming layer reports, the
/// events the client read, and how long it took.
async fn stream_chat(
    stream: ProtoStream,
    worker: &Arc<dyn Worker>,
) -> (Result<(), StreamFailure>, Vec<Value>, Duration) {
    let (tx, rx) = sse_channel();
    let started = Instant::now();
    let result = processor(false)
        .process_streaming_chunks(
            under_deadline(stream, worker),
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            chat_spec(false),
            &tx,
            None,
        )
        .await;
    let elapsed = started.elapsed();
    if let Err(failure) = &result {
        utils::send_stream_failure(&tx, failure, &source(worker)).await;
    }
    drop(tx);
    (result, collect_events(rx).await, elapsed)
}

/// Run `case` on a current-thread runtime under a local recorder: its result
/// and the rendered `/metrics` text.
fn with_metrics<T>(case: impl Future<Output = T>) -> (T, String) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let out = metrics::with_local_recorder(&recorder, || runtime().block_on(case));
    (out, handle.render())
}

/// A stream that never yields: the engine took the request and went silent.
async fn silent_after(
    first: Vec<proto::GenerateResponse>,
) -> (ProtoStream, JoinHandle<()>, oneshot::Sender<()>) {
    let (gate_tx, gate) = oneshot::channel();
    let (stream, server) = gated_stream(first, gate, vec![], "0").await;
    (stream, server, gate_tx)
}

fn the_timeout_warn(lines: &[&str]) -> Result<(), String> {
    let line = lines
        .iter()
        .find(|line| line.contains("Request timed out in the gateway"))
        .ok_or_else(|| "no WARN for the timed-out request".to_string())?;
    for needle in [
        "WARN",
        REQUEST_ID,
        WORKER_URL,
        r#"worker_type="regular""#,
        r#"phase="stream""#,
        "timeout_secs=1",
        r#"error_type="timeout""#,
    ] {
        if !line.contains(needle) {
            return Err(format!("{needle:?} missing from the WARN: {line}"));
        }
    }
    if lines
        .iter()
        .any(|line| line.contains("Stream ended by the engine before completion"))
    {
        return Err("the deadline is not reported as the engine's doing".to_string());
    }
    Ok(())
}

fn assert_timed_out(failure: &StreamFailure, elapsed: Duration, events: &[Value]) {
    assert!(failure.is_request_timeout(), "{failure:?}");
    assert!(
        elapsed >= TIMEOUT && elapsed < TIMEOUT + SLACK,
        "ended {elapsed:?} after the dispatch, deadline {TIMEOUT:?}"
    );
    // The client reads the HTTP path's 504: its type, its code, and what
    // happened in the gateway's own words.
    assert_eq!(failure.error_type(), "server_error");
    assert_eq!(failure.code(), Some(REQUEST_TIMEOUT_CODE));
    assert_eq!(failure.messages_error_type(), "timeout_error");
    assert!(
        failure
            .to_string()
            .starts_with("request timed out in the gateway after 1 s"),
        "{failure}"
    );
    let error = events
        .iter()
        .find(|event| event.get("error").is_some())
        .unwrap_or_else(|| panic!("an error event among {events:?}"));
    assert_eq!(error["error"]["type"], "server_error", "{error}");
    assert_eq!(error["error"]["code"], REQUEST_TIMEOUT_CODE, "{error}");
    assert_eq!(error["error"]["message"], failure.to_string(), "{error}");
    assert!(
        events
            .last()
            .is_some_and(|event| event.get("error").is_some()),
        "the error event ends the response: {events:?}"
    );
}

fn assert_timeout_counted(rendered: &str) {
    assert!(
        series(
            rendered,
            "smg_worker_errors_total",
            &[
                r#"worker_type="regular""#,
                r#"connection_mode="grpc""#,
                r#"error_type="timeout""#,
            ],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the worker is charged one timeout:\n{rendered}"
    );
    assert!(
        series(
            rendered,
            "smg_router_request_errors_total",
            &[r#"endpoint="chat""#, r#"error_type="timeout""#],
        )
        .is_some_and(|line| line.ends_with(" 1")),
        "the request's outcome is one timeout:\n{rendered}"
    );
    assert!(
        series(
            rendered,
            "smg_worker_errors_total",
            &[r#"error_type="backend_error""#]
        )
        .is_none()
            && series(
                rendered,
                "smg_router_request_errors_total",
                &[r#"error_type="backend_error""#]
            )
            .is_none(),
        "and nothing else:\n{rendered}"
    );
}

#[test]
#[traced_test]
fn a_stream_silent_before_its_first_token_ends_at_the_deadline() {
    let worker = worker();
    let ((result, events, elapsed), rendered) = with_metrics(async {
        let (stream, server, _gate) = silent_after(vec![]).await;
        let out = stream_chat(stream, &worker).await;
        server.abort();
        out
    });
    let failure = result.expect_err("the deadline ended the stream");
    assert_timed_out(&failure, elapsed, &events);
    assert!(
        matches!(
            failure,
            StreamFailure::Upstream {
                completion_tokens: 0,
                ..
            }
        ),
        "{failure:?}"
    );
    assert_timeout_counted(&rendered);
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Open,
        "the worker's breaker sampled the 504, as the HTTP path's client timeout is sampled"
    );
    logs_assert(|lines: &[&str]| the_timeout_warn(lines));
}

#[test]
#[traced_test]
fn a_stream_silent_after_its_first_tokens_ends_at_the_deadline_after_them() {
    let worker = worker();
    let ((result, events, elapsed), rendered) = with_metrics(async {
        let (stream, server, _gate) = silent_after(vec![chunk(0, "hel")]).await;
        let out = stream_chat(stream, &worker).await;
        server.abort();
        out
    });
    let failure = result.expect_err("the deadline ended the stream");
    assert_timed_out(&failure, elapsed, &events);
    assert!(
        matches!(
            failure,
            StreamFailure::Upstream {
                completion_tokens: 3,
                ..
            }
        ),
        "{failure:?}"
    );
    let delivered: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(delivered, "hel", "what the engine produced went out first");
    assert_timeout_counted(&rendered);
    logs_assert(|lines: &[&str]| the_timeout_warn(lines));
}

#[test]
#[traced_test]
fn a_stream_that_completes_within_the_deadline_is_untouched() {
    let worker = worker();
    let ((result, events, elapsed), rendered) = with_metrics(async {
        let (stream, server) =
            scripted_stream(vec![chunk(0, "hi"), complete(0, "stop")], "0").await;
        let out = stream_chat(stream, &worker).await;
        server.abort();
        out
    });
    result.expect("the stream completed");
    assert!(elapsed < TIMEOUT, "{elapsed:?}");
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "{events:?}"
    );
    assert!(
        series(&rendered, "smg_worker_errors_total", &[]).is_none()
            && series(&rendered, "smg_router_request_errors_total", &[]).is_none(),
        "nothing counted:\n{rendered}"
    );
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    logs_assert(|lines: &[&str]| {
        if lines
            .iter()
            .any(|line| line.contains("Request timed out in the gateway"))
        {
            Err("no timeout was logged".to_string())
        } else {
            Ok(())
        }
    });
}

#[tokio::test]
async fn the_deadline_drops_the_engine_stream_at_once_and_ends_the_stream() {
    let worker = worker();
    let (stream, server, _gate) = silent_after(vec![]).await;
    let mut stream = under_deadline(stream, &worker);
    assert_eq!(
        worker.tracked_load(),
        1,
        "the dispatched stream is one of the worker's pile"
    );
    let item = stream.next().await;
    let timed_out = matches!(&item, Some(Err(status)) if is_request_timeout(status));
    assert!(
        timed_out,
        "the deadline's status, not {:?}",
        item.map(|item| item.map(|_| "a response").map_err(|status| status.code()))
    );
    // The stream inside was dropped with the deadline (with it the abort to
    // the engine), while this handle still lives; nothing follows.
    assert_eq!(worker.tracked_load(), 0);
    assert!(stream.next().await.is_none());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    server.abort();
}
