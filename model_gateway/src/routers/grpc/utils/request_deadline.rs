//! The gateway's deadline on a request to a gRPC worker.
//!
//! `--request-timeout-secs` bounds a request to an HTTP worker through the
//! upstream client's timeout. A request to a gRPC worker is bounded here, the
//! same way for every API and on the regular and the disaggregated paths
//! alike: one deadline per dispatch attempt, started when the request leaves
//! for the worker, over the call that opens the worker's stream and over
//! every message on it, the one answer of a non-streaming generation
//! included. When it passes, the worker's stream ends with
//! [`RequestDeadline::status`] and is dropped, which aborts the engine-side
//! request (the abort-on-drop of the engine clients); the worker is charged
//! as the HTTP path charges its client timeout (its circuit breaker samples a
//! 504, the worker-error family counts a `timeout`); and the client reads
//! what the HTTP path answers for the same event: 504 [`REQUEST_TIMEOUT_CODE`]
//! before the response began, an error event typed for the 504 followed by
//! `[DONE]` once it has (see `stream_failure`).

use std::{sync::Arc, time::Duration};

use axum::{http::StatusCode, response::Response};
use tokio::time::Instant;
use tracing::warn;

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{common::retry::mark_non_retryable, error},
    worker::Worker,
};

/// The error code of a request the gateway timed out between itself and the
/// worker: the HTTP relay's code for its upstream client's timeout, so a
/// client reads one code whichever transport served the request.
pub(crate) const REQUEST_TIMEOUT_CODE: &str = "call_upstream_timeout";

/// How the message of a timed-out request's status begins; what tells the
/// gateway's own deadline from a `DEADLINE_EXCEEDED` the worker answered.
const MESSAGE_PREFIX: &str = "request timed out in the gateway after ";

/// A dispatch attempt's deadline: the instant `--request-timeout-secs` after
/// the request left for the worker.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestDeadline {
    at: Instant,
    timeout: Duration,
}

impl RequestDeadline {
    /// The deadline `timeout` from now.
    pub(crate) fn from_now(timeout: Duration) -> Self {
        let now = Instant::now();
        // The flag takes any number of seconds; a timeout the clock cannot
        // add is as good as none.
        let at = now
            .checked_add(timeout)
            .unwrap_or_else(|| now + Duration::from_secs(60 * 60 * 24 * 365));
        Self { at, timeout }
    }

    /// The instant the deadline passes.
    pub(crate) fn at(self) -> Instant {
        self.at
    }

    /// The timeout the deadline was started with.
    pub(crate) fn timeout(self) -> Duration {
        self.timeout
    }

    /// The status a worker's stream ends with when the deadline passes.
    pub(crate) fn status(self) -> tonic::Status {
        tonic::Status::deadline_exceeded(format!(
            "{MESSAGE_PREFIX}{} s: the worker did not finish the request within the gateway's request timeout",
            self.timeout.as_secs()
        ))
    }
}

/// Whether `status` is the gateway's own request deadline, as opposed to a
/// `DEADLINE_EXCEEDED` the worker answered.
pub(crate) fn is_request_timeout(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::DeadlineExceeded && status.message().starts_with(MESSAGE_PREFIX)
}

/// The client's answer for a request timed out before its response began:
/// the HTTP relay's 504 under [`REQUEST_TIMEOUT_CODE`], terminal for the
/// retry layer (the request's whole budget is spent; a replay on another
/// worker would hold the client for a second one).
pub(crate) fn response(status: &tonic::Status) -> Response {
    let mut response = error::gateway_timeout(REQUEST_TIMEOUT_CODE, status.message());
    mark_non_retryable(&mut response);
    response
}

/// The client's answer for a worker stream that failed with `status`: the
/// terminal 504 of [`response`] when the status is the request's own
/// deadline, else the HTTP error `code` / `message` the consumer would have
/// built. A consumer whose error branch only builds the response maps
/// through here; one that also accounts the failure (a leg's breaker, the
/// worker-error family) keeps its own [`is_request_timeout`] arm ahead of
/// that accounting, which the deadline must skip.
pub(crate) fn stream_error_response(
    status: &tonic::Status,
    code: &str,
    message: String,
) -> Response {
    use super::tonic_ext::TonicStatusExt;
    if is_request_timeout(status) {
        return response(status);
    }
    status.to_http_error(code, message)
}

/// The phase of the request the deadline ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// The call that opens the worker's stream had not returned.
    Dispatch,
    /// The stream was open and had not completed.
    Stream,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Dispatch => "dispatch",
            Self::Stream => "stream",
        }
    }
}

/// What the deadline's passing on `worker` leaves behind: one WARN naming
/// the request, the worker, its leg and the phase; the worker-error family's
/// `timeout`; and, when `sample_breaker`, the 504 the HTTP path samples into
/// the worker's circuit breaker for its client timeout (a timed-out dispatch
/// is sampled by its caller through the request's ledger instead, like every
/// failed dispatch).
pub(crate) fn note_expiry(
    worker: &Arc<dyn Worker>,
    worker_type: &'static str,
    request_id: &str,
    phase: Phase,
    timeout: Duration,
    sample_breaker: bool,
) {
    warn!(
        request_id = %request_id,
        worker = worker.url(),
        worker_type,
        phase = phase.label(),
        timeout_secs = timeout.as_secs(),
        http_status = StatusCode::GATEWAY_TIMEOUT.as_u16(),
        error_type = metrics_labels::ERROR_TIMEOUT,
        "Request timed out in the gateway: the worker did not finish it within --request-timeout-secs; the engine-side request is aborted"
    );
    if sample_breaker {
        worker.record_outcome(StatusCode::GATEWAY_TIMEOUT.as_u16());
    }
    Metrics::record_worker_error(
        worker_type,
        metrics_labels::CONNECTION_GRPC,
        metrics_labels::ERROR_TIMEOUT,
    );
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        observability::metrics::test_support::render_with_recorder,
        routers::{
            common::retry::is_retryable_response,
            error::{extract_error_code_from_response, HEADER_X_SMG_ERROR_CODE},
        },
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            BasicWorkerBuilder,
        },
    };

    #[test]
    fn the_deadlines_status_is_told_from_a_workers_own_deadline_exceeded() {
        let deadline = RequestDeadline::from_now(Duration::from_secs(15));
        let status = deadline.status();
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
        assert!(
            status
                .message()
                .starts_with("request timed out in the gateway after 15 s"),
            "{}",
            status.message()
        );
        assert!(is_request_timeout(&status));
        assert!(!is_request_timeout(&tonic::Status::deadline_exceeded(
            "engine scheduling deadline exceeded"
        )));
        assert!(!is_request_timeout(&tonic::Status::internal(
            status.message()
        )));
    }

    #[test]
    fn the_client_reads_the_http_paths_timeout_and_the_retry_layer_stops() {
        let status = RequestDeadline::from_now(Duration::from_secs(15)).status();
        let response = response(&status);
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            extract_error_code_from_response(&response),
            REQUEST_TIMEOUT_CODE
        );
        assert_eq!(
            response
                .headers()
                .get(HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some(REQUEST_TIMEOUT_CODE)
        );
        assert!(
            !is_retryable_response(&response),
            "a 504 is retryable by status; the deadline's is terminal"
        );
    }

    /// A failed stream maps to the deadline's terminal 504 when the status is
    /// the request's own deadline, and to the consumer's code otherwise.
    #[test]
    fn a_failed_stream_maps_to_the_deadlines_504_or_the_consumers_code() {
        let deadline = RequestDeadline::from_now(Duration::from_secs(15)).status();
        let response =
            stream_error_response(&deadline, "worker_stream_failed", "scoring failed".into());
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            extract_error_code_from_response(&response),
            REQUEST_TIMEOUT_CODE
        );
        assert!(
            !is_retryable_response(&response),
            "the deadline's answer is terminal for the retry layer"
        );

        let worker = tonic::Status::unavailable("the worker went away");
        let response =
            stream_error_response(&worker, "worker_stream_failed", "scoring failed".into());
        assert_ne!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_stream_failed"
        );
    }

    #[test]
    fn a_timeout_the_clock_cannot_add_does_not_panic() {
        let deadline = RequestDeadline::from_now(Duration::from_secs(u64::MAX));
        assert!(deadline.at() > Instant::now());
        assert_eq!(deadline.timeout(), Duration::from_secs(u64::MAX));
    }

    #[test]
    fn the_expiry_charges_the_worker_once_and_counts_a_timeout() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine-under-test:30000")
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 1,
                    ..Default::default()
                })
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let rendered = render_with_recorder(|| {
            note_expiry(
                &worker,
                metrics_labels::WORKER_REGULAR,
                "chatcmpl-deadline",
                Phase::Stream,
                Duration::from_secs(15),
                true,
            );
        });
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_worker_errors_total{")
                    && line.contains(r#"worker_type="regular""#)
                    && line.contains(r#"connection_mode="grpc""#)
                    && line.contains(r#"error_type="timeout""#)
                    && line.ends_with(" 1")
            }),
            "{rendered}"
        );
        // A timed-out dispatch is sampled through the request's ledger by its
        // caller: the note leaves the breaker alone when told so.
        let untouched: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine-under-test:30001")
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 1,
                    ..Default::default()
                })
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        note_expiry(
            &untouched,
            metrics_labels::WORKER_REGULAR,
            "chatcmpl-deadline",
            Phase::Dispatch,
            Duration::from_secs(15),
            false,
        );
        assert_eq!(untouched.circuit_breaker_state(), CircuitState::Closed);
    }
}
