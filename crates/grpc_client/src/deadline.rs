//! A call's deadline: the standard `grpc-timeout` header to the worker (which
//! cancels the call when it passes, aborting the engine-side request), and
//! `DEADLINE_EXCEEDED` on this side at the same instant. tonic's client sends
//! the header but bounds only the call that opens the stream, as `CANCELLED`:
//! [`CallDeadline::call`] restates that, and the engine stream wrapper
//! ([`crate::AbortOnDropStream`]) bounds every message after it.

use std::{future::Future, time::Duration};

use tokio::time::Instant;
use tonic::{Code, Request, Status};

/// How the deadline's status message begins (what tells it from a worker's).
const MESSAGE_PREFIX: &str = "request timed out after ";

/// The deadline of one call to a worker: `timeout` after the call was issued.
#[derive(Clone, Copy, Debug)]
pub struct CallDeadline {
    at: Instant,
    timeout: Duration,
}

impl CallDeadline {
    /// The deadline `timeout` from now; one the clock cannot add is a year away.
    pub fn after(timeout: Duration) -> Self {
        let now = Instant::now();
        let at = now
            .checked_add(timeout)
            .unwrap_or_else(|| now + Duration::from_secs(365 * 24 * 60 * 60));
        Self { at, timeout }
    }

    /// A request for `req` under a fresh deadline of `timeout`, as its header.
    pub fn request<T>(req: T, timeout: Duration) -> (Self, Request<T>) {
        let deadline = Self::after(timeout);
        let mut request = Request::new(req);
        // The header holds eight digits of hours at most; tonic panics beyond.
        request.set_timeout(timeout.min(Duration::from_secs(99_999_999 * 60 * 60)));
        (deadline, request)
    }

    /// The instant the deadline passes.
    pub fn at(self) -> Instant {
        self.at
    }

    /// The status a call ends with when the deadline passes.
    pub fn status(self) -> Status {
        Status::deadline_exceeded(format!(
            "{MESSAGE_PREFIX}{:?}: the worker did not finish the request within the call's deadline",
            self.timeout
        ))
    }

    /// Await `call`, reading tonic's own expiry of the header's bound
    /// (`CANCELLED`) as the deadline's status.
    pub async fn call<T>(self, call: impl Future<Output = Result<T, Status>>) -> Result<T, Status> {
        call.await.map_err(|status| {
            // tonic's own words for the bound it enforces from the header.
            if status.code() == Code::Cancelled && status.message() == "Timeout expired" {
                self.status()
            } else {
                status
            }
        })
    }
}

/// Whether `status` is a call's own deadline, not a worker's `DEADLINE_EXCEEDED`.
pub fn is_call_deadline(status: &Status) -> bool {
    status.code() == Code::DeadlineExceeded && status.message().starts_with(MESSAGE_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_is_told_from_a_workers_own_deadline_exceeded() {
        let status = CallDeadline::after(Duration::from_secs(15)).status();
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(
            status.message().starts_with("request timed out after 15s"),
            "{}",
            status.message()
        );
        assert!(is_call_deadline(&status));
        assert!(!is_call_deadline(&Status::deadline_exceeded(
            "engine scheduling deadline exceeded"
        )));
        assert!(!is_call_deadline(&Status::internal(status.message())));
    }

    #[tokio::test]
    async fn tonics_own_expiry_is_restated_and_other_statuses_are_not() {
        let deadline = CallDeadline::after(Duration::from_secs(1));
        let expired = deadline
            .call(async { Err::<(), _>(Status::cancelled("Timeout expired")) })
            .await
            .expect_err("an error");
        assert!(is_call_deadline(&expired));
        let other = deadline
            .call(async { Err::<(), _>(Status::cancelled("stream reset")) })
            .await
            .expect_err("an error");
        assert_eq!(
            (other.code(), other.message()),
            (Code::Cancelled, "stream reset")
        );
        assert_eq!(deadline.call(async { Ok(7) }).await.ok(), Some(7));
    }

    #[test]
    fn the_request_carries_the_header_and_a_huge_timeout_neither_panics_nor_overflows() {
        let (_, request) = CallDeadline::request((), Duration::from_secs(15));
        assert_eq!(
            request
                .metadata()
                .get("grpc-timeout")
                .map(|v| v.to_str().ok()),
            Some(Some("15000000u"))
        );
        let (huge, request) = CallDeadline::request((), Duration::from_secs(u64::MAX));
        assert!(huge.at() > Instant::now());
        assert_eq!(
            request
                .metadata()
                .get("grpc-timeout")
                .map(|v| v.to_str().ok()),
            Some(Some("99999999H"))
        );
    }
}
