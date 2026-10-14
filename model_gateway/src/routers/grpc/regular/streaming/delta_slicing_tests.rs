//! The shape of a streamed chat completion: one SSE event per engine chunk,
//! read as soon as the chunk arrives, unless the gateway was started with
//! `--stream-delta-slicing`, which re-slices a MiniMax stream's deltas to the
//! provider's event sizes.

use std::time::{Duration, Instant};

use axum::http::StatusCode;
use http_body_util::BodyExt;
use openai_protocol::chat::ChatCompletionRequest;

use super::{
    eof_tests::{chunk, complete, processor, scripted_stream_paced, CharacterTokenizer},
    *,
};

/// The gap between the engine's chunks: every chunk its own frame on the
/// wire, as an engine streams its steps.
const ENGINE_STEP: Duration = Duration::from_millis(20);
/// Chunks per stream: short (two-character) deltas, so the re-slicer would
/// hold all of them (its slice threshold is far above their total length).
const CHUNKS: usize = 12;
/// An event must reach the client within this of the chunk that produced it.
/// The re-slicer's idle flush is 250 ms; the whole paced stream is shorter.
const MAX_DELAY: Duration = Duration::from_millis(150);

/// A MiniMax-profile request (the model id selects the profile).
fn minimax_spec() -> ChatResponseSpec {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "MiniMax-M3", "messages": [], "stream": true
    }))
    .expect("chat request");
    assert_eq!(
        ChatResponseSpec::from(&request).provider,
        ProviderProfile::Minimax
    );
    ChatResponseSpec::from(&request)
}

/// Stream `CHUNKS` paced chunks through `processor`; per content event the
/// time since the stream started and the text.
async fn paced_content_events(processor: StreamingProcessor) -> Vec<(Duration, String)> {
    let mut frames: Vec<_> = (0..CHUNKS).map(|i| chunk(0, &format!("{i:02}"))).collect();
    frames.push(complete(0, "stop"));
    let (stream, server) = scripted_stream_paced(frames, "0", ENGINE_STEP).await;
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-paced".to_string(),
        model: "MiniMax-M3".to_string(),
        created: 1_700_000_000,
        weight_version: None,
    };
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let started = Instant::now();
    let response = Arc::new(processor)
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            minimax_spec(),
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
            source,
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut events = Vec::new();
    let mut pending = Vec::new();
    while let Some(frame) = body.frame().await {
        let frame = frame.expect("an SSE frame");
        let Some(data) = frame.data_ref() else {
            continue;
        };
        let at = started.elapsed();
        pending.extend_from_slice(data);
        while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = pending.drain(..end + 2).collect();
            let text = String::from_utf8_lossy(&event);
            let Some(payload) = text.strip_prefix("data: ") else {
                continue;
            };
            let Ok(json) = serde_json::from_str::<Value>(payload.trim()) else {
                continue;
            };
            if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                if !content.is_empty() {
                    events.push((at, content.to_string()));
                }
            }
        }
    }
    server.abort();
    events
}

/// By default every engine chunk is its own delta and leaves at once: the
/// client reads a token when the engine produced it, on a MiniMax model as on
/// any other.
#[tokio::test]
async fn a_minimax_stream_keeps_one_delta_per_engine_chunk_by_default() {
    let events = paced_content_events(processor(false)).await;
    let texts: Vec<&str> = events.iter().map(|(_, text)| text.as_str()).collect();
    assert_eq!(
        events.len(),
        CHUNKS,
        "one content event per engine chunk, got {texts:?}"
    );
    for (i, (at, text)) in events.iter().enumerate() {
        let produced = ENGINE_STEP * (i as u32 + 1);
        assert!(
            *at <= produced + MAX_DELAY,
            "event {i} ({text:?}) reached the client {:?} after the engine produced it",
            at.saturating_sub(produced)
        );
    }
}

/// With the knob on, the MiniMax re-slicing holds the short deltas and emits
/// them in fewer, larger events.
#[tokio::test]
async fn the_knob_turns_the_minimax_re_slicing_on() {
    let events = paced_content_events(processor(false).with_stream_delta_slicing(true)).await;
    assert!(
        events.len() < CHUNKS,
        "re-sliced into fewer events than engine chunks, got {}",
        events.len()
    );
    let text: String = events.iter().map(|(_, text)| text.as_str()).collect();
    let expected: String = (0..CHUNKS).map(|i| format!("{i:02}")).collect();
    assert_eq!(
        text, expected,
        "the re-slicing changes the cuts, not the text"
    );
}
