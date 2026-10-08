//! The Chat response paths when a symphony table reads the output, with exact gRPC
//! frames: what reaches the client, and what the envelope around the parse keeps doing.

use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::vllm_engine::proto::{self, generate_response::Response as GenerationEvent};

use super::{
    eof_tests::{
        chat_text, chunk, collect_events, complete, complete_with_prompt, dispatch,
        scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::{
    routers::grpc::{
        context::ExecutionResult, regular::processor::ResponseProcessor, symphony::SymphonySpec,
    },
    worker::WorkerRegistry,
};

const PROMPT: &str = "<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n";
const QWEN3: &str = "symphony:qwen3";

fn resolver() -> utils::ParserResolver {
    utils::ParserResolver::new(
        Arc::new(WorkerRegistry::new()),
        Some(QWEN3.to_string()),
        None,
        None,
    )
}

fn processor() -> StreamingProcessor {
    StreamingProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        resolver(),
        "vllm",
    )
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}
        }}}
    }])
}

/// A Chat spec armed the way request building arms it for a model that selects the Qwen3 table.
fn chat_spec(extra: Value) -> ChatResponseSpec {
    table_chat_spec(QWEN3, extra)
}

/// A Chat spec armed for a model that selects the table `name` names.
fn table_chat_spec(name: &str, extra: Value) -> ChatResponseSpec {
    let mut request = json!({"model": "eof-test", "messages": [], "stream": true});
    for (key, value) in extra.as_object().expect("object") {
        request[key] = value.clone();
    }
    let request: ChatCompletionRequest = serde_json::from_value(request).expect("chat request");
    let mut spec = ChatResponseSpec::from(&request);
    spec.symphony = SymphonySpec::for_chat(
        symphony::table(name).expect("table"),
        &spec,
        request.parallel_tool_calls,
        PROMPT,
        None,
        None,
    );
    spec
}

type StopParams = (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool);

fn no_stop() -> StopParams {
    (None, None, false, false, false)
}

fn stop_at(stop: &str) -> StopParams {
    (
        Some(StringOrArray::Array(vec![stop.to_string()])),
        None,
        false,
        false,
        false,
    )
}

async fn chat_events(
    responses: Vec<proto::GenerateResponse>,
    spec: ChatResponseSpec,
    stop: StopParams,
) -> Vec<Value> {
    let (stream, server) = scripted_stream(responses, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor()
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            stop,
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    events
}

/// The call deltas of one choice folded into `(id, name, arguments)` per call.
fn chat_calls(events: &[Value], index: u32) -> Vec<(String, String, String)> {
    let mut calls: Vec<(String, String, String)> = Vec::new();
    for event in events {
        let choice = &event["choices"][0];
        if choice["index"] != index {
            continue;
        }
        for call in choice["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(id) = call["id"].as_str() {
                calls.push((
                    id.to_string(),
                    call["function"]["name"].as_str().unwrap().to_string(),
                    String::new(),
                ));
            } else {
                let slot = call["index"].as_u64().unwrap() as usize;
                calls[slot]
                    .2
                    .push_str(call["function"]["arguments"].as_str().unwrap());
            }
        }
    }
    calls
}

fn finishes(events: &[Value], index: u32) -> Vec<Value> {
    events
        .iter()
        .map(|event| &event["choices"][0])
        .filter(|choice| choice["index"] == index && !choice["finish_reason"].is_null())
        .map(|choice| choice["finish_reason"].clone())
        .collect()
}

const CALL: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

#[tokio::test]
async fn chat_stream_reads_reasoning_content_and_calls_through_the_table() {
    let events = chat_events(
        vec![
            chunk(0, "<think>\npl"),
            chunk(0, "an\n</think>\n\nHi\n<tool"),
            chunk(
                0,
                "_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa",
            ),
            chunk(0, "ris\"}}\n</tool_call>"),
            complete(0, "stop"),
        ],
        chat_spec(json!({"tools": tools()})),
        no_stop(),
    )
    .await;
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "\nplan\n");
    assert_eq!(chat_text(&events, 0, "content"), "\n\nHi\n");
    let calls = chat_calls(&events, 0);
    assert_eq!(calls.len(), 1);
    assert!(calls[0].0.starts_with("call_"), "{calls:?}");
    assert_eq!(calls[0].1, "get_weather");
    assert_eq!(calls[0].2, "{\"city\": \"Paris\"}");
    assert_eq!(finishes(&events, 0), [json!("tool_calls")]);
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert!(last["choices"][0]["delta"]["content"].is_null());
}

#[tokio::test]
async fn chat_stream_reads_each_choice_with_its_own_table_and_finish() {
    // With parallel calls off, each choice returns its own first call.
    let events = chat_events(
        vec![
            chunk(0, "<tool_call>\n{\"name\": \"get_weather\", "),
            chunk(1, "<think>\nhm"),
            chunk(0, "\"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"),
            chunk(1, "</think>\n\nIt is sunny"),
            chunk(0, CALL),
            complete(1, "length"),
            complete(0, "stop"),
        ],
        chat_spec(json!({"tools": tools(), "n": 2, "parallel_tool_calls": false})),
        no_stop(),
    )
    .await;
    assert_eq!(chat_calls(&events, 0).len(), 1);
    assert_eq!(chat_text(&events, 0, "content"), "");
    assert_eq!(chat_text(&events, 1, "reasoning_content"), "\nhm");
    assert_eq!(chat_text(&events, 1, "content"), "\n\nIt is sunny");
    assert!(chat_calls(&events, 1).is_empty());
    assert_eq!(finishes(&events, 0), [json!("tool_calls")]);
    assert_eq!(finishes(&events, 1), [json!("length")]);
}

#[tokio::test]
async fn chat_stream_end_reason_reaches_the_table() {
    // A call cut short keeps what arrived and is not a call made: a stop stays a stop.
    let cut = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pa";
    for (reason, finish) in [("length", "length"), ("stop", "stop"), ("abort", "abort")] {
        let events = chat_events(
            vec![chunk(0, cut), complete(0, reason)],
            chat_spec(json!({"tools": tools()})),
            no_stop(),
        )
        .await;
        let calls = chat_calls(&events, 0);
        assert_eq!(calls.len(), 1, "{reason}");
        assert_eq!(calls[0].2, "{\"city\": \"Pa", "{reason}");
        assert_eq!(finishes(&events, 0), [json!(finish)], "{reason}");
    }
}

#[tokio::test]
async fn chat_stream_local_stop_ends_the_choice_with_stop() {
    // The stop decoder fires first: the engine's later "length" is ignored, the matched stop
    // is reported, and the table is told the stream stopped.
    let events = chat_events(
        vec![
            chunk(
                0,
                "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}",
            ),
            chunk(0, "\n</tool_call>DONE ignored"),
            complete(0, "length"),
        ],
        chat_spec(json!({"tools": tools()})),
        stop_at("DONE"),
    )
    .await;
    assert_eq!(chat_text(&events, 0, "content"), "Checking.\n");
    assert_eq!(chat_calls(&events, 0).len(), 1);
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(last["choices"][0]["matched_stop"], "DONE");
}

#[tokio::test]
async fn chat_stream_a_stop_sequence_inside_a_call_leaves_it_cut() {
    // A stop sequence never completes a call: what arrived is kept, the call is not made, and
    // the finish is the stop the client asked for.
    let events = chat_events(
        vec![
            chunk(
                0,
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par",
            ),
            chunk(0, "isDONE ignored"),
            complete(0, "length"),
        ],
        chat_spec(json!({"tools": tools()})),
        stop_at("DONE"),
    )
    .await;
    let calls = chat_calls(&events, 0);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].2, "{\"city\": \"Paris");
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["choices"][0]["matched_stop"], "DONE");
}

#[tokio::test]
async fn chat_stream_without_tools_streams_a_call_as_content() {
    for extra in [json!({}), json!({"tools": tools(), "tool_choice": "none"})] {
        let events = chat_events(
            vec![chunk(0, "Sure.\n"), chunk(0, CALL), complete(0, "stop")],
            chat_spec(extra.clone()),
            no_stop(),
        )
        .await;
        assert_eq!(
            chat_text(&events, 0, "content"),
            "Sure.\n\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
            "{extra}"
        );
        assert!(chat_calls(&events, 0).is_empty(), "{extra}");
        assert_eq!(finishes(&events, 0), [json!("stop")], "{extra}");
    }
}

#[tokio::test]
async fn chat_stream_returns_one_call_when_parallel_calls_are_off() {
    let events = chat_events(
        vec![chunk(0, CALL), chunk(0, CALL), complete(0, "stop")],
        chat_spec(json!({"tools": tools(), "parallel_tool_calls": false})),
        no_stop(),
    )
    .await;
    assert_eq!(chat_calls(&events, 0).len(), 1);
    assert_eq!(finishes(&events, 0), [json!("tool_calls")]);
}

#[tokio::test]
async fn chat_stream_without_complete_flushes_held_text_and_sends_no_finish() {
    let events = chat_events(
        vec![chunk(0, "answer <tool_")],
        chat_spec(json!({"tools": tools()})),
        no_stop(),
    )
    .await;
    assert_eq!(chat_text(&events, 0, "content"), "answer <tool_");
    assert!(finishes(&events, 0).is_empty());
}

#[tokio::test]
async fn chat_stream_keeps_the_usage_dialects() {
    // Continuous usage rides on every chunk, symphony's included; DeepSeek's aggregate rides on
    // the last finish chunk only.
    let mut last = complete_with_prompt(0, "stop", 10);
    if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
        complete.completion_tokens = 7;
    }
    let events = chat_events(
        vec![chunk(0, "<think>r</think>answer"), last.clone()],
        chat_spec(json!({
            "stream_options": {"include_usage": true, "continuous_usage_stats": true}
        })),
        no_stop(),
    )
    .await;
    for event in &events {
        assert!(event["usage"].is_object(), "{event}");
    }
    assert_eq!(chat_text(&events, 0, "content"), "answer");

    let events = chat_events(
        vec![chunk(0, "<think>r</think>answer"), last],
        chat_spec(json!({"model": "deepseek-flash", "stream_options": {"include_usage": true}})),
        no_stop(),
    )
    .await;
    let (last, rest) = events.split_last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["prompt_tokens"], 10);
    assert!(rest.iter().all(|event| event["usage"].is_null()));
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "r");
}

#[tokio::test]
async fn chat_stream_prefill_decode_reads_the_decode_stream_through_the_table() {
    let (prefill, prefill_server) = scripted_stream(vec![complete(0, "stop")], "0").await;
    let (decode, decode_server) = scripted_stream(
        vec![chunk(0, CALL), chunk(0, CALL), complete(0, "stop")],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let result = processor()
        .process_prefill_decode_streaming_chunks(
            prefill,
            decode,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            no_stop(),
            chat_spec(json!({"tools": tools(), "parallel_tool_calls": false})),
            &tx,
            context::PdTiming {
                prefill_start: Instant::now(),
                decode_sent: Instant::now(),
                runtime: "vllm",
            },
            vec![None],
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    prefill_server.abort();
    decode_server.abort();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_calls(&events, 0).len(), 1);
    assert_eq!(finishes(&events, 0), [json!("tool_calls")]);
}

/// A complete frame that carries the whole output, for the non-streaming paths.
fn complete_with_output(index: u32, text: &str, reason: &str) -> proto::GenerateResponse {
    let mut frame = complete(index, reason);
    if let Some(GenerationEvent::Complete(complete)) = &mut frame.response {
        complete.output_ids = text.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    frame
}

fn response_processor() -> ResponseProcessor {
    ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        resolver(),
    )
}

#[tokio::test]
async fn chat_unary_reads_each_choice_through_the_table() {
    let (stream, server) = scripted_stream(
        vec![
            complete_with_output(0, &format!("<think>\nplan\n</think>\n\n{CALL}"), "stop"),
            complete_with_output(1, "Sunny DONE later", "length"),
        ],
        "0",
    )
    .await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let stop = Some(StringOrArray::Array(vec!["DONE".to_string()]));
    let mut decoder =
        utils::create_stop_decoder(&tokenizer, stop.as_ref(), None, false, false, false);
    let response = response_processor()
        .process_non_streaming_chat_response(
            ExecutionResult::Single { stream },
            chat_spec(json!({"tools": tools(), "n": 2, "stream": false})),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .map_err(|_| "chat response failed")
        .unwrap();
    server.abort();
    let response = serde_json::to_value(response).unwrap();
    let first = &response["choices"][0];
    assert_eq!(first["finish_reason"], "tool_calls");
    assert_eq!(first["message"]["reasoning_content"], "\nplan\n");
    assert!(first["message"]["content"].is_null());
    assert_eq!(
        first["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\": \"Paris\"}"
    );
    let second = &response["choices"][1];
    assert_eq!(second["finish_reason"], "stop");
    assert_eq!(second["matched_stop"], "DONE");
    assert_eq!(second["message"]["content"], "Sunny ");
}

/// A call written without an arguments member.
const BARE_CALL: &str = "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>";

#[tokio::test]
async fn chat_a_call_without_arguments_has_an_empty_object_streamed_and_whole() {
    let events = chat_events(
        vec![
            chunk(0, "<tool_call>\n{\"name\": "),
            chunk(0, "\"get_weather\"}\n</tool_call>"),
            complete(0, "stop"),
        ],
        chat_spec(json!({"tools": tools()})),
        no_stop(),
    )
    .await;
    let calls = chat_calls(&events, 0);
    assert_eq!(calls.len(), 1);
    assert_eq!(
        (calls[0].1.as_str(), calls[0].2.as_str()),
        ("get_weather", "{}")
    );
    assert_eq!(finishes(&events, 0), [json!("tool_calls")]);

    let (stream, server) =
        scripted_stream(vec![complete_with_output(0, BARE_CALL, "stop")], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let response = response_processor()
        .process_non_streaming_chat_response(
            ExecutionResult::Single { stream },
            chat_spec(json!({"tools": tools(), "stream": false})),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .map_err(|_| "chat response failed")
        .unwrap();
    server.abort();
    let response = serde_json::to_value(response).unwrap();
    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(
        choice["message"]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

/// Tools whose integer parameter a tagged call writes at its close, not streamed.
fn typed_tools() -> Value {
    json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}
        }}}
    }])
}

/// Hy4 and DeepSeek V4.1 calls the end of the stream cuts before their first argument fragment:
/// inside an integer value, and right after the invoke's tag.
const CUT_BEFORE_ARGUMENTS: [(&str, &str); 3] = [
    (
        "symphony:hy4",
        "<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>days\
         </arg_key:opensource><arg_value:opensource>3",
    ),
    (
        "symphony:deepseek_v4.1",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
    ),
    (
        "symphony:deepseek_v4.1",
        "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n\
         <｜DSML｜ parameter name=\"days\" string=\"false\">3",
    ),
];

/// One choice's whole Chat response, read by the table `name` selects.
async fn chat_unary(name: &str, output: &str, reason: &str, stop: Option<&str>) -> Value {
    let (stream, server) =
        scripted_stream(vec![complete_with_output(0, output, reason)], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let stop = stop.map(|stop| StringOrArray::Array(vec![stop.to_string()]));
    let mut decoder =
        utils::create_stop_decoder(&tokenizer, stop.as_ref(), None, false, false, false);
    let response = response_processor()
        .process_non_streaming_chat_response(
            ExecutionResult::Single { stream },
            table_chat_spec(name, json!({"tools": typed_tools(), "stream": false})),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .map_err(|_| "chat response failed")
        .unwrap();
    server.abort();
    serde_json::to_value(response).unwrap()["choices"][0].clone()
}

#[tokio::test]
async fn chat_a_call_cut_before_its_arguments_stays_open_and_never_finishes_as_a_call() {
    // The table names the call, and the end of the stream cuts it before any argument: its
    // arguments stay empty rather than read as a complete `{}`, and the choice ends with the
    // engine's own reason, a stop included, never `tool_calls`.
    for (name, cut) in CUT_BEFORE_ARGUMENTS {
        let with_stop = format!("{cut}DONE\"}}");
        for (frames, stop, finish) in [
            (
                vec![chunk(0, cut), complete(0, "length")],
                no_stop(),
                Some("length"),
            ),
            (
                vec![chunk(0, cut), complete(0, "stop")],
                no_stop(),
                Some("stop"),
            ),
            (
                vec![chunk(0, &with_stop), complete(0, "length")],
                stop_at("DONE"),
                Some("stop"),
            ),
            (
                vec![chunk(0, cut), complete(0, "abort")],
                no_stop(),
                Some("abort"),
            ),
            // A stream that ends without a Complete frame: no finish at all.
            (vec![chunk(0, cut)], no_stop(), None),
        ] {
            let events = chat_events(
                frames,
                table_chat_spec(name, json!({"tools": typed_tools()})),
                stop,
            )
            .await;
            let calls = chat_calls(&events, 0);
            assert_eq!(calls.len(), 1, "{name} {finish:?}: {events:?}");
            assert_eq!(
                (calls[0].1.as_str(), calls[0].2.as_str()),
                ("get_weather", ""),
                "{name} {finish:?}"
            );
            let expected: Vec<Value> = finish.into_iter().map(|f| json!(f)).collect();
            assert_eq!(finishes(&events, 0), expected, "{name} {finish:?}");
        }
        for (output, reason, stop, finish) in [
            (cut, "length", None, "length"),
            (cut, "stop", None, "stop"),
            (with_stop.as_str(), "length", Some("DONE"), "stop"),
            (cut, "abort", None, "abort"),
        ] {
            let choice = chat_unary(name, output, reason, stop).await;
            assert_eq!(
                choice["message"]["tool_calls"][0]["function"]["arguments"], "",
                "{name} {reason}: {choice}"
            );
            assert_eq!(choice["finish_reason"], finish, "{name} {reason}");
        }
    }
}

#[tokio::test]
async fn chat_a_tagged_call_without_arguments_has_an_empty_object_streamed_and_whole() {
    for (name, output) in [
        (
            "symphony:hy4",
            "<tool_calls:opensource><tool_call:opensource>get_weather</tool_call:opensource>\
             </tool_calls:opensource>",
        ),
        (
            "symphony:deepseek_v4.1",
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n\
             </｜DSML｜ calls>",
        ),
    ] {
        let events = chat_events(
            vec![chunk(0, output), complete(0, "stop")],
            table_chat_spec(name, json!({"tools": typed_tools()})),
            no_stop(),
        )
        .await;
        let calls = chat_calls(&events, 0);
        assert_eq!(calls.len(), 1, "{name}");
        assert_eq!(calls[0].2, "{}", "{name}");
        assert_eq!(finishes(&events, 0), [json!("tool_calls")], "{name}");
        let choice = chat_unary(name, output, "stop", None).await;
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["arguments"], "{}",
            "{name}"
        );
        assert_eq!(choice["finish_reason"], "tool_calls", "{name}");
    }
}
