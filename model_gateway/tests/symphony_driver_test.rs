//! A model whose parser setting names a symphony table, end to end through a real gRPC worker
//! connection: Chat Completions, Messages and Responses, streamed and whole, read by the table.
//! The same outputs under the legacy parser names show the default left alone.
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test fixtures and helpers: a failure fails the test"
)]

#[path = "common/mod.rs"]
mod common;

#[expect(
    dead_code,
    reason = "the shared fixture also serves the multi-turn tests"
)]
#[path = "common/scripted_tokenizer.rs"]
mod scripted_tokenizer;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::{json, Value};
use smg::{
    app_context::{AppContext, AppContextBuildError},
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};
use tool_parser::ParserFactory as ToolParserFactory;

const MODEL: &str = "symphony-driver-model";
const CALL: &str =
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

/// How the model's parsers are named: by flag, or by the worker's model card.
#[derive(Clone, Copy)]
enum Named {
    Flags(Option<&'static str>, Option<&'static str>),
    Card(&'static str),
}

const QWEN3: Named = Named::Flags(Some("symphony:qwen3"), None);
const LEGACY: Named = Named::Flags(Some("qwen"), Some("qwen3"));

/// A router over one scripted worker whose output is `chunks`, one per token, and which ends with
/// `finish`.
struct Gateway {
    router: Arc<dyn RouterTrait>,
    server: JoinHandle<()>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.server.abort();
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "test fixture: the in-process worker is aborted when the gateway drops"
)]
async fn gateway(named: Named, chunks: &[&str], finish: &'static str) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            // One more token than chunks: the last decodes to nothing, like an end-of-sequence.
            output_tokens: u32::try_from(chunks.len() + 1).unwrap(),
            finish_reasons: vec![finish],
            generation: Default::default(),
        }
        .serve(listener),
    );
    let mut card = ModelCard::new(MODEL);
    if let Named::Card(name) = named {
        card = card.with_tool_parser(name);
    }
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(card)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    if let Named::Flags(tool, reasoning) = named {
        config.tool_call_parser = tool.map(str::to_string);
        config.reasoning_parser = reasoning.map(str::to_string);
    }
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(
            chunks.iter().map(|chunk| chunk.to_string()).collect(),
        ));
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
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    Gateway {
        router: Arc::from(router),
        server,
    }
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("test-tenant"))
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}
        }}}
    }])
}

enum Api {
    Chat,
    Messages,
    Responses,
}

/// Sends `body` and returns the whole response, or the JSON payloads of a stream.
async fn send(gateway: &Gateway, api: Api, body: Value) -> Vec<Value> {
    let stream = body["stream"] == true;
    let response = timeout(Duration::from_secs(30), async {
        let router = &gateway.router;
        match api {
            Api::Chat => {
                let request = serde_json::from_value(body).unwrap();
                router.route_chat(None, &tenant(), request, MODEL).await
            }
            Api::Messages => {
                let request = serde_json::from_value(body).unwrap();
                router.route_messages(None, &tenant(), request, MODEL).await
            }
            Api::Responses => {
                let request = serde_json::from_value(body).unwrap();
                router
                    .route_responses(None, &tenant(), request, MODEL)
                    .await
            }
        }
    })
    .await
    .expect("the response ends");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    if !stream {
        return vec![serde_json::from_str(&text).unwrap()];
    }
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

/// A Chat response's message and finish, from the whole response or folded from its stream.
fn chat_message(payloads: &[Value]) -> (Value, Value) {
    if let [response] = payloads {
        if response["object"] == "chat.completion" {
            let choice = &response["choices"][0];
            return (choice["message"].clone(), choice["finish_reason"].clone());
        }
    }
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut finish = Value::Null;
    for payload in payloads {
        let Some(choice) = payload["choices"].get(0) else {
            continue;
        };
        let delta = &choice["delta"];
        content.push_str(delta["content"].as_str().unwrap_or_default());
        reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            if call["id"].is_string() {
                calls.push(json!({
                    "id": call["id"],
                    "function": {"name": call["function"]["name"], "arguments": ""},
                }));
            }
            let slot = &mut calls[call["index"].as_u64().unwrap() as usize];
            let joined = format!(
                "{}{}",
                slot["function"]["arguments"].as_str().unwrap(),
                call["function"]["arguments"].as_str().unwrap_or_default()
            );
            slot["function"]["arguments"] = Value::String(joined);
        }
        if !choice["finish_reason"].is_null() {
            finish = choice["finish_reason"].clone();
        }
    }
    let mut message = json!({"role": "assistant"});
    if !content.trim().is_empty() {
        message["content"] = Value::String(content);
    }
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    (message, finish)
}

/// The call names and parsed arguments of a Chat message.
fn calls(message: &Value) -> Vec<(String, Value)> {
    message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|call| {
            (
                call["function"]["name"].as_str().unwrap().to_string(),
                serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                    .unwrap_or(Value::Null),
            )
        })
        .collect()
}

#[tokio::test]
async fn chat_is_read_by_the_selected_table_streamed_and_whole() {
    let chunks = ["<think>\nplan\n</think>\n\n", "Let me look.\n", CALL];
    for named in [
        QWEN3,
        Named::Flags(None, Some("symphony:qwen3")),
        Named::Card("symphony:qwen3"),
    ] {
        let gateway = gateway(named, &chunks, "stop").await;
        for stream in [false, true] {
            let payloads = send(
                &gateway,
                Api::Chat,
                json!({
                    "model": MODEL, "stream": stream, "tools": tools(),
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                }),
            )
            .await;
            let (message, finish) = chat_message(&payloads);
            assert_eq!(message["reasoning_content"], "\nplan\n", "{payloads:?}");
            assert_eq!(message["content"], "\n\nLet me look.\n", "{payloads:?}");
            assert_eq!(
                calls(&message),
                [("get_weather".to_string(), json!({"city": "Paris"}))]
            );
            assert!(message["tool_calls"][0]["id"]
                .as_str()
                .unwrap()
                .starts_with("call_"));
            assert_eq!(finish, "tool_calls");
            // Reasoning left in the content: the text the model wrote, its markers included, as
            // SGLang returns it without a reasoning parser; the returned call takes its own
            // text and markers out.
            let payloads = send(
                &gateway,
                Api::Chat,
                json!({
                    "model": MODEL, "stream": stream, "tools": tools(),
                    "separate_reasoning": false,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                }),
            )
            .await;
            let (message, _) = chat_message(&payloads);
            assert_eq!(
                message["content"], "<think>\nplan\n</think>\n\nLet me look.\n",
                "{payloads:?}"
            );
            assert_eq!(calls(&message).len(), 1);
        }
    }
}

#[tokio::test]
async fn the_engine_s_end_reason_reaches_the_table() {
    // The same cut call keeps what arrived and is not a call made: the engine's finish stands.
    let chunks = [
        "<tool_call>\n{\"name\": \"get_weather\", ",
        "\"arguments\": {\"city\": \"Pa",
    ];
    for (finish, expected) in [("length", "length"), ("stop", "stop"), ("abort", "abort")] {
        let gateway = gateway(QWEN3, &chunks, finish).await;
        for stream in [false, true] {
            let payloads = send(
                &gateway,
                Api::Chat,
                json!({
                    "model": MODEL, "stream": stream, "tools": tools(),
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                }),
            )
            .await;
            let (message, reason) = chat_message(&payloads);
            assert_eq!(reason, expected, "{finish} stream={stream}");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"Pa"
            );
        }
    }
}

#[tokio::test]
async fn a_local_stop_sequence_ends_the_table_s_stream_with_stop() {
    // The stop decoder matches before the engine's "length": the output stopped on a stop
    // sequence, which never completes a call, so the call it cut keeps what arrived and the
    // finish is `stop`.
    let chunks = [
        "Checking.\n<tool_call>\n{\"name\": \"get_weather\", ",
        "\"arguments\": {\"city\": \"Pa",
        "DONE",
        "ris\"}}\n</tool_call>",
    ];
    let gateway = gateway(QWEN3, &chunks, "length").await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "stop": ["DONE"], "tools": tools(),
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(message["content"], "Checking.\n", "stream={stream}");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"], "{\"city\": \"Pa",
            "stream={stream}"
        );
        assert_eq!(finish, "stop", "stream={stream}");
    }
}

#[tokio::test]
async fn chat_policy_applies_to_the_table_s_events() {
    let gateway = gateway(QWEN3, &["<think>\nplan\n</think>\n\n", CALL, CALL], "stop").await;
    for stream in [false, true] {
        // Parallel calls off: the first call only.
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "tools": tools(), "parallel_tool_calls": false,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(calls(&message).len(), 1, "stream={stream}");
        assert_eq!(finish, "tool_calls");
        // Parallel calls off, reasoning not separated: the first call is returned and takes its
        // own bytes out of the raw text; the call dropped after it is text, as the model wrote it.
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "tools": tools(), "parallel_tool_calls": false,
                "separate_reasoning": false,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(calls(&message).len(), 1, "stream={stream}");
        assert_eq!(
            message["content"],
            format!("<think>\nplan\n</think>\n\n{CALL}"),
            "stream={stream}"
        );
        assert_eq!(finish, "tool_calls");
        // Tool choice none, reasoning not separated: the raw text the model wrote.
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "tools": tools(), "tool_choice": "none",
                "separate_reasoning": false,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(
            message["content"],
            format!("<think>\nplan\n</think>\n\n{CALL}{CALL}"),
            "stream={stream}"
        );
        assert!(message["tool_calls"].is_null());
        assert!(message["reasoning_content"].is_null());
        assert_eq!(finish, "stop");
    }
}

#[tokio::test]
async fn the_legacy_names_keep_the_legacy_parsers() {
    // The same output under the names the gateway takes today: the call is the same call, and
    // the parts the legacy parsers render differently stay as they were.
    let chunks = ["<think>\nplan\n</think>\n\n", "Let me look.\n", CALL];
    let legacy = gateway(LEGACY, &chunks, "stop").await;
    let table = gateway(QWEN3, &chunks, "stop").await;
    for stream in [false, true] {
        let body = json!({
            "model": MODEL, "stream": stream, "tools": tools(),
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
        });
        let (old, old_finish) = chat_message(&send(&legacy, Api::Chat, body.clone()).await);
        let (new, new_finish) = chat_message(&send(&table, Api::Chat, body).await);
        assert_eq!(calls(&old), calls(&new), "stream={stream}");
        assert_eq!(old_finish, new_finish);
        assert_eq!(
            old["reasoning_content"].as_str().map(str::trim),
            Some("plan")
        );
    }
    // Where they differ: a call the request turned off is the model's raw text for the legacy
    // parsers, markers included.
    let body = json!({
        "model": MODEL, "stream": false, "tools": tools(), "tool_choice": "none",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
    });
    let (old, _) = chat_message(&send(&legacy, Api::Chat, body.clone()).await);
    let (new, _) = chat_message(&send(&table, Api::Chat, body).await);
    assert!(old["content"].as_str().unwrap().contains("<tool_call>"));
    assert!(!new["content"].as_str().unwrap().contains("<tool_call>"));
}

#[tokio::test]
async fn a_forced_call_without_a_structural_tag_stays_with_the_legacy_reader() {
    // Qwen forces a call by JSON schema, so the model writes the bare arguments; the table is
    // not armed and the legacy reader returns the call.
    let gateway = gateway(QWEN3, &["{\"city\": \"Paris\"}"], "stop").await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "tools": tools(),
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, _) = chat_message(&payloads);
        assert_eq!(
            calls(&message),
            [("get_weather".to_string(), json!({"city": "Paris"}))],
            "stream={stream}"
        );
    }
}

#[tokio::test]
async fn a_forced_call_under_a_structural_tag_is_read_by_the_table() {
    // Hy4 forces a call with the structural tag its legacy parser builds, so the model writes
    // its own call syntax and the table reads it.
    // With parallel calls off, the table returns the first of the two.
    let output = [
        "<tool_calls:opensource><tool_call:opensource>get_weather",
        "<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris",
        "</arg_value:opensource></tool_call:opensource>",
        "<tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>",
        "<arg_value:opensource>Lyon</arg_value:opensource></tool_call:opensource>",
        "</tool_calls:opensource>",
    ];
    let gateway = gateway(Named::Flags(Some("symphony:hy4"), None), &output, "stop").await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "tools": tools(), "parallel_tool_calls": false,
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(
            calls(&message),
            [("get_weather".to_string(), json!({"city": "Paris"}))],
            "stream={stream}"
        );
        assert_eq!(finish, "tool_calls");
        assert!(message["content"].is_null(), "{message}");
    }
}

#[tokio::test]
async fn a_json_response_format_stays_with_the_table_as_content() {
    // A constraint on the content is not a tool constraint: the table reads the output, here
    // without reasoning separation, so the content is the raw text, the thought's markers
    // included.
    let gateway = gateway(
        QWEN3,
        &["<think>\n</think>\n\n", "{\"answer\": ", "42}"],
        "stop",
    )
    .await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Chat,
            json!({
                "model": MODEL, "stream": stream, "separate_reasoning": false,
                "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {
                    "type": "object", "properties": {"answer": {"type": "integer"}}
                }}},
                "messages": [{"role": "user", "content": "The answer?"}],
            }),
        )
        .await;
        let (message, finish) = chat_message(&payloads);
        assert_eq!(
            message["content"], "<think>\n</think>\n\n{\"answer\": 42}",
            "stream={stream}"
        );
        assert_eq!(finish, "stop");
    }
}

/// A Messages response's blocks as `(type, text or name, input)`, and its stop reason.
fn messages_blocks(payloads: &[Value]) -> (Vec<(String, String, Value)>, Value) {
    if let [message] = payloads {
        if message["type"] == "message" {
            let blocks = message["content"]
                .as_array()
                .unwrap()
                .iter()
                .map(|block| {
                    let text = block["text"]
                        .as_str()
                        .or_else(|| block["thinking"].as_str())
                        .or_else(|| block["name"].as_str())
                        .unwrap_or_default();
                    (
                        block["type"].as_str().unwrap().to_string(),
                        text.to_string(),
                        block["input"].clone(),
                    )
                })
                .collect();
            return (blocks, message["stop_reason"].clone());
        }
    }
    let mut blocks: Vec<(String, String, Value)> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut stop = Value::Null;
    for payload in payloads {
        match payload["type"].as_str().unwrap() {
            "content_block_start" => {
                let block = &payload["content_block"];
                let name = block["name"].as_str().unwrap_or_default().to_string();
                let input = if block["type"] == "tool_use" {
                    json!({})
                } else {
                    Value::Null
                };
                blocks.push((block["type"].as_str().unwrap().to_string(), name, input));
                inputs.push(String::new());
            }
            "content_block_delta" => {
                let delta = &payload["delta"];
                let block = blocks.last_mut().unwrap();
                for key in ["text", "thinking"] {
                    block.1.push_str(delta[key].as_str().unwrap_or_default());
                }
                inputs
                    .last_mut()
                    .unwrap()
                    .push_str(delta["partial_json"].as_str().unwrap_or_default());
            }
            "content_block_stop" => {
                let block = blocks.last_mut().unwrap();
                if block.0 == "tool_use" {
                    block.2 = serde_json::from_str(inputs.last().unwrap()).unwrap_or(json!({}));
                }
            }
            "message_delta" => stop = payload["delta"]["stop_reason"].clone(),
            _ => {}
        }
    }
    (blocks, stop)
}

#[tokio::test]
async fn messages_are_read_by_the_selected_table_streamed_and_whole() {
    let gateway = gateway(
        QWEN3,
        &["<think>\nplan\n</think>\n\n", "Let me look.\n", CALL],
        "stop",
    )
    .await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Messages,
            json!({
                "model": MODEL, "max_tokens": 64, "stream": stream,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {
                    "city": {"type": "string"}
                }}}],
            }),
        )
        .await;
        let (blocks, stop) = messages_blocks(&payloads);
        assert_eq!(
            blocks,
            [
                ("thinking".to_string(), "\nplan\n".to_string(), Value::Null),
                (
                    "text".to_string(),
                    "\n\nLet me look.\n".to_string(),
                    Value::Null
                ),
                (
                    "tool_use".to_string(),
                    "get_weather".to_string(),
                    json!({"city": "Paris"})
                ),
            ],
            "stream={stream}"
        );
        assert_eq!(stop, "tool_use");
    }
}

#[tokio::test]
async fn messages_end_reasons_reach_the_table() {
    let cut = [
        "<tool_call>\n{\"name\": \"get_weather\", ",
        "\"arguments\": {\"city\": \"Pa",
    ];
    // A call cut short is not a call made: no end turns the turn into tool_use.
    for (finish, expected) in [
        ("length", "max_tokens"),
        ("stop", "end_turn"),
        ("abort", "end_turn"),
    ] {
        let gateway = gateway(QWEN3, &cut, finish).await;
        for stream in [false, true] {
            let payloads = send(
                &gateway,
                Api::Messages,
                json!({
                    "model": MODEL, "max_tokens": 64, "stream": stream,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                }),
            )
            .await;
            let (blocks, stop) = messages_blocks(&payloads);
            assert_eq!(stop, expected, "{finish} stream={stream}");
            // The cut arguments are not a JSON value.
            assert_eq!(
                blocks,
                [("tool_use".to_string(), "get_weather".to_string(), json!({}))]
            );
        }
    }
}

/// A Responses result's output items as `(type, summary)`, and its status.
fn responses_items(payloads: &[Value]) -> (Vec<(String, Value)>, Value) {
    let response = match payloads {
        [response] if response["object"] == "response" => response.clone(),
        _ => payloads.last().unwrap()["response"].clone(),
    };
    let items = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| {
            let summary = match item["type"].as_str().unwrap() {
                "function_call" => json!([
                    item["name"],
                    serde_json::from_str::<Value>(item["arguments"].as_str().unwrap())
                        .unwrap_or(Value::Null)
                ]),
                "message" => item["content"][0]["text"].clone(),
                "reasoning" => item["content"][0]["text"].clone(),
                _ => Value::Null,
            };
            (item["type"].as_str().unwrap().to_string(), summary)
        })
        .collect();
    (items, response["status"].clone())
}

#[tokio::test]
async fn responses_are_read_by_the_selected_table_streamed_and_whole() {
    // With parallel calls off, the table returns the first of the two calls.
    let chunks = ["<think>\nplan\n</think>\n\n", "Let me look.\n", CALL, CALL];
    let gateway = gateway(QWEN3, &chunks, "stop").await;
    for stream in [false, true] {
        let payloads = send(
            &gateway,
            Api::Responses,
            json!({
                "model": MODEL, "input": "Weather in Paris?", "stream": stream, "store": false,
                "max_output_tokens": 64, "parallel_tool_calls": false,
                "tools": [{"type": "function", "name": "get_weather", "parameters": {
                    "type": "object", "properties": {"city": {"type": "string"}}
                }}],
            }),
        )
        .await;
        let (items, status) = responses_items(&payloads);
        assert_eq!(status, "completed", "{payloads:?}");
        let calls: Vec<_> = items
            .iter()
            .filter(|(kind, _)| kind == "function_call")
            .collect();
        assert_eq!(calls.len(), 1, "{items:?}");
        let call = calls[0];
        assert_eq!(
            call.1,
            json!(["get_weather", {"city": "Paris"}]),
            "stream={stream}"
        );
        // The markers the table consumed never reach the output.
        let rendered = serde_json::to_string(&payloads).unwrap();
        assert!(!rendered.contains("<tool_call>"), "stream={stream}");
        assert!(!rendered.contains("<think>"), "stream={stream}");
    }
}

const HY4: Named = Named::Flags(Some("symphony:hy4"), None);
const DSML: Named = Named::Flags(Some("symphony:deepseek_v4.1"), None);

#[tokio::test]
async fn a_call_without_arguments_has_an_empty_object_on_every_api() {
    // A JSON call without an arguments member gets the driver's `{}`; a tagged call closed
    // with no value gets its table's.
    let calls: [(Named, &[&str]); 3] = [
        (
            QWEN3,
            &["<tool_call>\n{\"name\": ", "\"get_weather\"}\n</tool_call>"],
        ),
        (
            HY4,
            &[
                "<tool_calls:opensource><tool_call:opensource>get_weather",
                "</tool_call:opensource></tool_calls:opensource>",
            ],
        ),
        (
            DSML,
            &[
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
                "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            ],
        ),
    ];
    for (named, chunks) in calls {
        let gateway = gateway(named, chunks, "stop").await;
        for stream in [false, true] {
            let payloads = send(
                &gateway,
                Api::Chat,
                json!({
                    "model": MODEL, "stream": stream, "tools": tools(),
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                }),
            )
            .await;
            let (message, finish) = chat_message(&payloads);
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"], "{}",
                "chat {chunks:?} stream={stream}"
            );
            assert_eq!(finish, "tool_calls");

            let payloads = send(
                &gateway,
                Api::Responses,
                json!({
                    "model": MODEL, "input": "Weather in Paris?", "stream": stream,
                    "store": false, "max_output_tokens": 64,
                    "tools": [{"type": "function", "name": "get_weather", "parameters": {
                        "type": "object", "properties": {"city": {"type": "string"}}
                    }}],
                }),
            )
            .await;
            assert_eq!(
                function_call(&payloads)["arguments"],
                "{}",
                "responses {chunks:?} stream={stream}"
            );

            let payloads = send(
                &gateway,
                Api::Messages,
                json!({
                    "model": MODEL, "max_tokens": 64, "stream": stream,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                }),
            )
            .await;
            let (blocks, stop) = messages_blocks(&payloads);
            assert_eq!(
                blocks,
                [("tool_use".to_string(), "get_weather".to_string(), json!({}))],
                "messages {chunks:?} stream={stream}"
            );
            assert_eq!(stop, "tool_use");
        }
    }
}

/// A Responses result's first function call item, from the whole response or the stream's last
/// event.
fn function_call(payloads: &[Value]) -> Value {
    let response = match payloads {
        [response] if response["object"] == "response" => response.clone(),
        _ => payloads.last().unwrap()["response"].clone(),
    };
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("a function call")
        .clone()
}

#[tokio::test]
async fn a_call_cut_before_its_arguments_stays_open_and_never_finishes_as_a_call() {
    // Hy4 and DeepSeek V4.1 calls the stream ends before their first argument fragment: inside
    // an integer value, right after the invoke's tag, inside an integer parameter.
    let cuts: [(Named, &[&str]); 3] = [
        (
            HY4,
            &[
                "<tool_calls:opensource><tool_call:opensource>get_weather",
                "<arg_key:opensource>days</arg_key:opensource>",
                "<arg_value:opensource>3",
            ],
        ),
        (
            DSML,
            &[
                "<｜DSML｜ calls>\n",
                "<｜DSML｜ invoke name=\"get_weather\">\n",
            ],
        ),
        (
            DSML,
            &[
                "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
                "<｜DSML｜ parameter name=\"days\" string=\"false\">3",
            ],
        ),
    ];
    let properties = json!({"city": {"type": "string"}, "days": {"type": "integer"}});
    for (named, cut) in cuts {
        let with_stop: Vec<&str> = cut.iter().copied().chain(["DONE", "\"}"]).collect();
        // The engine's length and stop, a stop string the gateway matches, and an abort.
        for (chunks, finish, stop, chat_finish, messages_stop) in [
            (cut, "length", None, "length", "max_tokens"),
            (cut, "stop", None, "stop", "end_turn"),
            (
                with_stop.as_slice(),
                "length",
                Some("DONE"),
                "stop",
                "stop_sequence",
            ),
            (cut, "abort", None, "abort", "end_turn"),
        ] {
            let gateway = gateway(named, chunks, finish).await;
            for stream in [false, true] {
                let label = format!("{cut:?} {finish} stop={stop:?} stream={stream}");
                let mut body = json!({
                    "model": MODEL, "stream": stream,
                    "tools": [{"type": "function", "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": properties},
                    }}],
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                });
                if let Some(stop) = stop {
                    body["stop"] = json!([stop]);
                }
                let (message, reason) = chat_message(&send(&gateway, Api::Chat, body).await);
                assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
                assert_eq!(
                    message["tool_calls"][0]["function"]["arguments"], "",
                    "chat {label}"
                );
                assert_eq!(reason, chat_finish, "chat {label}");

                // Responses has no stop strings.
                if stop.is_none() {
                    let payloads = send(
                        &gateway,
                        Api::Responses,
                        json!({
                            "model": MODEL, "input": "Weather in Paris?", "stream": stream,
                            "store": false, "max_output_tokens": 64,
                            "tools": [{"type": "function", "name": "get_weather", "parameters": {
                                "type": "object", "properties": properties
                            }}],
                        }),
                    )
                    .await;
                    let call = function_call(&payloads);
                    assert_eq!(call["arguments"], "", "responses {label}");
                    // Arguments that are not JSON: the call and the response are incomplete.
                    assert_eq!(call["status"], "incomplete", "responses {label}");
                    assert_eq!(
                        responses_items(&payloads).1,
                        "incomplete",
                        "responses {label}"
                    );
                }

                let mut body = json!({
                    "model": MODEL, "max_tokens": 64, "stream": stream,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    "tools": [{"name": "get_weather", "input_schema": {
                        "type": "object", "properties": properties
                    }}],
                });
                if let Some(stop) = stop {
                    body["stop_sequences"] = json!([stop]);
                }
                let (blocks, stop_reason) =
                    messages_blocks(&send(&gateway, Api::Messages, body).await);
                let calls: Vec<_> = blocks
                    .iter()
                    .filter(|block| block.0 == "tool_use")
                    .collect();
                assert_eq!(calls.len(), 1, "messages {label}: {blocks:?}");
                assert_eq!(stop_reason, messages_stop, "messages {label}");
            }
        }
    }
}

#[tokio::test]
async fn a_call_cut_after_its_first_argument_is_returned_as_it_arrived_and_not_made() {
    // A Hy4 call the stream ends inside its string value: Chat returns what arrived with the
    // engine's finish, Messages ends the turn without tool_use, and Responses marks the call
    // and the response incomplete, whatever ended the stream.
    let cut: &[&str] = &[
        "<tool_calls:opensource><tool_call:opensource>get_weather",
        "<arg_key:opensource>city</arg_key:opensource>",
        "<arg_value:opensource>Par",
    ];
    let properties = json!({"city": {"type": "string"}, "days": {"type": "integer"}});
    for (finish, chat_finish, messages_stop, details) in [
        (
            "length",
            "length",
            "max_tokens",
            json!({"reason": "max_output_tokens"}),
        ),
        ("stop", "stop", "end_turn", Value::Null),
        ("abort", "abort", "end_turn", Value::Null),
    ] {
        let gateway = gateway(HY4, cut, finish).await;
        for stream in [false, true] {
            let label = format!("{finish} stream={stream}");
            let (message, reason) = chat_message(
                &send(
                    &gateway,
                    Api::Chat,
                    json!({
                        "model": MODEL, "stream": stream,
                        "tools": [{"type": "function", "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": properties},
                        }}],
                        "messages": [{"role": "user", "content": "Weather in Paris?"}],
                    }),
                )
                .await,
            );
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"], "{\"city\": \"Par",
                "chat {label}"
            );
            assert_eq!(reason, chat_finish, "chat {label}");

            let payloads = send(
                &gateway,
                Api::Responses,
                json!({
                    "model": MODEL, "input": "Weather in Paris?", "stream": stream,
                    "store": false, "max_output_tokens": 64,
                    "tools": [{"type": "function", "name": "get_weather", "parameters": {
                        "type": "object", "properties": properties
                    }}],
                }),
            )
            .await;
            let call = function_call(&payloads);
            assert_eq!(call["arguments"], "{\"city\": \"Par", "responses {label}");
            assert_eq!(call["status"], "incomplete", "responses {label}");
            let response = match payloads.as_slice() {
                [response] if response["object"] == "response" => response.clone(),
                _ => payloads.last().unwrap()["response"].clone(),
            };
            assert_eq!(response["status"], "incomplete", "responses {label}");
            assert_eq!(response["incomplete_details"], details, "responses {label}");
            if stream {
                assert_eq!(
                    payloads.last().unwrap()["type"],
                    "response.incomplete",
                    "responses {label}"
                );
            }

            let (blocks, stop_reason) = messages_blocks(
                &send(
                    &gateway,
                    Api::Messages,
                    json!({
                        "model": MODEL, "max_tokens": 64, "stream": stream,
                        "messages": [{"role": "user", "content": "Weather in Paris?"}],
                        "tools": [{"name": "get_weather", "input_schema": {
                            "type": "object", "properties": properties
                        }}],
                    }),
                )
                .await,
            );
            assert_eq!(
                blocks.iter().filter(|block| block.0 == "tool_use").count(),
                1,
                "messages {label}: {blocks:?}"
            );
            assert_eq!(stop_reason, messages_stop, "messages {label}");
        }
    }
}

/// Two tools, for the turns that call both.
fn two_tools(api: &Api) -> Value {
    let weather = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let time = json!({"type": "object", "properties": {"tz": {"type": "string"}}});
    match api {
        Api::Chat => json!([
            {"type": "function", "function": {"name": "get_weather", "parameters": weather}},
            {"type": "function", "function": {"name": "get_time", "parameters": time}},
        ]),
        Api::Messages => json!([
            {"name": "get_weather", "input_schema": weather},
            {"name": "get_time", "input_schema": time},
        ]),
        Api::Responses => json!([
            {"type": "function", "name": "get_weather", "parameters": weather},
            {"type": "function", "name": "get_time", "parameters": time},
        ]),
    }
}

#[tokio::test]
async fn a_turn_whose_second_call_is_cut_short_is_no_tool_turn_on_any_api() {
    // A whole call, then a second one the stream ends inside its value: both are returned as
    // they arrived, and no API reports a tool turn, so a client runs neither. The end is an
    // abort, the token limit or a stop sequence of the request's.
    let first = format!("{CALL}\n");
    let cut = [
        first.as_str(),
        "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UT",
    ];
    for (finish, chat_finish, messages_stop, details) in [
        ("abort", "abort", "end_turn", Value::Null),
        (
            "length",
            "length",
            "max_tokens",
            json!({"reason": "max_output_tokens"}),
        ),
    ] {
        let gateway = gateway(QWEN3, &cut, finish).await;
        for stream in [false, true] {
            let label = format!("{finish} stream={stream}");
            let (message, reason) = chat_message(
                &send(
                    &gateway,
                    Api::Chat,
                    json!({
                        "model": MODEL, "stream": stream, "tools": two_tools(&Api::Chat),
                        "messages": [{"role": "user", "content": "Weather and time?"}],
                    }),
                )
                .await,
            );
            let calls = message["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 2, "chat {label}");
            assert_eq!(
                calls[0]["function"]["arguments"], "{\"city\": \"Paris\"}",
                "chat {label}"
            );
            assert_eq!(
                calls[1]["function"]["arguments"], "{\"tz\": \"UT",
                "chat {label}"
            );
            assert_eq!(reason, chat_finish, "chat {label}");

            let (blocks, stop_reason) = messages_blocks(
                &send(
                    &gateway,
                    Api::Messages,
                    json!({
                        "model": MODEL, "max_tokens": 64, "stream": stream,
                        "tools": two_tools(&Api::Messages),
                        "messages": [{"role": "user", "content": "Weather and time?"}],
                    }),
                )
                .await,
            );
            assert_eq!(
                blocks.iter().filter(|block| block.0 == "tool_use").count(),
                2,
                "messages {label}: {blocks:?}"
            );
            assert_eq!(stop_reason, messages_stop, "messages {label}");

            let payloads = send(
                &gateway,
                Api::Responses,
                json!({
                    "model": MODEL, "input": "Weather and time?", "stream": stream,
                    "store": false, "max_output_tokens": 64,
                    "tools": two_tools(&Api::Responses),
                }),
            )
            .await;
            let response = match payloads.as_slice() {
                [response] if response["object"] == "response" => response.clone(),
                _ => payloads.last().unwrap()["response"].clone(),
            };
            let statuses: Vec<_> = response["output"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|item| item["type"] == "function_call")
                .map(|item| (item["arguments"].clone(), item["status"].clone()))
                .collect();
            assert_eq!(
                statuses,
                [
                    (json!("{\"city\": \"Paris\"}"), json!("completed")),
                    (json!("{\"tz\": \"UT"), json!("incomplete")),
                ],
                "responses {label}"
            );
            assert_eq!(response["status"], "incomplete", "responses {label}");
            assert_eq!(response["incomplete_details"], details, "responses {label}");
        }
    }

    // A stop sequence matched inside the second call's value (the stop decoder's match comes
    // before the engine's own finish): Chat's finish is `stop` and Messages' `stop_sequence`.
    let stopped = [
        first.as_str(),
        "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"tz\": \"U",
        "TC",
        "\"}}\n</tool_call>",
    ];
    let gateway = gateway(QWEN3, &stopped, "length").await;
    for stream in [false, true] {
        let (message, reason) = chat_message(
            &send(
                &gateway,
                Api::Chat,
                json!({
                    "model": MODEL, "stream": stream, "stop": ["TC"],
                    "tools": two_tools(&Api::Chat),
                    "messages": [{"role": "user", "content": "Weather and time?"}],
                }),
            )
            .await,
        );
        let calls = message["tool_calls"].as_array().expect("calls");
        assert_eq!(calls.len(), 2, "chat stream={stream}");
        assert_eq!(
            calls[1]["function"]["arguments"], "{\"tz\": \"U",
            "chat stream={stream}"
        );
        assert_eq!(reason, "stop", "chat stream={stream}");

        let payloads = send(
            &gateway,
            Api::Messages,
            json!({
                "model": MODEL, "max_tokens": 64, "stream": stream, "stop_sequences": ["TC"],
                "tools": two_tools(&Api::Messages),
                "messages": [{"role": "user", "content": "Weather and time?"}],
            }),
        )
        .await;
        let (blocks, stop_reason) = messages_blocks(&payloads);
        assert_eq!(
            blocks.iter().filter(|block| block.0 == "tool_use").count(),
            2,
            "messages stream={stream}: {blocks:?}"
        );
        assert_eq!(stop_reason, "stop_sequence", "messages stream={stream}");
        let stop_sequence = match payloads.as_slice() {
            [message] if message["type"] == "message" => message["stop_sequence"].clone(),
            _ => payloads
                .iter()
                .find(|payload| payload["type"] == "message_delta")
                .map(|payload| payload["delta"]["stop_sequence"].clone())
                .unwrap_or_default(),
        };
        assert_eq!(stop_sequence, "TC", "messages stream={stream}");
    }
}

#[tokio::test]
async fn startup_accepts_a_known_table_and_refuses_an_unknown_one() {
    for (tool, reasoning, accepted) in [
        (Some("symphony:qwen3"), None, true),
        (None, Some("symphony:deepseek_v4.1"), true),
        (Some("symphony:nope"), None, false),
        (None, Some("symphony:nope"), false),
    ] {
        let mut config = RouterConfig::builder().build_unchecked();
        config.tool_call_parser = tool.map(str::to_string);
        config.reasoning_parser = reasoning.map(str::to_string);
        let built = AppContext::builder()
            .router_config(config)
            .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
            .tool_parser_factory(Some(ToolParserFactory::new()))
            .build();
        // The parser names are checked before anything else is required.
        match built {
            Err(AppContextBuildError::InvalidConfig(message)) => {
                assert!(!accepted, "{tool:?} {reasoning:?}: {message}");
                assert!(message.contains("symphony:nope"), "{message}");
            }
            Err(AppContextBuildError::MissingField(_)) => {
                assert!(accepted, "{tool:?} {reasoning:?}");
            }
            Ok(_) => panic!("a context without its registries must not build"),
        }
    }
}
