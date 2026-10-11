//! Constraints derived from a compiled table: a model whose output a table compiled from its
//! response template reads ([`ParserResolver::template_table`](super::ParserResolver)), and whose
//! table derives the structural tags of its calls ([`Table::tags`]), has its forced tool choices
//! and JSON `response_format`s constrained to its own format, so the reasoning stays and the
//! table reads the output. The tags come from the table's rows ([`AddressedTags`]); no parser
//! name is registered for them.
//!
//! Under tool choice `auto` (without a JSON `response_format`) calls are not constrained, so a
//! tool's `strict: true` is not enforced there; its calls are read back as the model writes them.
//!
//! [`Table::tags`]: crate::routers::grpc::symphony::Table::tags

use axum::response::Response;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, Tool, ToolChoice, ToolChoiceValue},
    messages::{CreateMessageRequest, ToolChoice as MessagesToolChoice},
};
use serde_json::{json, Value};
use symphony::AddressedTags;
use tool_parser::ToolConstraint;

use crate::routers::{error, grpc::symphony::Table};

/// The schema a JSON `response_format` asks for.
fn json_schema(format: Option<&ResponseFormat>) -> Option<Value> {
    match format? {
        ResponseFormat::JsonObject => Some(json!({"type": "object"})),
        ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        ResponseFormat::Text => None,
    }
}

/// Whether a Chat tool choice forces a call: `required`, a named function, or `allowed_tools` in
/// required mode.
fn forces_call(tool_choice: Option<&ToolChoice>) -> bool {
    match tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::Required) | ToolChoice::Function { .. }) => true,
        Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
        _ => false,
    }
}

/// Whether a table's tag frames `request`'s JSON `response_format`: preparation sends the tag in
/// its place, so the backend request must not carry the `response_format` too. The caller tells
/// that the model's table derives tags.
pub(crate) fn frames_response_format(request: &ChatCompletionRequest) -> bool {
    json_schema(request.response_format.as_ref()).is_some()
}

/// The constraint for a Chat request over `callable`, the tools the choice lets the model call:
/// for a forced choice, one or more calls, or one call exactly for a named function or with
/// `parallel_tool_calls: false`; else, for a JSON `response_format`, the thought then an answer
/// holding the JSON, or (choice `auto`) calls to `callable`; else none.
///
/// A forced choice wins over a `response_format` sent with it: the turn that calls a tool has no
/// answer to hold the JSON, and the backend gets no `response_format`
/// ([`frames_response_format`]), so the format applies to a later turn's answer as the client
/// sends it again. `prompt` is the rendered prompt and `continuation` whether it continues the
/// final assistant message, either of which [`check_tag`] refuses.
pub(crate) fn chat_constraint(
    tags: &AddressedTags,
    request: &ChatCompletionRequest,
    callable: &[Tool],
    prompt: &str,
    continuation: bool,
) -> Result<Option<ToolConstraint>, Response> {
    let forced = !callable.is_empty() && forces_call(request.tool_choice.as_ref());
    let schema = json_schema(request.response_format.as_ref());
    let (grammar, tools) = match (forced, schema) {
        (false, None) => return Ok(None),
        (true, _) => {
            let single = matches!(request.tool_choice, Some(ToolChoice::Function { .. }))
                || request.parallel_tool_calls == Some(false);
            let grammar = if single {
                tags.single_call(callable)
            } else {
                tags.calls(callable)
            };
            (grammar, callable)
        }
        (false, Some(schema)) => {
            let tools = response_format_tools(request.tool_choice.as_ref(), callable);
            (tags.response_format(&schema, tools), tools)
        }
    };
    check_tag(tags, &request.model, tools, prompt, continuation)?;
    Ok(Some(ToolConstraint::StructuralTag(
        grammar.payload().to_string(),
    )))
}

/// The constraint for a Messages request over `tools`, the tools its choice lets the model call:
/// for `any` one or more calls, for `tool`, or `any` with `disable_parallel_tool_use: true`, one
/// call exactly; else none. `prompt` and `continuation` as for [`chat_constraint`].
pub(crate) fn messages_constraint(
    tags: &AddressedTags,
    request: &CreateMessageRequest,
    tools: &[Tool],
    prompt: &str,
    continuation: bool,
) -> Result<Option<ToolConstraint>, Response> {
    let grammar = match request.tool_choice {
        _ if tools.is_empty() => return Ok(None),
        Some(
            MessagesToolChoice::Tool { .. }
            | MessagesToolChoice::Any {
                disable_parallel_tool_use: Some(true),
            },
        ) => tags.single_call(tools),
        Some(MessagesToolChoice::Any { .. }) => tags.calls(tools),
        _ => return Ok(None),
    };
    check_tag(tags, &request.model, tools, prompt, continuation)?;
    Ok(Some(ToolConstraint::StructuralTag(
        grammar.payload().to_string(),
    )))
}

/// The tools a JSON `response_format`'s tag lets the model call instead of answering:
/// `callable`, unless the tool choice is `none`.
fn response_format_tools<'a>(tool_choice: Option<&ToolChoice>, callable: &'a [Tool]) -> &'a [Tool] {
    match tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => &[],
        _ => callable,
    }
}

/// Refuses, before any tag is built, a request to `model` whose tag would be too large: for a
/// choice that forces a call to one of `tools`, the tag for one or more calls (one call's, built
/// after it in place of it, is shorter); else, for a JSON `response_format` (Chat only), its tag.
/// First a tool the tag would hold that declares more than
/// [`AddressedTags::MAX_TOOL_PARAMETERS`] parameters, then tools that declare more than
/// [`AddressedTags::MAX_TOTAL_PARAMETERS`] together, then a tag that would take more than
/// [`AddressedTags::MAX_TAG_BYTES`], told by its length alone. A tag holds each parameter's
/// schema, and the tool's definitions again for each one that refers to them.
pub(crate) fn check_tag_size(
    tags: &AddressedTags,
    model: &str,
    tool_choice: Option<&ToolChoice>,
    tools: &[Tool],
    response_format: Option<&ResponseFormat>,
) -> Result<(), Response> {
    let (schema, tools) = if forces_call(tool_choice) && !tools.is_empty() {
        (None, tools)
    } else if let Some(schema) = json_schema(response_format) {
        (Some(schema), response_format_tools(tool_choice, tools))
    } else {
        return Ok(());
    };
    AddressedTags::check_parameter_counts(tools).map_err(|e| {
        error::bad_request(
            "invalid_tool_configuration",
            format!("Invalid tool configuration: {e}"),
        )
    })?;
    let len = match &schema {
        None => tags.calls_len(tools),
        Some(schema) => tags.response_format_len(schema, tools),
    };
    let max = AddressedTags::MAX_TAG_BYTES;
    if len <= max {
        return Ok(());
    }
    Err(if tools.is_empty() {
        error::bad_request(
            "invalid_request_parameters",
            format!(
                "Model '{model}' would constrain this response_format with a structural tag \
                 of {len} bytes, over its limit of {max} bytes"
            ),
        )
    } else {
        error::bad_request(
            "invalid_tool_configuration",
            format!(
                "Invalid tool configuration: model '{model}' would constrain these tools' calls \
                 with a structural tag of {len} bytes, over its limit of {max} bytes; the tag \
                 holds each parameter's schema, and a tool's $defs again for each parameter \
                 that refers to them"
            ),
        )
    })
}

/// The constraint for a Chat request to a model whose table is `table`, after the refusals
/// ([`check_tag_size`], [`check_grammar`], [`check_tag`]), or `None` when the table derives no
/// tags and the request takes the constraint the model's parsers give.
pub(crate) fn for_chat(
    table: &Table,
    request: &ChatCompletionRequest,
    callable: &[Tool],
    prompt: &str,
    continuation: bool,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let tags = table.tags()?;
    let model = &request.model;
    let format = request.response_format.as_ref();
    Some(
        check_tag_size(&tags, model, request.tool_choice.as_ref(), callable, format)
            .and_then(|()| check_grammar(model, request.regex.is_some() || request.ebnf.is_some()))
            .and_then(|()| chat_constraint(&tags, request, callable, prompt, continuation)),
    )
}

/// The constraint for a Messages request to a model whose table is `table`, as [`for_chat`]
/// gives a Chat request's. `tool_choice` is the request's choice as a Chat one.
pub(crate) fn for_messages(
    table: &Table,
    request: &CreateMessageRequest,
    tools: &[Tool],
    tool_choice: Option<&ToolChoice>,
    prompt: &str,
    continuation: bool,
) -> Option<Result<Option<ToolConstraint>, Response>> {
    let tags = table.tags()?;
    Some(
        check_tag_size(&tags, &request.model, tool_choice, tools, None)
            .and_then(|()| messages_constraint(&tags, request, tools, prompt, continuation)),
    )
}

/// Refuses a `regex` or `ebnf` constraint: the engine would apply it from the first token,
/// outside the format the model's table reads.
pub(crate) fn check_grammar(model: &str, grammar: bool) -> Result<(), Response> {
    if !grammar {
        return Ok(());
    }
    Err(error::bad_request(
        "unsupported_constraint",
        format!(
            "Model '{model}' writes its output in the format its response template declares, \
             and a regex or ebnf constraint would apply outside it"
        ),
    ))
}

/// Refuses what the tags cannot constrain: a tag starts a new turn, so the rendered prompt must
/// end with the turn's opener, which a continued assistant message does not; and each of `tools`
/// must be one whose calls the table reads back.
pub(crate) fn check_tag(
    tags: &AddressedTags,
    model: &str,
    tools: &[Tool],
    prompt: &str,
    continuation: bool,
) -> Result<(), Response> {
    if continuation || !prompt.ends_with(tags.generation_prompt()) {
        return Err(error::bad_request(
            "invalid_request_parameters",
            format!(
                "Model '{model}' constrains a forced tool choice or a JSON response_format \
                 from the start of a new assistant message, so it cannot continue one"
            ),
        ));
    }
    tags.check_tools(tools).map_err(|e| {
        error::bad_request(
            "invalid_tool_configuration",
            format!("Invalid tool configuration: {e}"),
        )
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::http::StatusCode;
    use smg_response_template::{load_response_template, to_symphony};
    use symphony::Format;

    use super::*;

    /// The turn opener of [`neutral_template`].
    pub(crate) const OPENER: &str = "<|head|>assistant";

    /// A response template whose calls are addressed to a tool, with neutral markers: a thought,
    /// an answer and calls, each opened after the turn opener; `<eos>` ends a turn that calls.
    pub(crate) fn neutral_template() -> Value {
        json!({
            "start_anchor_pattern": "<\\|head\\|>assistant",
            "fields": {
                "thinking": {
                    "open": "<|via|>notes<|body|>",
                    "close": "<|stop|><|head|>assistant",
                    "content": "text"
                },
                "content": {
                    "open": "<|via|>reply<|body|>",
                    "close": ["<|done|>", "<|stop|>", "<eos>"],
                    "content": "text"
                },
                "tool_calls": {
                    "open_pattern":
                        " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
                    "close": ["<|stop|><|head|>assistant", "<eos>"],
                    "repeats": true,
                    "content": "xml-inline",
                    "content_args": {
                        "tag_pattern":
                            "<arg name=\"(?P<key>[^\"]+)\">\\n(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))",
                        "value_parser": {"name": "text", "args": {"strip": false}}
                    },
                    "transform": {"type": "function",
                                  "function": {"name": "{name}", "arguments": "{content}"}}
                }
            }
        })
    }

    /// [`neutral_template`] compiled, with `<eos>` its stop.
    pub(crate) fn neutral_format() -> Format {
        let template = load_response_template(&neutral_template()).unwrap();
        to_symphony::compile(&template, &["<eos>"])
            .unwrap()
            .stop_tokens(["<eos>"])
    }

    fn tags() -> AddressedTags {
        neutral_format().addressed_tags().unwrap()
    }

    fn named(name: &str) -> Tool {
        serde_json::from_value(json!({"type": "function", "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }}))
        .unwrap()
    }

    /// One tool whose one parameter is a string enum of one value, `len` ASCII bytes long: its
    /// tag grows by the value's length.
    fn tool_of(len: usize) -> Vec<Tool> {
        vec![
            serde_json::from_value(json!({"type": "function", "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"p": {"enum": ["a".repeat(len)]}},
                               "required": ["p"]}
            }}))
            .unwrap(),
        ]
    }

    /// A schema of one string constant, `len` ASCII bytes long, and the response_format that
    /// asks for it.
    fn schema_of(len: usize) -> Value {
        json!({"const": "a".repeat(len)})
    }

    fn format_of(len: usize) -> ResponseFormat {
        serde_json::from_value(json!({"type": "json_schema", "json_schema": {
            "name": "c", "schema": schema_of(len)}}))
        .unwrap()
    }

    async fn code(refused: Response) -> String {
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(refused.into_body(), 1 << 16)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        body["error"]["code"].as_str().unwrap().to_string()
    }

    fn chat(extra: Value) -> ChatCompletionRequest {
        let mut request = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    /// A forced choice gets the tag of one or more calls, a named function or parallel calls off
    /// the tag of one call, a JSON response_format the answer's tag, and choice auto none; the
    /// tag is the payload the table's tags build.
    #[test]
    fn a_forced_choice_or_a_json_format_takes_the_tables_tag() {
        let tags = tags();
        let tools = vec![named("get_weather")];
        let prompt = format!("user{OPENER}");
        let constraint = |extra: Value| {
            chat_constraint(&tags, &chat(extra), &tools, &prompt, false)
                .ok()
                .unwrap()
                .map(|c| c.to_tuple())
        };
        let tag = |grammar: symphony::Grammar| {
            Some(("structural_tag".to_string(), grammar.payload().to_string()))
        };
        assert_eq!(
            constraint(json!({"tool_choice": "required"})),
            tag(tags.calls(&tools))
        );
        assert_eq!(
            constraint(json!({"tool_choice": "required", "parallel_tool_calls": false})),
            tag(tags.single_call(&tools))
        );
        let function = json!({"type": "function", "function": {"name": "get_weather"}});
        assert_eq!(
            constraint(json!({"tool_choice": function})),
            tag(tags.single_call(&tools))
        );
        assert_eq!(
            constraint(json!({"response_format": {"type": "json_object"}})),
            tag(tags.response_format(&json!({"type": "object"}), &tools))
        );
        assert_eq!(
            constraint(json!({"tool_choice": "none", "response_format": {"type": "json_object"}})),
            tag(tags.response_format(&json!({"type": "object"}), &[]))
        );
        assert_eq!(constraint(json!({"tool_choice": "auto"})), None);
    }

    /// A prompt that does not end at the table's turn opener, or continues the final assistant
    /// message, and a tool whose calls the table cannot read back, are refused.
    #[tokio::test]
    async fn a_tag_needs_a_new_turn_and_tools_the_table_reads() {
        let tags = tags();
        let tools = vec![named("get_weather")];
        let prompt = format!("user{OPENER}");
        assert!(check_tag(&tags, "m", &tools, &prompt, false).is_ok());
        for (prompt, continuation) in [(prompt.as_str(), true), ("user", false)] {
            let refused = check_tag(&tags, "m", &tools, prompt, continuation).unwrap_err();
            assert_eq!(code(refused).await, "invalid_request_parameters");
        }
        let refused = check_tag(&tags, "m", &[named("a b")], &prompt, false).unwrap_err();
        assert_eq!(code(refused).await, "invalid_tool_configuration");
        let refused = check_grammar("m", true).unwrap_err();
        assert_eq!(code(refused).await, "unsupported_constraint");
        assert!(check_grammar("m", false).is_ok());
    }

    /// A tag of exactly `MAX_TAG_BYTES` passes and one of a byte more is refused, told by the
    /// tag's length alone: no tag is built. A forced choice's tag holds each call twice (one,
    /// then any more), so its length keeps its parity: there the longest that fits passes and
    /// the next is refused.
    #[tokio::test]
    async fn the_tag_limit_takes_a_tag_of_exactly_its_size() {
        let tags = tags();
        let max = AddressedTags::MAX_TAG_BYTES;
        let required = ToolChoice::Value(ToolChoiceValue::Required);
        let forced = |len: usize| tags.calls_len(&tool_of(len));
        let step = forced(1) - forced(0);
        assert_eq!(step, 2);
        let at = (max - forced(0)) / step;
        assert!(forced(at) <= max && forced(at) + step > max);
        assert!(check_tag_size(&tags, "m", Some(&required), &tool_of(at), None).is_ok());
        let refused =
            check_tag_size(&tags, "m", Some(&required), &tool_of(at + 1), None).unwrap_err();
        assert_eq!(code(refused).await, "invalid_tool_configuration");

        // A response_format alone, and with a tool it may call instead.
        for (tools, refusal) in [
            (Vec::new(), "invalid_request_parameters"),
            (tool_of(3), "invalid_tool_configuration"),
        ] {
            let at = max - tags.response_format_len(&schema_of(0), &tools);
            assert_eq!(tags.response_format_len(&schema_of(at), &tools), max);
            assert!(check_tag_size(&tags, "m", None, &tools, Some(&format_of(at))).is_ok());
            let refused =
                check_tag_size(&tags, "m", None, &tools, Some(&format_of(at + 1))).unwrap_err();
            assert_eq!(code(refused).await, refusal);
        }
    }
}
