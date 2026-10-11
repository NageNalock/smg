//! The structural tags of a table whose calls are addressed to a tool, derived from its rows.
//! `fixtures/addressed/addressed.json` holds tools, renders of assistant turns in a synthetic
//! spelling of Harmony's channels with tagged parameters, and the tags built for them;
//! `fixtures/addressed/check_xgrammar.py` checks those tags against xgrammar with a tokenizer
//! that holds the spelling's markers as tokens. Here: the tables of three row layouts of that
//! spelling build exactly those tags, and read each render back into its message.

#![expect(clippy::unwrap_used, reason = "test helpers")]

use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    time::{Duration, Instant},
};

use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};
use symphony::{
    adapt,
    tagged::{addressed::Spelling, Declared},
    AddressedTags, CallSyntax, Emits, Engine, EngineFinish, Events, Format, Input, Parser,
};

const FIXTURE: &str = include_str!("fixtures/addressed/addressed.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

const SPELLING: Spelling = Spelling {
    header_close: Cow::Borrowed("<|channel|>commentary <|constrain|>xml<|message|>"),
    key_open: Cow::Borrowed("<parameter name=\""),
    key_close: Cow::Borrowed("\">"),
    value_close: Cow::Borrowed("</parameter>"),
    next_key_closes: true,
};

const TOKENS: [&str; 7] = [
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
];

const STOPS: [&str; 2] = ["<|return|>", "<|call|>"];

/// Messages close at the next message's opening: a header state and a state between messages,
/// each holding only markers.
fn split_header() -> Format {
    Format::new("split_header")
        .terminal("start", "<|start|>assistant")
        .terminal("analysis", "<|channel|>analysis<|message|>")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("recipient", " to=")
        .terminal("next", "<|end|><|start|>assistant")
        .terminal("end", "<|end|>")
        .terminal("return", "<|return|>")
        .terminal("call", "<|call|>")
        .state("header", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("between", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("header", "analysis", "reasoning")
        .transition("header", "final", "content")
        .transition("header", "recipient", "calls")
        .transition("reasoning", "next", "header")
        .transition("content", "next", "header")
        .transition("content", "end", "between")
        .transition("content", "return", "between")
        .transition("content", "call", "between")
        .transition("calls", "next", "header")
        .transition("calls", "call", "between")
        .transition("between", "start", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .closes_at_stop("calls")
        .marker_only("header")
        .marker_only("between")
        .opens_turn("<|start|>assistant")
        .marker_tokens(TOKENS)
        .stop_tokens(STOPS)
}

/// The same with one state outside every message, as a template compiler writes it: each field
/// opens from it and closes back to it.
fn one_outside() -> Format {
    Format::new("one_outside")
        .terminal("start", "<|start|>assistant")
        .terminal("analysis", "<|channel|>analysis<|message|>")
        .terminal("next", "<|end|><|start|>assistant")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("return", "<|return|>")
        .terminal("end", "<|end|>")
        .terminal("call", "<|call|>")
        .terminal("recipient", " to=")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("outside", "start", "outside")
        .transition("outside", "analysis", "reasoning")
        .transition("reasoning", "next", "outside")
        .transition("outside", "final", "content")
        .transition("content", "return", "outside")
        .transition("content", "end", "outside")
        .transition("content", "call", "outside")
        .transition("outside", "recipient", "calls")
        .transition("calls", "next", "outside")
        .transition("calls", "call", "outside")
        .calls(CallSyntax::Addressed(SPELLING))
        .closes_at_stop("calls")
        .marker_only("outside")
        .opens_turn("<|start|>assistant")
        .marker_tokens(TOKENS)
        .stop_tokens(STOPS)
}

/// Harmony's own layout: every message ends at `<|end|>`, and the next opens with its own row.
fn channels() -> Format {
    Format::new("channels")
        .terminal("start", "<|start|>assistant")
        .terminal("analysis", "<|channel|>analysis<|message|>")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("recipient", " to=")
        .terminal("end", "<|end|>")
        .terminal("return", "<|return|>")
        .terminal("call", "<|call|>")
        .state("header", Emits::Wrapper)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("between", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "analysis", "reasoning")
        .transition("header", "final", "content")
        .transition("header", "recipient", "calls")
        .transition("reasoning", "end", "between")
        .transition("content", "end", "between")
        .transition("content", "return", "between")
        .transition("calls", "end", "between")
        .transition("calls", "call", "between")
        .transition("between", "start", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .closes_at_stop("calls")
        .opens_turn("<|start|>assistant")
        .marker_tokens(TOKENS)
        .stop_tokens(STOPS)
}

fn tags() -> AddressedTags {
    one_outside().addressed_tags().unwrap()
}

/// The fixture's tools named in `names`, in fixture order.
fn tools(fixture: &Value, names: &Value) -> Vec<Tool> {
    let names: Vec<&str> = names
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    let all: Vec<Tool> = serde_json::from_value(fixture["tools"].clone()).unwrap();
    all.into_iter()
        .filter(|tool| names.contains(&tool.function.name.as_str()))
        .collect()
}

fn tool(parameters: Value) -> Tool {
    serde_json::from_value(json!({
        "type": "function", "function": {"name": "f", "parameters": parameters}
    }))
    .unwrap()
}

/// `ADDRESSED_BLESS=1` rewrites the tags in the fixture; run `check_xgrammar.py` on them then, as
/// only it checks them with xgrammar. The three row layouts build the same tags.
#[test]
fn the_tables_build_the_checked_tags() {
    let mut fixture = fixture();
    let mut changed = Vec::new();
    let cases = fixture["tags"].as_array().unwrap().clone();
    for format in [one_outside(), split_header(), channels()] {
        let tags = format.addressed_tags().unwrap();
        assert_eq!(tags.generation_prompt(), "<|start|>assistant");
        for (index, case) in cases.iter().enumerate() {
            let tools = tools(&fixture, &case["tools"]);
            let tag = match case["kind"].as_str().unwrap() {
                "tool" => tags.calls(&tools),
                "single_call" => tags.single_call(&tools),
                _ => tags.response_format(&case["schema"], &tools),
            }
            .payload();
            if tag != case["tag"] {
                changed.push(format!("{} {}", format.name(), case["name"]));
                fixture["tags"][index]["tag"] = tag;
            }
        }
    }
    if std::env::var_os("ADDRESSED_BLESS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/addressed/addressed.json"
        );
        let text = serde_json::to_string_pretty(&fixture).unwrap();
        std::fs::write(path, text + "\n").unwrap();
        return;
    }
    assert!(
        changed.is_empty(),
        "tags differ from the fixture: {changed:?}"
    );
}

/// The table's own grammar is the forced tag; a choice that forces no call, or a prompt left
/// inside the thought, has none.
#[test]
fn the_tables_grammar_is_the_forced_tag() {
    let fixture = fixture();
    let all: Vec<Tool> = serde_json::from_value(fixture["tools"].clone()).unwrap();
    let format = one_outside();
    assert_eq!(format.grammar(&all, true, false), Some(tags().calls(&all)));
    assert_eq!(format.grammar(&all, false, false), None);
    assert_eq!(format.grammar(&all, true, true), None);
    assert_eq!(format.grammar(&[], true, false), None);
}

/// Each render is read back by each table into the message it renders.
#[test]
fn each_render_reads_back_into_its_message() {
    let fixture = fixture();
    let all: Vec<Tool> = serde_json::from_value(fixture["tools"].clone()).unwrap();
    for format in [one_outside(), split_header(), channels()] {
        for render in fixture["renders"].as_array().unwrap() {
            let text = render["text"].as_str().unwrap();
            // The stop decoder hides the stop token that ends the turn.
            let text = STOPS
                .iter()
                .find_map(|stop| text.strip_suffix(stop))
                .unwrap_or(text);
            let mut engine = Engine::new(format.clone(), Declared::of(&all));
            let mut out = Events::new();
            engine
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: "<|start|>user<|message|>Hi<|end|><|start|>assistant",
                    },
                    &mut out,
                )
                .unwrap();
            engine
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text,
                        spans: &[],
                    },
                    &mut out,
                )
                .unwrap();
            engine
                .feed(
                    Input::End {
                        finish: EngineFinish::Stop,
                    },
                    &mut out,
                )
                .unwrap();
            let events = out.drain();
            let choice = adapt::chat::message(0, &events);
            let message = &render["message"];
            let case = format!("{} {}", format.name(), render["name"]);
            let thinking = message["thinking"].as_str().unwrap_or_default();
            assert_eq!(
                choice.message.reasoning_content.unwrap_or_default(),
                thinking,
                "{case}"
            );
            assert_eq!(
                choice.message.content.as_deref(),
                message["content"].as_str(),
                "{case}"
            );
            let calls: Vec<(String, Value)> = choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    let arguments = call.function.arguments.unwrap_or_default();
                    (
                        call.function.name,
                        serde_json::from_str(&arguments).unwrap(),
                    )
                })
                .collect();
            let expected: Vec<(String, Value)> = message["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|call| {
                    (
                        call["function"]["name"].as_str().unwrap().to_string(),
                        call["function"]["arguments"].clone(),
                    )
                })
                .collect();
            assert_eq!(calls, expected, "{case}");
        }
    }
}

/// A table names what keeps it from giving tags: calls that are not addressed, no turn opener,
/// no stop token among the rows out of a call, no row from one call to the next, or words that
/// end a value without one character they all begin with.
#[test]
fn a_table_without_the_rows_gives_no_tags_and_says_why() {
    let refused = |format: Format| format.addressed_tags().unwrap_err();
    assert!(refused(symphony::formats::glm()).contains("not addressed"));
    let no_opener = Format::new("no_opener")
        .terminal("recipient", " to=")
        .terminal("call", "<|call|>")
        .state("header", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "recipient", "calls")
        .transition("calls", "call", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .stop_tokens(STOPS);
    assert!(refused(no_opener).contains("turn opener"));
    let more_stops = one_outside().stop_tokens(["<|endoftext|>"]);
    assert!(more_stops.addressed_tags().is_ok(), "stops add up");
    let unstopped = Format::new("unstopped")
        .terminal("recipient", " to=")
        .terminal("call", "<|call|>")
        .state("header", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "recipient", "calls")
        .transition("calls", "call", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .opens_turn("<|start|>assistant");
    assert!(refused(unstopped.clone()).contains("stop token"));
    // One call and its stop, but no row to a second call.
    let one_call = unstopped.stop_tokens(STOPS);
    assert!(refused(one_call).contains("leads to another call"));
    // A word that ends a value and begins otherwise than the others.
    let other_lead = Format::new("other_lead")
        .terminal("recipient", " to=")
        .terminal("end", "<|end|>")
        .terminal("call", "[CALL]")
        .state("header", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "recipient", "calls")
        .transition("calls", "end", "header")
        .transition("calls", "call", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .opens_turn("<|start|>assistant")
        .stop_tokens(["[CALL]"]);
    assert!(refused(other_lead).contains("begin with one character"));
}

/// Only the tokens and stops the table's markers hold are kept; without tokens, every marker is
/// a constant string, and the stop alone still ends the calls.
#[test]
fn a_table_keeps_the_tokens_and_stops_its_markers_hold() {
    let plain = Format::new("plain")
        .terminal("start", "<|start|>assistant")
        .terminal("recipient", " to=")
        .terminal("end", "<|end|>")
        .terminal("call", "<|call|>")
        .state("header", Emits::Wrapper)
        .state("between", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "recipient", "calls")
        .transition("calls", "end", "between")
        .transition("calls", "call", "between")
        .transition("between", "start", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .opens_turn("<|start|>assistant")
        .stop_tokens(["<|call|>", "<|return|>", "<|end|><|start|>"]);
    let tag = plain
        .addressed_tags()
        .unwrap()
        .calls(&[tool(json!({}))])
        .payload();
    assert!(!tag.to_string().contains("\"token\""), "{tag}");
    // No thought: the first call opens at once, the others after `<|end|><|start|>assistant`.
    assert_eq!(
        tag,
        json!({"format": {"type": "sequence", "elements": [
            {"type": "const_string", "value": " to=f<|channel|>commentary <|constrain|>xml<|message|>"},
            {"type": "star", "content": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "<|end|><|start|>assistant"},
                {"type": "const_string", "value": " to=f<|channel|>commentary <|constrain|>xml<|message|>"},
            ]}},
            {"type": "const_string", "value": "<|call|>"},
        ]}})
    );
    // With tokens, only those the markers hold.
    let tokens = plain
        .marker_tokens(["<|end|>", "<|unused|>", "<|call|>", "<|constrain|>", ""])
        .addressed_tags()
        .unwrap()
        .single_call(&[tool(json!({}))])
        .payload();
    assert_eq!(
        tokens,
        json!({"format": {"type": "sequence", "elements": [
            {"type": "sequence", "elements": [
                {"type": "const_string", "value": " to=f<|channel|>commentary "},
                {"type": "token", "token": "<|constrain|>"},
                {"type": "const_string", "value": "xml<|message|>"},
            ]},
            {"type": "token", "token": "<|call|>"},
        ]}})
    );
}

/// Refuses tools whose calls the tags cannot write so that the table reads them back.
#[test]
fn tools_the_format_cannot_write_are_refused() {
    let tags = tags();
    let object = json!({"type": "object", "properties": {"city": {"type": "string"}}});
    let named = |name: &str, parameters: &Value| -> Tool {
        serde_json::from_value(json!({
            "type": "function", "function": {"name": name, "parameters": parameters}
        }))
        .unwrap()
    };
    for name in [
        "get_weather",
        "_x",
        "ns.tool-v2",
        "A1",
        "pkg.tool",
        "1st",
        "über",
        "a/b",
    ] {
        assert!(tags.check_tools(&[named(name, &object)]).is_ok(), "{name}");
    }
    assert!(tags.check_tools(&[tool(json!({}))]).is_ok());
    for name in ["", "has space", "tab\tname", "nb\u{a0}sp", "a<b"] {
        let error = tags.check_tools(&[named(name, &object)]).unwrap_err();
        assert!(error.contains("cannot be read back"), "{name}: {error}");
    }
    for parameters in [
        json!({"type": "string"}),
        json!(true),
        json!({"type": "array"}),
    ] {
        let error = tags.check_tools(&[tool(parameters.clone())]).unwrap_err();
        assert!(
            error.contains("object parameters schema"),
            "{parameters}: {error}"
        );
    }
    for key in ["", "say \"hi\"", "a<|end|>", "x</parameter>"] {
        let parameters = json!({"type": "object", "properties": {key: {"type": "string"}}});
        let error = tags.check_tools(&[tool(parameters)]).unwrap_err();
        assert!(error.contains("parameter name"), "{key:?}: {error}");
    }
    let parameters = json!({"properties": {"mode": {"enum": ["a", "b<|call|>"]}}});
    let error = tags.check_tools(&[tool(parameters)]).unwrap_err();
    assert!(error.contains("delimiter"), "{error}");
    let parameters = json!({"properties": {"mode": {"enum": []}}});
    let error = tags.check_tools(&[tool(parameters)]).unwrap_err();
    assert!(error.contains("empty enum"), "{error}");
    // A required parameter the tag cannot write would be left out.
    for parameters in [
        json!({"properties": {"a": {}}, "required": ["a", "b"]}),
        json!({"required": ["a"]}),
        json!({"properties": {"a": {}}, "required": [1]}),
    ] {
        let error = tags.check_tools(&[tool(parameters.clone())]).unwrap_err();
        assert!(
            error.contains("not among its properties"),
            "{parameters}: {error}"
        );
    }
}

/// The body of the one call of a single-tool tag: the element after the header's close, if any.
fn body(parameters: Value) -> Option<Value> {
    let tag = tags().calls(&[tool(parameters)]).payload();
    let call = &tag["format"]["elements"][1];
    call["elements"].as_array().unwrap().get(7).cloned()
}

/// The value format of the only parameter `p`.
fn value_of(schema: Value) -> Value {
    let parameters = json!({"type": "object", "properties": {"p": schema}, "required": ["p"]});
    body(parameters).unwrap()["content"].clone()
}

/// A value the parser keeps as a string is free text, a string enum lists its values, and any
/// other value is JSON of its schema, with the root definitions only where it refers into them.
#[test]
fn values_follow_what_the_parser_reads_back() {
    let text = json!({"type": "any_text", "excludes": [
        "</parameter>", "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
        "<|constrain|>", "<|return|>", "<|call|>"
    ]});
    for schema in [
        json!({"type": "string"}),
        json!({"type": ["integer", "string"]}),
        json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
        json!({"description": "anything"}),
        json!({"type": "string", "pattern": "^a+$"}),
        json!({"allOf": [{"type": "string"}], "description": "wrapped"}),
    ] {
        assert_eq!(value_of(schema.clone()), text, "{schema}");
    }
    let one_of = |values: &[&str]| -> Value {
        let elements: Vec<Value> = values
            .iter()
            .map(|v| json!({"type": "const_string", "value": v}))
            .collect();
        match elements.as_slice() {
            [one] => one.clone(),
            _ => json!({"type": "or", "elements": elements}),
        }
    };
    assert_eq!(value_of(json!({"enum": ["c", "f"]})), one_of(&["c", "f"]));
    assert_eq!(value_of(json!({"const": "only"})), one_of(&["only"]));
    for schema in [
        json!({"type": "integer"}),
        json!({"type": ["number", "null"]}),
        json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        json!({"type": "array", "items": {"type": "integer"}}),
        json!({"enum": [1, 2]}),
    ] {
        let expected = json!({"type": "json_schema", "json_schema": schema});
        assert_eq!(value_of(schema.clone()), expected, "{schema}");
    }

    let point = json!({"type": "object", "properties": {"x": {"type": "integer"}}});
    let defs = json!({"Point": point, "Line": {"type": "array", "items": {"$ref": "#/$defs/Point"}},
                      "Name": {"type": "string", "enum": ["a"]}});
    let with_defs = |property: Value| {
        let parameters = json!({
            "type": "object", "properties": {"p": property}, "required": ["p"], "$defs": defs
        });
        body(parameters).unwrap()["content"].clone()
    };
    assert_eq!(
        with_defs(json!({"$ref": "#/$defs/Point"})),
        json!({"type": "json_schema", "json_schema": {"$ref": "#/$defs/Point", "$defs": defs}})
    );
    let nested = json!({"type": "array", "items": {"$ref": "#/$defs/Line"}});
    let mut expected = nested.clone();
    expected["$defs"] = defs.clone();
    assert_eq!(
        with_defs(nested),
        json!({"type": "json_schema", "json_schema": expected})
    );
    // A value that refers to none carries no definitions.
    assert_eq!(
        with_defs(json!({"type": "integer"})),
        json!({"type": "json_schema", "json_schema": {"type": "integer"}})
    );
    assert_eq!(with_defs(json!({"$ref": "#/$defs/Name"})), text);
    assert_eq!(with_defs(json!({"$ref": "#/$defs/Missing"})), text);
    // An `allOf` of one schema is that schema, as Pydantic wraps a described enum or model.
    assert_eq!(
        with_defs(json!({"allOf": [{"$ref": "#/$defs/Name"}]})),
        text
    );
    let wrapped = with_defs(json!({"allOf": [{"$ref": "#/$defs/Point"}]}));
    assert_eq!(wrapped["type"], "json_schema");
}

/// References that branch into shared definitions level after level are each followed once to
/// tell whether a value may be a string, so the work stays in proportion to the schema.
#[test]
fn each_reference_is_followed_once() {
    let refs = |name: &str| -> Vec<Value> {
        (0..1000)
            .map(|_| json!({"$ref": format!("#/$defs/{name}")}))
            .collect()
    };
    let parameters = json!({
        "type": "object",
        "$defs": {
            "A": {"anyOf": refs("B")},
            "B": {"anyOf": refs("C")},
            "C": {"anyOf": refs("D")},
            "D": {"type": "integer"}
        },
        "properties": {"p": {"$ref": "#/$defs/A"}},
        "required": ["p"]
    });
    assert_eq!(body(parameters).unwrap()["content"]["type"], "json_schema");
}

/// Parameters keep the schema's order; a `\n` joins them: each optional parameter before the
/// first required one is written with the `\n` after it, and each one after it with the `\n`
/// before it.
#[test]
fn parameters_keep_the_schema_order() {
    let value = json!({"type": "const_string", "value": "v"});
    let tag = |key: &str, after_another: bool| {
        let lead = if after_another { "\n" } else { "" };
        json!({"type": "tag", "begin": format!("{lead}<parameter name=\"{key}\">\n"),
               "content": value, "end": "\n</parameter>"})
    };
    let optional = |content: Value| json!({"type": "optional", "content": content});
    let sequence = |elements: Vec<Value>| json!({"type": "sequence", "elements": elements});
    let or = |elements: Vec<Value>| json!({"type": "or", "elements": elements});
    let newline = json!({"type": "const_string", "value": "\n"});
    let parameters = |keys: &[&str], required: &[&str]| {
        let properties: Map<String, Value> = keys
            .iter()
            .map(|key| ((*key).to_owned(), json!({"const": "v"})))
            .collect();
        json!({"type": "object", "properties": properties, "required": required})
    };

    assert_eq!(
        body(parameters(&["a", "b", "c", "d"], &["b", "d"])).unwrap(),
        sequence(vec![
            optional(sequence(vec![tag("a", false), newline.clone()])),
            tag("b", false),
            optional(tag("c", true)),
            tag("d", true),
        ])
    );
    assert_eq!(body(parameters(&["a"], &["a"])).unwrap(), tag("a", false));
    assert_eq!(
        body(parameters(&["a"], &[])).unwrap(),
        optional(tag("a", false))
    );
    // All optional: one or more of the first half then the second half each after another, or
    // one or more of the second half.
    let either = |a: &str, b: &str| {
        or(vec![
            sequence(vec![tag(a, false), optional(tag(b, true))]),
            tag(b, false),
        ])
    };
    assert_eq!(
        body(parameters(&["a", "b"], &[])).unwrap(),
        optional(either("a", "b"))
    );
    assert_eq!(
        body(parameters(&["a", "b", "c"], &[])).unwrap(),
        optional(or(vec![
            sequence(vec![either("a", "b"), optional(tag("c", true))]),
            tag("c", false),
        ]))
    );
    assert_eq!(
        body(parameters(&["a", "b", "c", "d"], &[])).unwrap(),
        optional(or(vec![
            sequence(vec![
                either("a", "b"),
                optional(tag("c", true)),
                optional(tag("d", true))
            ]),
            either("c", "d"),
        ]))
    );

    assert_eq!(body(json!({})), None);
    assert_eq!(body(json!({"type": "object", "properties": {}})), None);
}

/// The ends of the matches of `format` in `text` from `start`, for the elements a call body of
/// constant values is made of.
fn ends(format: &Value, text: &str, start: usize) -> BTreeSet<usize> {
    let after = |at: BTreeSet<usize>, format: &Value| -> BTreeSet<usize> {
        at.into_iter()
            .flat_map(|start| ends(format, text, start))
            .collect()
    };
    let literal = |value: &Value| {
        let value = value.as_str().unwrap();
        text[start..]
            .starts_with(value)
            .then_some(start + value.len())
            .into_iter()
            .collect::<BTreeSet<usize>>()
    };
    match format["type"].as_str().unwrap() {
        "const_string" => literal(&format["value"]),
        "sequence" => format["elements"]
            .as_array()
            .unwrap()
            .iter()
            .fold(BTreeSet::from([start]), after),
        "or" => format["elements"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|element| ends(element, text, start))
            .collect(),
        "optional" => {
            let mut out = ends(&format["content"], text, start);
            out.insert(start);
            out
        }
        "tag" => {
            let begin = literal(&format["begin"]);
            let content = after(begin, &format["content"]);
            let end = json!({"type": "const_string", "value": format["end"]});
            after(content, &end)
        }
        // No other element is in a body of constant values; were one there, nothing would
        // match, and the lists taken would fail.
        _ => BTreeSet::new(),
    }
}

/// A call body takes exactly the parameter lists the table reads back: the parameters in schema
/// order, each required one, any of the others, joined by one `\n`; and nothing else. Every subset
/// of up to 7 parameters under every set of required ones, and the near misses of each list taken.
#[test]
fn a_body_takes_exactly_the_lists_of_parameters() {
    let list = |keys: &[String]| -> String {
        keys.iter()
            .map(|key| format!("<parameter name=\"{key}\">\nv{key}\n</parameter>"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut checked = 0;
    for n in 1..=7usize {
        let keys: Vec<String> = (0..n).map(|i| format!("p{i}")).collect();
        let properties: Map<String, Value> = keys
            .iter()
            .map(|key| (key.clone(), json!({"const": format!("v{key}")})))
            .collect();
        for required in 0..1u32 << n {
            let has = |set: u32, i: usize| set >> i & 1 == 1;
            let names: Vec<&String> = (0..n)
                .filter(|&i| has(required, i))
                .map(|i| &keys[i])
                .collect();
            let body = body(json!({"type": "object", "properties": properties,
                                   "required": names}))
            .unwrap();
            let takes = |text: &str| ends(&body, text, 0).contains(&text.len());
            for subset in 0..1u32 << n {
                let chosen: Vec<String> = (0..n)
                    .filter(|&i| has(subset, i))
                    .map(|i| keys[i].clone())
                    .collect();
                let text = list(&chosen);
                let expected = subset & required == required;
                assert_eq!(
                    takes(&text),
                    expected,
                    "n={n} required={required:b} {text:?}"
                );
                checked += 1;
                if !expected {
                    continue;
                }
                let mut misses = vec![format!("\n{text}"), format!("{text}\n")];
                if chosen.len() >= 2 {
                    let (first, rest) = chosen.split_at(1);
                    misses.push(format!("{}\n\n{}", list(first), list(rest)));
                    misses.push(format!("{}{}", list(first), list(rest)));
                    let mut swapped = chosen.clone();
                    swapped.swap(0, 1);
                    misses.push(list(&swapped));
                    let mut repeated = chosen.clone();
                    repeated.insert(1, chosen[0].clone());
                    misses.push(list(&repeated));
                }
                for miss in misses {
                    assert!(!takes(&miss), "n={n} required={required:b} {miss:?}");
                }
            }
        }
    }
    assert_eq!(
        checked,
        (1..=7).map(|n: u32| 1usize << (2 * n)).sum::<usize>()
    );
}

/// A tag holds each parameter's value once when the tool requires one, and at most 1 + ⌈log₂ n⌉
/// times for n parameters all optional, nested about as deep: in proportion to the schema, give
/// or take that logarithm.
#[test]
fn a_body_holds_each_value_a_few_times() {
    fn copies(element: &Value, counts: &mut HashMap<String, usize>) {
        match element {
            Value::Object(map) => {
                if let Some(begin) = map.get("begin").and_then(Value::as_str) {
                    let key = begin.trim_start_matches('\n');
                    *counts.entry(key.to_owned()).or_default() += 1;
                }
                map.values().for_each(|v| copies(v, counts));
            }
            Value::Array(items) => items.iter().for_each(|v| copies(v, counts)),
            _ => {}
        }
    }
    fn depth(element: &Value) -> usize {
        match element {
            Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
            Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
            _ => 0,
        }
    }
    for n in (1..=64usize).chain([100, 255, 256, 257, 300]) {
        let properties: Map<String, Value> = (0..n)
            .map(|i| (format!("p{i}"), json!({"type": "integer"})))
            .collect();
        let log = usize::BITS as usize - (n - 1).leading_zeros() as usize;
        for required in [None, Some(0), Some(n / 2), Some(n - 1)] {
            let required: Vec<String> = required.into_iter().map(|i| format!("p{i}")).collect();
            let body = body(json!({"type": "object", "properties": properties,
                                   "required": required}))
            .unwrap();
            let mut counts = HashMap::new();
            copies(&body, &mut counts);
            assert_eq!(counts.len(), n);
            let most = counts.values().copied().max().unwrap();
            let total: usize = counts.values().sum();
            if required.is_empty() {
                assert!(most <= 1 + log, "n={n}: {most}");
                assert!(total <= n * (2 + log) / 2 + n, "n={n}: {total}");
                assert!(depth(&body) <= 6 + 4 * log, "n={n}: {}", depth(&body));
            } else {
                assert_eq!((most, total), (1, n), "n={n} {required:?}");
                assert!(depth(&body) <= 8, "n={n}: {}", depth(&body));
            }
        }
    }
}

/// A seeded generator, so the random cases are the same on every run.
struct Random(u64);

impl Random {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }

    fn pick(&mut self, items: &[Value]) -> Value {
        items[self.below(items.len())].clone()
    }
}

/// A random schema of what the tags tell apart: strings written as text and in JSON (with other
/// keys, in union types), enums, unions, `allOf`s, references into `$defs`, `definitions` or
/// nowhere, a value's own `$defs` (an object or not), open and closed objects and maps (with or
/// without declared, required or counted keys, or their own patterns), keywords read before a
/// type, arrays, any JSON, and text that JSON escapes.
fn random_schema(random: &mut Random, depth: usize) -> Value {
    let nested = |random: &mut Random| random_schema(random, depth + 1);
    match random.below(if depth >= 3 { 12 } else { 30 }) {
        0 => json!({"type": "string"}),
        1 => json!({"type": "string", "description": "a \"quoted\" <|end|> line\n"}),
        2 => json!({"type": ["string", "null"]}),
        3 => json!({"type": "integer", "minimum": 0}),
        4 => json!({"type": "number"}),
        5 => json!({"type": "boolean"}),
        6 => json!({"enum": ["a", "b é", "c\"d"]}),
        7 => random.pick(&[json!({"enum": [1, 2.5, null]}), json!({"const": 3})]),
        8 => json!({"const": "only"}),
        9 => random.pick(&[json!(true), json!({}), json!({"description": "anything"})]),
        10 => random.pick(&[
            json!({"$ref": "#/$defs/A"}),
            json!({"$ref": "#/$defs/B"}),
            json!({"$ref": "#/definitions/C"}),
            json!({"$ref": "#/$defs/Missing"}),
        ]),
        11 => json!({"type": "array", "items": {"$ref": "#/$defs/A"}}),
        12 | 13 => {
            let mut properties = Map::new();
            for index in 0..1 + random.below(3) {
                properties.insert(format!("k{index}"), nested(random));
            }
            let mut object = json!({"type": "object", "properties": properties,
                                    "required": ["k0"]});
            match random.below(4) {
                0 => object["additionalProperties"] = json!(false),
                1 => object["additionalProperties"] = json!(true),
                2 => object["additionalProperties"] = nested(random),
                _ => {}
            }
            object
        }
        14 => random.pick(&[json!({"type": "object"}), json!({"type": "array"})]),
        15 => json!({"type": "array", "items": nested(random)}),
        16 => json!({"prefixItems": [nested(random), nested(random)]}),
        17 => json!({"anyOf": [nested(random), nested(random)]}),
        18 => json!({"oneOf": [nested(random), {"type": "null"}]}),
        19 => json!({"allOf": [nested(random)], "description": "wrapped"}),
        20 => json!({"allOf": [nested(random), nested(random)]}),
        21 => json!({"not": nested(random)}),
        22 => match random.below(3) {
            0 => {
                json!({"patternProperties": {"^x": nested(random)}, "unevaluatedProperties": false})
            }
            // An open object whose own pattern is empty.
            _ => json!({"type": "object", "patternProperties": {"": nested(random)}}),
        },
        23 => {
            let own = random.pick(&[json!("Own"), json!("symphony_json")]);
            json!({"$ref": "#/$defs/Own", "$defs": {"Own": nested(random), own.as_str().unwrap(): {}}})
        }
        24 => match random.below(8) {
            0 => json!({"type": "string", "maxLength": 3}),
            1 => json!({"type": ["string", "object"]}),
            2 => json!({"type": ["array", "object"], "properties": {"k": nested(random)}}),
            3 => json!({"additionalProperties": nested(random)}),
            4 => json!({"unevaluatedProperties": nested(random),
                        "additionalProperties": nested(random)}),
            5 => json!({"type": "object", "patternProperties": [1]}),
            6 => json!({"items": [nested(random)], "allOf": [{"type": "array"}]}),
            _ => json!({"type": "object", "unevaluatedProperties": false,
                        "additionalProperties": true}),
        },
        25 => json!({"$ref": "#/$defs/A", "$defs": [1], "definitions": {"D": nested(random)}}),
        26 => match random.below(4) {
            0 => json!({"type": "string", "nullable": true}),
            1 => json!({"type": ["string", "integer"], "minimum": 0}),
            2 => json!({"type": ["array", "string", "null"], "items": nested(random)}),
            _ => json!({"type": "string", "properties": {"k": nested(random)}}),
        },
        27 => match random.below(4) {
            0 => json!({"type": "object", "additionalProperties": nested(random)}),
            1 => json!({"additionalProperties": {"additionalProperties": nested(random)}}),
            2 => json!({"type": "object", "properties": {}, "unevaluatedProperties": true}),
            _ => json!({"type": ["object", "string"], "properties": {"k": nested(random)},
                        "additionalProperties": nested(random)}),
        },
        28 => match random.below(4) {
            0 => json!({"type": "object", "required": ["k"]}),
            1 => json!({"type": "object", "maxProperties": 2, "additionalProperties": true}),
            2 => json!({"type": "object", "patternProperties": {"^a": nested(random)},
                        "additionalProperties": nested(random)}),
            _ => json!({"type": "object", "properties": {"a": nested(random)},
                        "propertyNames": {"maxLength": 3}, "additionalProperties": true}),
        },
        _ => match random.below(3) {
            0 => json!({"$ref": "#/$defs/A", "type": "string"}),
            1 => json!({"anyOf": [nested(random)], "type": ["string", "object"]}),
            _ => json!({"enum": ["x", 1], "type": "object", "additionalProperties": true}),
        },
    }
}

/// A random tool: up to six parameters, some required (one maybe not declared), root definitions
/// of each kind, maybe names the tags' own definitions take, and now and then parameters that are
/// no object.
fn random_tool(random: &mut Random, index: usize) -> Tool {
    let mut properties = Map::new();
    for key in 0..random.below(7) {
        let key = random.pick(&[json!(format!("p{key}")), json!(format!("ü\t\"{key}"))]);
        properties.insert(key.as_str().unwrap().to_owned(), random_schema(random, 0));
    }
    let keys: Vec<String> = properties.keys().cloned().collect();
    let mut required: Vec<Value> = keys
        .iter()
        .filter(|_| random.below(3) == 0)
        .map(|key| json!(key))
        .collect();
    if random.below(8) == 0 {
        required.push(json!("undeclared"));
    }
    let mut parameters = json!({"type": "object", "properties": properties,
                                "required": required});
    let mut defs = Map::new();
    defs.insert("A".to_owned(), random_schema(random, 1));
    defs.insert(
        "B".to_owned(),
        json!({"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}),
    );
    if random.below(4) == 0 {
        defs.insert("symphony_string".to_owned(), json!({"type": "integer"}));
        defs.insert("symphony_json".to_owned(), json!({"type": "integer"}));
    }
    match random.below(5) {
        0 => {}
        1 => parameters["$defs"] = json!("not an object"),
        _ => parameters["$defs"] = Value::Object(defs),
    }
    match random.below(3) {
        0 => parameters["definitions"] = json!({"C": random_schema(random, 1)}),
        1 => parameters["definitions"] = json!(5),
        _ => {}
    }
    if random.below(10) == 0 {
        parameters = random.pick(&[json!(true), json!("x"), json!({})]);
    }
    let name = random.pick(&[json!(format!("f{index}")), json!(format!("ns.é\"{index}"))]);
    serde_json::from_value(json!({
        "type": "function", "function": {"name": name, "parameters": parameters}
    }))
    .unwrap()
}

/// The lengths worked out without building a tag are never less than the built tag's, and the
/// same unless a definition takes a name the tag's own definitions would (`symphony_*`), whose
/// references then count long. Each of the three row layouts, and a table with no thought.
#[test]
fn tag_lengths_are_worked_out_without_building_the_tags() {
    let check = |tags: &AddressedTags, tools: &[Tool], schema: &Value, exact: bool| {
        let cases = [
            (tags.calls(tools), tags.calls_len(tools)),
            (tags.single_call(tools), tags.single_call_len(tools)),
            (
                tags.response_format(schema, tools),
                tags.response_format_len(schema, tools),
            ),
            (
                tags.response_format(schema, &[]),
                tags.response_format_len(schema, &[]),
            ),
        ];
        for (kind, (tag, worked_out)) in cases.into_iter().enumerate() {
            let built = tag.payload().to_string().len();
            let case = || format!("kind {kind}: {}, {schema}", json!(tools));
            assert!(worked_out >= built, "{worked_out} < {built}: {}", case());
            if exact {
                assert_eq!(worked_out, built, "{}", case());
            }
        }
    };
    let fixture = fixture();
    let all: Vec<Tool> = serde_json::from_value(fixture["tools"].clone()).unwrap();
    let thoughtless = Format::new("thoughtless")
        .terminal("start", "<|start|>assistant")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("recipient", " to=")
        .terminal("end", "<|end|>")
        .terminal("return", "<|return|>")
        .terminal("call", "<|call|>")
        .state("header", Emits::Wrapper)
        .state("content", Emits::Content)
        .state("between", Emits::Wrapper)
        .state("calls", Emits::Arguments)
        .transition("header", "final", "content")
        .transition("header", "recipient", "calls")
        .transition("content", "return", "between")
        .transition("calls", "end", "between")
        .transition("calls", "call", "between")
        .transition("between", "start", "header")
        .calls(CallSyntax::Addressed(SPELLING))
        .opens_turn("<|start|>assistant")
        .marker_tokens(TOKENS)
        .stop_tokens(STOPS);
    for format in [one_outside(), split_header(), channels(), thoughtless] {
        let tags = format.addressed_tags().unwrap();
        for case in fixture["tags"].as_array().unwrap() {
            check(
                &tags,
                &tools(&fixture, &case["tools"]),
                &case["schema"],
                true,
            );
        }
        check(&tags, &all, &json!({"type": "object"}), true);
    }

    let tags = tags();
    let mut random = Random(2026);
    let mut exact = 0;
    for _ in 0..1000 {
        let tools: Vec<Tool> = (0..1 + random.below(3))
            .map(|index| random_tool(&mut random, index))
            .collect();
        let schema = random_schema(&mut random, 0);
        let taken = json!(tools).to_string().contains("\"symphony_");
        check(&tags, &tools, &schema, !taken);
        exact += usize::from(!taken);
    }
    assert!(exact > 500, "{exact}");
}

/// A tool whose `n` optional parameters each refer to one object of `m` integer fields: each JSON
/// value carries the object.
fn amplifying_tool(n: usize, m: usize) -> Tool {
    let properties: Map<String, Value> = (0..n)
        .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/X"})))
        .collect();
    let fields: Map<String, Value> = (0..m)
        .map(|j| (format!("f{j}"), json!({"type": "integer"})))
        .collect();
    tool(json!({"type": "object", "properties": properties,
                "$defs": {"X": {"type": "object", "properties": fields}}}))
}

/// The best of three runs of `len`, and how long it took.
fn timed(len: &dyn Fn() -> usize) -> (Duration, usize) {
    (0..3)
        .map(|_| {
            let start = Instant::now();
            let len = len();
            (start.elapsed(), len)
        })
        .min()
        .unwrap()
}

/// Amplifying tools: their tags hold the object at most 1 + ⌈log₂ n⌉ times per parameter, under a
/// tenth of the bytes of a body that held each optional value once for each parameter after it
/// (about n²/2 copies); each JSON value still carries the object, as a structural tag cannot refer
/// to one written elsewhere, so the larger ones stay over the limit. Their lengths are worked out
/// quickly however large they are.
#[test]
fn a_tag_too_large_to_build_is_measured_without_building_it() {
    let tags = tags();
    // The single-call tags of such a body, in bytes.
    for (n, m, before) in [
        (100, 20, 3_636_822),
        (100, 200, 26_866_822),
        (100, 500, 66_256_822),
        (100, 2000, 268_256_822),
        (200, 1000, 525_027_972),
    ] {
        let tools = [amplifying_tool(n, m)];
        let schema = json!({"type": "object"});
        let lens: [&dyn Fn() -> usize; 3] = [
            &|| tags.single_call_len(&tools),
            &|| tags.calls_len(&tools),
            &|| tags.response_format_len(&schema, &tools),
        ];
        let [single, forced, response_format] = lens.map(|len| {
            let (elapsed, len) = timed(len);
            assert!(
                elapsed < Duration::from_millis(50),
                "n={n} m={m}: {elapsed:?}"
            );
            len
        });
        assert!(single < before / 10, "n={n} m={m}: {single}");
        assert!(single < forced && forced < response_format, "n={n} m={m}");
        if m == 20 {
            let built = tags.single_call(&tools).payload().to_string();
            assert_eq!(single, built.len(), "n={n} m={m}");
        }
        assert_eq!(
            forced > AddressedTags::MAX_TAG_BYTES,
            m >= 200,
            "n={n} m={m}: {forced}"
        );
    }
}

/// Definitions named as the tags' own would be (`symphony_string`, then with `_` after it, ...)
/// push those apart with longer names; the length works them out once per tool, not again for each
/// value that carries the tool's definitions, and counts a reference to a long name without
/// writing it, so it stays quick.
#[test]
fn definitions_named_as_the_tags_own_are_measured_quickly() {
    let tags = tags();
    let colliding = |k: usize| -> Map<String, Value> {
        (0..k)
            .map(|i| (format!("symphony_string{}", "_".repeat(i)), json!(0)))
            .collect()
    };
    let tool_of = |k: usize, n: usize, value: &Value| {
        let mut defs = colliding(k);
        defs.insert("a".to_owned(), json!({"type": "integer"}));
        let properties: Map<String, Value> =
            (0..n).map(|i| (format!("p{i}"), value.clone())).collect();
        tool(json!({"type": "object", "$defs": defs, "properties": properties}))
    };
    let refers = json!({"$ref": "#/$defs/a"});
    let strings = json!({"type": "object", "properties": {
        "s": {"type": "string"}, "t": {"type": "string"}, "r": {"$ref": "#/$defs/a"}}});
    let response_format = |k: usize, m: usize| {
        let properties: Map<String, Value> = (0..m)
            .map(|j| (format!("s{j}"), json!({"type": "string"})))
            .collect();
        json!({"type": "object", "$defs": colliding(k), "properties": properties})
    };
    // Exact where the tags are small enough to build.
    for value in [&refers, &strings] {
        let tools = [tool_of(40, 30, value)];
        let schema = response_format(40, 30);
        assert_eq!(
            tags.calls_len(&tools),
            tags.calls(&tools).payload().to_string().len()
        );
        assert_eq!(
            tags.response_format_len(&schema, &[]),
            tags.response_format(&schema, &[])
                .payload()
                .to_string()
                .len()
        );
    }
    let quick = |len: &dyn Fn() -> usize| {
        let (elapsed, len) = timed(len);
        assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
        assert!(len > AddressedTags::MAX_TAG_BYTES, "{len}");
    };
    let tools = [tool_of(3000, 3000, &refers)];
    quick(&|| tags.calls_len(&tools));
    let tools = [tool_of(3000, 200, &strings)];
    quick(&|| tags.calls_len(&tools));
    let schema = response_format(3000, 20_000);
    quick(&|| tags.response_format_len(&schema, &[]));
}

/// A tool may declare up to `MAX_TOOL_PARAMETERS` parameters; one more is refused by its name and
/// the limit. Only its own parameters count, not the fields of a parameter's value, nor a schema
/// that is no object.
#[test]
fn a_tool_of_too_many_parameters_is_refused() {
    let max = AddressedTags::MAX_TOOL_PARAMETERS;
    assert_eq!(max, 1024);
    let parameters = |n: usize| {
        let properties: Map<String, Value> = (0..n)
            .map(|i| (format!("p{i}"), json!({"type": "integer"})))
            .collect();
        json!({"type": "object", "properties": properties})
    };
    let named = |name: &str, parameters: Value| -> Tool {
        serde_json::from_value(json!({
            "type": "function", "function": {"name": name, "parameters": parameters}
        }))
        .unwrap()
    };
    let nested = json!({"type": "object", "properties": {"v": parameters(4 * max)}});
    let fits = [
        tool(parameters(max)),
        named("nested", nested),
        named("none", json!({"type": "object"})),
        named("odd", json!({"type": "object", "properties": [1, 2]})),
    ];
    assert_eq!(AddressedTags::check_parameter_counts(&fits), Ok(()));
    let mut tools = fits.to_vec();
    tools.push(named("wide", parameters(max + 1)));
    let refused = AddressedTags::check_parameter_counts(&tools).unwrap_err();
    assert!(refused.contains("'wide'"), "{refused}");
    assert!(refused.contains("1025 parameters"), "{refused}");
    assert!(refused.contains("limit of 1024"), "{refused}");
}

/// The tools of one tag may declare up to `MAX_TOTAL_PARAMETERS` together; one more is refused by
/// the count and the limit, even when no tool is over its own limit. A tool's limit is checked
/// first.
#[test]
fn tools_of_too_many_parameters_together_are_refused() {
    let max = AddressedTags::MAX_TOTAL_PARAMETERS;
    assert_eq!(max, 4096);
    // Tools of `n` parameters together, each of at most 1,024.
    let tools = |n: usize| -> Vec<Tool> {
        let per_tool = AddressedTags::MAX_TOOL_PARAMETERS;
        (0..n.div_ceil(per_tool))
            .map(|t| {
                let count = per_tool.min(n - t * per_tool);
                let properties: Map<String, Value> = (0..count)
                    .map(|i| (format!("t{t}p{i}"), json!({"type": "integer"})))
                    .collect();
                serde_json::from_value(json!({"type": "function", "function": {
                    "name": format!("t{t}"),
                    "parameters": {"type": "object", "properties": properties}}}))
                .unwrap()
            })
            .collect()
    };
    let fits = tools(max);
    assert_eq!(fits.len(), 4);
    assert_eq!(AddressedTags::check_parameter_counts(&fits), Ok(()));
    let over = tools(max + 1);
    assert_eq!(over.len(), 5);
    let refused = AddressedTags::check_parameter_counts(&over).unwrap_err();
    assert!(refused.contains("4097 parameters together"), "{refused}");
    assert!(
        refused.contains("limit of 4096 parameters per request"),
        "{refused}"
    );
    let mut wide = over;
    wide.push(tool(json!({"type": "object", "properties": (0..1025)
        .map(|i| (format!("w{i}"), json!({})))
        .collect::<Map<String, Value>>()})));
    let refused = AddressedTags::check_parameter_counts(&wide).unwrap_err();
    assert!(refused.contains("tool 'f' declares 1025"), "{refused}");
}

#[test]
fn every_fixture_render_is_named_once() {
    let fixture = fixture();
    let names: BTreeSet<&str> = fixture["renders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|render| render["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), fixture["renders"].as_array().unwrap().len());
    for case in fixture["tags"].as_array().unwrap() {
        for name in ["accepts", "rejects"]
            .iter()
            .flat_map(|key| case[*key].as_array().unwrap())
        {
            assert!(
                names.contains(name.as_str().unwrap()),
                "{name} is not a render"
            );
        }
    }
}
