//! Templates compiled into symphony tables, over transformers' recordings.
//!
//! - The `transformers serve` templates compile into tables that read every recorded output, in
//!   every chunking, event for event as symphony's own tables for the family do; so do the two
//!   addressed-call shapes in `fixtures/symphony/`, against the table written by hand for them,
//!   on their outputs, on every cut of them and in seeded random chunkings.
//! - What a compiled table reads gives the message transformers parses, except in the sessions
//!   [`DIFFERS`] lists, each for a reason the compiler's documentation gives.
//! - Every recorded template either compiles or is refused for the reason `to_symphony.tsv`
//!   pins.
//!
//! `fixtures/symphony/` is recorded from `fixtures/symphony/cases.jsonl` with
//! `generate_hf_fixtures.py --extra-cases cases.jsonl --out DIR` (transformers 5.17.0).

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture should panic"
)]

mod common;

use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::Write as _,
    path::{Path, PathBuf},
};

use common::{chunk_lengths, fixtures_dir, read_json, read_jsonl, split_chars, untag};
use openai_protocol::common::Tool;
use serde_json::Value;
use smg_response_template::{load_response_template, to_symphony::compile};
use symphony::{
    adapt,
    tagged::{addressed, Placement, Spelling},
    CallSyntax, Declared, Emits, Engine, EngineFinish, Event, Events, Format, Input, Parser,
};

const QWEN_STOPS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];
const ADDRESSED_STOPS: [&str; 2] = ["<|done|>", "<|invoke|>"];
const FEEDS: [&str; 4] = ["whole", "chars", "step:3", "step:7"];

/// Sessions whose message differs from transformers', and why.
const DIFFERS: [(&str, &str); 11] = [
    (
        "serve/qwen2#eot",
        "a close of the field with no open that is no stop text stays content",
    ),
    (
        "serve/qwen2#tail-0",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen2#tail-0-no-tools",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen2#tail-2",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen2#tail-2-no-tools",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen3_5#tail-0",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen3_5#tail-0-no-tools",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen3_5#tail-2",
        "the prompt's text is not the output's",
    ),
    (
        "serve/qwen3_5#tail-2-no-tools",
        "the prompt's text is not the output's",
    ),
    (
        "extra/addressed-lookahead#repeated-key",
        "a key keeps its first value",
    ),
    (
        "extra/addressed#repeated-key",
        "a key keeps its first value",
    ),
];

fn addressed_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/symphony")
}

fn template(dir: &Path, id: &str) -> Value {
    read_jsonl(dir, "templates")
        .into_iter()
        .find(|row| row["id"] == id)
        .map(|row| untag(&row["template"]))
        .unwrap_or_else(|| panic!("no template {id}"))
}

fn compiled(dir: &Path, id: &str, stops: &[&str]) -> Format {
    let spec = template(dir, id);
    let loaded = load_response_template(&spec).unwrap();
    compile(&loaded, stops).unwrap_or_else(|why| panic!("{id}: {why}"))
}

/// The recorded sessions of a template, with transformers' message for each.
fn sessions(dir: &Path, id: &str) -> Vec<Value> {
    ["sessions", "random"]
        .iter()
        .flat_map(|name| read_jsonl(dir, name))
        .filter(|row| row["template"] == id)
        .collect()
}

fn declared(dir: &Path, session: &Value) -> Declared {
    let Some(set) = session["tools"].as_str() else {
        return Declared::default();
    };
    // A set may hold a bare function, which transformers reads and the API does not.
    let tools: Vec<Tool> = untag(&read_json(dir, "tools.json")[set])
        .as_array()
        .unwrap()
        .iter()
        .filter(|tool| tool.get("type").is_some())
        .map(|tool| serde_json::from_value(tool.clone()).unwrap())
        .collect();
    Declared::of(&tools)
}

/// The output as the stop decoder hands it over: up to the first stop text, which it hides.
fn output<'a>(text: &'a str, stops: &[&str]) -> (&'a str, EngineFinish) {
    match stops.iter().filter_map(|stop| text.find(stop)).min() {
        Some(end) => (&text[..end], EngineFinish::Stop),
        None => (text, EngineFinish::Length),
    }
}

fn run(
    format: Format,
    declared: Declared,
    prompt: &str,
    pieces: &[&str],
    end: EngineFinish,
) -> Vec<Event> {
    let mut parser = Engine::new(format, declared);
    let mut out = Events::new();
    if !prompt.is_empty() {
        let prompt = Input::Prompt {
            token_ids: &[],
            text: prompt,
        };
        parser.feed(prompt, &mut out).unwrap();
    }
    for piece in pieces {
        let delta = Input::Delta {
            token_ids: &[],
            text: piece,
            spans: &[],
        };
        parser.feed(delta, &mut out).unwrap();
    }
    parser.feed(Input::End { finish: end }, &mut out).unwrap();
    out.drain()
}

/// Every session in every chunking reads the same events through both tables, the family's
/// reading as transformers does, as a compiled table does ([`Format::reads_as_transformers`]).
fn same_events(dir: &Path, id: &str, stops: &[&str], family: &dyn Fn() -> Format) {
    let table = compiled(dir, id, stops);
    let sessions = sessions(dir, id);
    assert!(sessions.len() >= 10, "{id}: {} sessions", sessions.len());
    for session in &sessions {
        let (text, end) = output(session["text"].as_str().unwrap(), stops);
        let prompt = session["prefix"].as_str().unwrap();
        for feed in FEEDS {
            let pieces = split_chars(text, &chunk_lengths(text, feed));
            let ours = run(
                table.clone(),
                declared(dir, session),
                prompt,
                &pieces,
                end.clone(),
            );
            let theirs = run(
                family().reads_as_transformers(),
                declared(dir, session),
                prompt,
                &pieces,
                end.clone(),
            );
            assert_eq!(ours, theirs, "{} ({feed})", session["id"]);
        }
    }
}

#[test]
fn qwen2_reads_as_qwen3_json() {
    same_events(&fixtures_dir(), "serve/qwen2", &QWEN_STOPS, &|| {
        symphony::qwen3(CallSyntax::Json)
    });
}

#[test]
fn qwen3_5_reads_as_qwen3_tagged() {
    same_events(&fixtures_dir(), "serve/qwen3_5", &QWEN_STOPS, &|| {
        symphony::qwen3(CallSyntax::Tagged(Spelling::Json, Placement::OwnLine))
    });
}

/// The table written by hand for the addressed shapes of `fixtures/symphony/`.
fn addressed_by_hand(next_key_closes: bool) -> Format {
    let spelling = addressed::Spelling {
        header_close: Cow::Borrowed("<|via|>notes <|as|>xml<|body|>"),
        key_open: Cow::Borrowed("<arg name=\""),
        key_close: Cow::Borrowed("\">"),
        value_close: Cow::Borrowed("</arg>"),
        next_key_closes,
    };
    Format::new("addressed")
        .reads_as_transformers()
        .terminal("start", "<|head|>assistant")
        .terminal("notes", "<|via|>notes<|body|>")
        .terminal("reply", "<|via|>reply<|body|>")
        .terminal("recipient", " to=")
        .terminal("next", "<|stop|><|head|>assistant")
        .terminal("stop", "<|stop|>")
        .terminal("done", "<|done|>")
        .terminal("invoke", "<|invoke|>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("outside", "notes", "reasoning")
        .transition("outside", "reply", "content")
        .transition("outside", "recipient", "calls")
        .transition("outside", "start", "outside")
        .transition("reasoning", "next", "outside")
        .transition("content", "next", "outside")
        .transition("content", "stop", "outside")
        .transition("content", "done", "outside")
        .transition("content", "invoke", "outside")
        .transition("calls", "next", "outside")
        .transition("calls", "invoke", "outside")
        .calls(CallSyntax::Addressed(spelling))
        .closes_at_stop("calls")
        .marker_only("outside")
        .opens_turn("<|head|>assistant")
}

#[test]
fn addressed_shapes_read_as_their_table() {
    let dir = addressed_dir();
    for (id, next_key_closes) in [
        ("extra/addressed-lookahead", true),
        ("extra/addressed", false),
    ] {
        same_events(&dir, id, &ADDRESSED_STOPS, &|| {
            addressed_by_hand(next_key_closes)
        });
    }
}

/// A small seeded generator, so that the cuts and chunkings are the same on every run.
struct Seeded(u64);

impl Seeded {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(self.0 >> 33).unwrap_or_default() % n.max(1)
    }
}

#[test]
fn addressed_shapes_read_as_their_table_when_cut_and_chunked_at_random() {
    let dir = addressed_dir();
    let mut seeded = Seeded(7);
    for (id, next_key_closes) in [
        ("extra/addressed-lookahead", true),
        ("extra/addressed", false),
    ] {
        let table = compiled(&dir, id, &ADDRESSED_STOPS);
        for session in sessions(&dir, id) {
            let (text, _) = output(session["text"].as_str().unwrap(), &ADDRESSED_STOPS);
            let chars: Vec<usize> = text.char_indices().map(|(at, _)| at).collect();
            for cut in 0..=chars.len() {
                let cut_at = chars.get(cut).copied().unwrap_or(text.len());
                let text = &text[..cut_at];
                let mut pieces = Vec::new();
                let mut at = 0;
                while at < text.len() {
                    let mut end = (at + 1 + seeded.below(9)).min(text.len());
                    while !text.is_char_boundary(end) {
                        end += 1;
                    }
                    pieces.push(&text[at..end]);
                    at = end;
                }
                for end in [EngineFinish::Stop, EngineFinish::Length] {
                    let ours = run(
                        table.clone(),
                        declared(&dir, &session),
                        "",
                        &pieces,
                        end.clone(),
                    );
                    let theirs = run(
                        addressed_by_hand(next_key_closes),
                        declared(&dir, &session),
                        "",
                        &pieces,
                        end,
                    );
                    assert_eq!(ours, theirs, "{} cut at {cut_at}", session["id"]);
                }
            }
        }
    }
}

/// A message as both readers give it: reasoning, content, and each call's name and arguments.
#[derive(Debug, PartialEq)]
struct Message {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
}

fn ours(events: &[Event]) -> Message {
    let message = adapt::chat::message(0, events).message;
    Message {
        reasoning: message.reasoning_content.unwrap_or_default(),
        content: message.content.unwrap_or_default(),
        calls: message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments = call.function.arguments.unwrap_or_default();
                let arguments =
                    serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
                (call.function.name, arguments)
            })
            .collect(),
    }
}

fn theirs(message: &Value) -> Message {
    let text = |key: &str| {
        message
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    Message {
        reasoning: text("thinking") + &text("reasoning_content"),
        content: text("content"),
        calls: message
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|call| {
                let function = &call["function"];
                (
                    function["name"].as_str().unwrap_or_default().to_owned(),
                    function["arguments"].clone(),
                )
            })
            .collect(),
    }
}

/// Sessions whose messages differ, by id, with both readings. The text fields' edges are
/// stripped, as transformers strips them and a stream cannot; a session transformers raised on
/// is left out.
fn differences(dir: &Path, id: &str, stops: &[&str]) -> BTreeMap<String, String> {
    let table = compiled(dir, id, stops);
    let strip = |mut message: Message| {
        message.reasoning = message.reasoning.trim().to_owned();
        message.content = message.content.trim().to_owned();
        message
    };
    let mut out = BTreeMap::new();
    for session in sessions(dir, id) {
        let final_ = &session["unary"]["final"];
        if final_[0] != "final" {
            continue;
        }
        let (text, end) = output(session["text"].as_str().unwrap(), stops);
        let prompt = session["prefix"].as_str().unwrap();
        let events = run(table.clone(), declared(dir, &session), prompt, &[text], end);
        let (ours, theirs) = (strip(ours(&events)), strip(theirs(&untag(&final_[1]))));
        if ours != theirs {
            out.insert(
                session["id"].as_str().unwrap().to_owned(),
                format!("ours   {ours:?}\ntheirs {theirs:?}"),
            );
        }
    }
    out
}

#[test]
fn compiled_tables_read_the_messages_transformers_parses() {
    let hf = fixtures_dir();
    let addressed = addressed_dir();
    let cases: [(&Path, &str, &[&str]); 4] = [
        (&hf, "serve/qwen2", &QWEN_STOPS),
        (&hf, "serve/qwen3_5", &QWEN_STOPS),
        (&addressed, "extra/addressed-lookahead", &ADDRESSED_STOPS),
        (&addressed, "extra/addressed", &ADDRESSED_STOPS),
    ];
    let mut unexpected = String::new();
    let mut seen = Vec::new();
    for (dir, id, stops) in cases {
        for (session, diff) in differences(dir, id, stops) {
            if !DIFFERS.iter().any(|(listed, _)| *listed == session) {
                writeln!(unexpected, "{session}\n{diff}\n").unwrap();
            }
            seen.push(session);
        }
    }
    assert!(unexpected.is_empty(), "{unexpected}");
    let mut listed: Vec<String> = DIFFERS.iter().map(|(id, _)| (*id).to_owned()).collect();
    seen.sort();
    listed.sort();
    assert_eq!(seen, listed, "a listed difference is gone");
}

/// A call region whose opener leads to no call shows the opener with the text after it, as
/// transformers keeps an open pattern that did not match in the content: at a stop inside the
/// header, `<tool_call>` and ` to=` stay with the cut name. transformers gives the Qwen 3.5 output
/// `<tool_call>\n<function=get_wea` as content; it drops the addressed one, which falls outside
/// every field, where the table keeps outside text as content. A cut drops the header as the
/// template's, the opener with it, and a call that starts drops the opener as before. Where the
/// call syntax shows a header as text at any end (a header closed with no name in it; Qwen's
/// function tag, which its assembler reports at a cut too), the opener goes with it.
#[test]
fn an_opener_whose_call_never_starts_stays_with_its_text() {
    let hf = fixtures_dir();
    let addressed = addressed_dir();
    // (template, stops, output, content at a stop, whether a cut drops it all)
    let cases = [
        (
            &hf,
            "serve/qwen3_5",
            &QWEN_STOPS,
            "<think>\nx\n</think>\n\n<tool_call>\n<function=get_wea",
            "<tool_call>\n<function=get_wea",
            false,
        ),
        (
            &addressed,
            "extra/addressed",
            &ADDRESSED_STOPS,
            "<|via|>notes<|body|>T<|stop|><|head|>assistant to=get_wea",
            "to=get_wea",
            true,
        ),
        (
            &addressed,
            "extra/addressed",
            &ADDRESSED_STOPS,
            "<|via|>notes<|body|>T<|stop|><|head|>assistant to=a b<|via|>notes <|as|>xml<|body|>",
            "to=a b<|via|>notes <|as|>xml<|body|>",
            false,
        ),
    ];
    for (dir, id, stops, text, content, cut_drops) in cases {
        let table = compiled(dir, id, stops);
        let pieces: Vec<&str> = text
            .char_indices()
            .map(|(at, c)| &text[at..at + c.len_utf8()])
            .collect();
        for feed in [vec![text], pieces] {
            let read = |end| ours(&run(table.clone(), Declared::default(), "", &feed, end));
            let stopped = read(EngineFinish::Stop);
            assert_eq!(stopped.content.trim(), content, "{id} {text:?}");
            assert!(stopped.calls.is_empty());
            // A header the model closed wrong is its text however the stream ends.
            let cut = read(EngineFinish::Length).content;
            let expected = if cut_drops { "" } else { content };
            assert_eq!(cut.trim(), expected, "{id} {text:?} cut");
        }
    }
    // A call that starts keeps its opener out of the content.
    let table = compiled(&addressed, "extra/addressed", &ADDRESSED_STOPS);
    let text = "<|via|>notes<|body|>T<|stop|><|head|>assistant to=f<|via|>notes <|as|>xml<|body|>";
    let read = ours(&run(
        table,
        Declared::default(),
        "",
        &[text],
        EngineFinish::Stop,
    ));
    assert_eq!(read.content, "");
    assert_eq!(read.calls.len(), 1);
}

/// Every recorded template that loads, compiled: `table`, or the reason it is refused.
#[test]
fn templates_compile_or_are_refused_as_pinned() {
    let mut seen = String::new();
    for dir in [fixtures_dir(), addressed_dir()] {
        for row in read_jsonl(&dir, "templates") {
            let id = row["id"].as_str().unwrap();
            let Ok(loaded) = load_response_template(&untag(&row["template"])) else {
                continue;
            };
            let outcome = match compile(&loaded, &QWEN_STOPS) {
                Ok(_) => "table".to_owned(),
                Err(why) => why.to_string(),
            };
            writeln!(seen, "{id}\t{outcome}").unwrap();
        }
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/to_symphony.tsv");
    if std::env::var_os("TO_SYMPHONY_WRITE").is_some() {
        std::fs::write(&path, &seen).unwrap();
    }
    let pinned = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        seen, pinned,
        "compile outcomes changed (TO_SYMPHONY_WRITE=1 rewrites the list)"
    );
}

#[test]
fn the_families_without_a_table_are_refused_for_their_construct() {
    let dir = fixtures_dir();
    for (id, why) in [
        ("serve/gemma4", "response_template: several start anchors"),
        (
            "shapes/literal-anchors-json-tools",
            "response_template: several start anchors",
        ),
        (
            "shapes/xml-invoke-word-boundary",
            "Field 'tool_calls': an open with an anchor or a word boundary",
        ),
    ] {
        let loaded = load_response_template(&template(&dir, id)).unwrap();
        let refused = compile(&loaded, &QWEN_STOPS).unwrap_err();
        assert_eq!(refused.to_string(), why, "{id}");
    }
}
