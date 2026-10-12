//! The values of a symphony table's calls, typed as transformers types them: a call's value reads
//! as transformers' `_sub_parse`, `_coerce` and `_coerce_tool_calls` read it, over every row
//! `fixtures/compiled/values.jsonl` records with transformers' own functions.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "a test: a bad fixture should panic"
)]

mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use common::{read_json, read_jsonl, untag};
use openai_protocol::common::Tool;
use serde_json::Value;
use smg_response_template::to_symphony::TRANSFORMERS;
use symphony::{
    shape::{Dialect, Duplicates, Schema, ValueParser},
    Declared, ValueRead,
};

/// The public templates, transformers' sessions for them, and its typing of values.
fn compiled_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compiled")
}

/// A value parser as the fixture writes it (`value_parser` of `content_args`), as a table's shape
/// holds it.
fn value_parser(spec: &Value) -> ValueParser {
    if spec.is_null() {
        return ValueParser::Raw;
    }
    let args = &spec["args"];
    let flag = |key: &str, default: bool| args.get(key).and_then(Value::as_bool).unwrap_or(default);
    match spec["name"].as_str().unwrap_or("text") {
        "text" => ValueParser::Text {
            strip: flag("strip", true),
        },
        "json" => ValueParser::Json {
            dialect: Dialect {
                bare_keys: flag("unquoted_keys", false),
                string_delims: args["string_delims"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|pair| {
                        (
                            pair[0].as_str().unwrap().into(),
                            pair[1].as_str().unwrap().into(),
                        )
                    })
                    .collect(),
            },
            allow_non_json: flag("allow_non_json", false),
            strip: flag("strip", true),
        },
        "int" => ValueParser::Int,
        "float" => ValueParser::Float,
        "bool" => ValueParser::Bool {
            strip: flag("strip", true),
        },
        other => panic!("no value parser {other}"),
    }
}

/// What the reader writes, read back as JSON.
fn json_of(written: &str) -> Value {
    serde_json::from_str(written).unwrap_or_else(|e| panic!("{written:?} is no JSON: {e}"))
}

/// The value transformers recorded, or, where it raised, the text: the reader writes the text
/// where transformers raises.
fn recorded(row: &Value) -> Value {
    match row.get("output") {
        Some(output) => untag(output),
        None => row["text"].clone(),
    }
}

/// A tagged value reads as `_sub_parse` and then `_coerce` with `_schema_types` read it, a string
/// member of JSON arguments is cast the same way when it closes, and held until then only when a
/// cast could change it; over every schema and text `fixtures/compiled/values.jsonl` records,
/// recorded with transformers' own functions.
#[test]
fn values_are_typed_as_transformers_types_them() {
    let dir = compiled_dir();
    let manifest = read_json(&dir, "manifest.json");
    let schemas = untag(&manifest["value_schemas"]);
    let tools = untag(&manifest["value_tools"]);
    let root = &tools[0]["function"]["parameters"];
    let api: Vec<Tool> = serde_json::from_value(tools.clone()).unwrap();
    let declared = Declared::of(&api);
    let mut casts: BTreeMap<String, bool> = BTreeMap::new();
    let mut counted = BTreeMap::new();
    for row in read_jsonl(&dir, "values") {
        let kind = row["kind"].as_str().unwrap();
        *counted.entry(kind.to_owned()).or_insert(0) += 1;
        match kind {
            "coerce" => {
                let key = row["schema"].as_str().unwrap();
                let text = row["text"].as_str().unwrap();
                let schema = Schema {
                    root,
                    property: &schemas[key],
                };
                let expected = recorded(&row);
                let read = TRANSFORMERS.read(&ValueParser::Raw, text, Some(schema));
                assert_eq!(json_of(&read), expected, "{key} {text:?}");
                let kept = expected == Value::String(text.to_owned());
                let member =
                    TRANSFORMERS.member(&Value::String(text.to_owned()).to_string(), Some(schema));
                assert_eq!(
                    member.as_deref().map(json_of),
                    (!kept).then_some(expected),
                    "{key} {text:?} as a member"
                );
                *casts.entry(key.to_owned()).or_default() |= !kept && row.get("error").is_none();
            }
            "sub_parse" => {
                let parser = value_parser(&row["value_parser"]);
                let text = row["text"].as_str().unwrap();
                let read = TRANSFORMERS.read(&parser, text, None);
                assert_eq!(json_of(&read), recorded(&row), "{parser:?} {text:?}");
            }
            "coerce_tool_calls" => {
                let call = untag(&row["value"]);
                let name = call["function"]["name"].as_str().unwrap();
                let output = untag(&row["output"]);
                let expected = &output["function"]["arguments"];
                for (key, value) in call["function"]["arguments"].as_object().unwrap() {
                    let schema = declared
                        .parameter(name, key)
                        .map(|(root, property)| Schema { root, property });
                    // A tagged value whose text reads as this JSON is typed as the argument is.
                    let read = TRANSFORMERS.read(&ValueParser::json(), &value.to_string(), schema);
                    assert_eq!(json_of(&read), expected[key], "{call} {key} as a value");
                    // A string member of JSON arguments is cast when it closes.
                    if value.is_string() {
                        let member = TRANSFORMERS
                            .member(&value.to_string(), schema)
                            .map_or_else(|| value.clone(), |typed| json_of(&typed));
                        assert_eq!(member, expected[key], "{call} {key} as a member");
                    }
                }
            }
            other => panic!("no kind {other}"),
        }
    }
    // A member waits until it is whole only where a cast could change it.
    for (key, cast) in &casts {
        let schema = Schema {
            root,
            property: &schemas[key.as_str()],
        };
        assert_eq!(
            TRANSFORMERS.streams(&ValueParser::json(), Some(schema)),
            !cast,
            "{key}"
        );
    }
    assert!(TRANSFORMERS.streams(&ValueParser::json(), None));
    assert_eq!(TRANSFORMERS.duplicates(), Duplicates::Last);
    assert_eq!(
        counted,
        BTreeMap::from([
            ("coerce".to_owned(), 810),
            ("coerce_tool_calls".to_owned(), 9),
            ("sub_parse".to_owned(), 588),
        ])
    );
}
