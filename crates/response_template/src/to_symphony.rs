//! A response template compiled into a symphony table.
//!
//! symphony reads a model's output with a table of literal markers (a [`Format`]) and never runs
//! a regex while it streams. Many response templates describe exactly such a table: each field
//! opens and closes at literal markers, and a tool call is a shape one of symphony's call
//! assemblers reads. [`compile`] recognizes those templates, field by field and pattern by
//! pattern, through the parse tree of each pattern, and builds the table. A template, or a part
//! of one, that is not such a shape is refused with the construct named ([`Unsupported`]), and
//! nothing changes for that model.
//!
//! What a template becomes:
//!
//! - The start anchor, one literal, is the table's turn opener. When every field has an open, the
//!   text outside the fields is a state of its own that holds only markers, and the anchor is also
//!   a marker there that moves nothing.
//! - A field with no open, which must be `content` as text, is the state outside the other
//!   fields. Its close ends the message, which the stream's stop does, so it adds no row.
//! - `thinking` (or `reasoning_content`) as text is a reasoning state, `content` as text a content
//!   state, and `tool_calls` the arguments state; each open of a field moves into its state from
//!   outside, and each close moves back out. A marker that is a close followed by other markers
//!   (`<|end|>` then the start anchor) moves where those markers move in turn.
//! - `tool_calls` is read by one of three assemblers: a JSON object between literal markers
//!   (`CallSyntax::Json`); the Qwen 3.5 tags, in exactly the spelling that assembler reads
//!   (`CallSyntax::Tagged`); or a call addressed to the tool, the name after a literal recipient
//!   marker and before a literal header close, with tagged parameters
//!   (`CallSyntax::Addressed`, its spelling taken from the template). A close of the tool calls
//!   that is a stop text closes the open call when the stream stops.
//!
//! Where the table reads differently from transformers, by design, since a stream cannot take
//! back what it already sent: text outside every field is content, not dropped; a second region
//! of a field that does not repeat is added to the first, not put in its place; the edges of a
//! text field are not stripped; whitespace a pattern allows before a marker (`\s*<tool_call>`) or
//! around a tagged value stays where the model wrote it, or is the assembler's to drop; a key
//! written twice in one call keeps its first value; with no tools declared, a value is read as
//! the JSON its text is; text the prompt wrote after the turn opener is the prompt's, not the
//! output's; a close of the field with no open that is no stop text stays content; and a call the
//! stream cut stays cut.

use std::{borrow::Cow, fmt};

use serde_json::{json, Value};
use symphony::{
    format::{CallSyntax, Emits, Format},
    tagged::{addressed, Placement, Spelling},
};

use crate::{
    content_parsers::{Content, ContentParser},
    pyre::{Atom, Body, Pattern},
    response_templates::{Delimiter, ResponseTemplate, ResponseTemplateField},
};

/// Why a template has no table: where in it (`response_template` or a field), and the construct.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{scope}: {construct}")]
pub struct Unsupported {
    pub scope: String,
    pub construct: String,
}

fn refuse<T>(scope: &str, construct: impl fmt::Display) -> Result<T, Unsupported> {
    Err(Unsupported {
        scope: scope.to_owned(),
        construct: construct.to_string(),
    })
}

const TEMPLATE: &str = "response_template";
const OUTSIDE: &str = "outside";

/// The transform of a call whose name the open captures and whose arguments are the content.
fn named_call() -> Value {
    json!({"type": "function", "function": {"name": "{name}", "arguments": "{content}"}})
}

/// The transform of a call that is one JSON object holding its name and arguments.
fn json_call() -> Value {
    json!({"type": "function", "function": "{content}"})
}

/// Compile `template` into a symphony table. `stops` are the texts of the tokenizer's EOS and stop
/// tokens: a close of the tool calls that is one of them closes the open call at the stream's
/// stop, since the stop decoder hides it.
pub fn compile(template: &ResponseTemplate, stops: &[&str]) -> Result<Format, Unsupported> {
    let template = &template.0;
    if let Some(key) = template.defaults.keys().find(|key| *key != "role") {
        return refuse(TEMPLATE, format_args!("a default for {key:?}"));
    }
    let opener = match literals(&template.start_anchor) {
        Ok((false, anchors)) if anchors.len() == 1 => anchors[0].clone(),
        Ok((false, _)) => return refuse(TEMPLATE, "several start anchors"),
        Ok((true, _)) => return refuse(TEMPLATE, "whitespace before the start anchor"),
        Err(what) => return refuse(TEMPLATE, format_args!("a start anchor with {what}")),
    };

    let mut table = Table::default();
    let outside_holds_content = match template.implicit.map(|i| &template.fields[i]) {
        None => false,
        Some(field) if field.name == "content" => {
            text_field(field)?;
            true
        }
        Some(field) => {
            return refuse(
                &scope(field),
                "a field with no open that is not the content",
            )
        }
    };
    table.state(OUTSIDE, Emits::Content);

    let mut kinds: Vec<&str> = Vec::new();
    for field in template.fields.iter().filter(|field| field.open.is_some()) {
        let scope = scope(field);
        let emits = match field.name.as_str() {
            "thinking" | "reasoning_content" => Emits::Reasoning,
            "content" => Emits::Content,
            "tool_calls" => Emits::Arguments,
            name => return refuse(&scope, format_args!("a field named {name:?}")),
        };
        let kind = match emits {
            Emits::Reasoning => "reasoning",
            Emits::Content => "content",
            _ => "tool calls",
        };
        if kinds.contains(&kind) {
            return refuse(&scope, format_args!("a second field of {kind}"));
        }
        kinds.push(kind);
        let state = table.state(&field.name, emits);
        let (opens, closes) = if emits == Emits::Arguments {
            let call = tool_calls(field)?;
            table.syntax = Some(call.syntax);
            if call.reenters {
                for open in &call.opens {
                    table.row(state, open, state);
                }
            }
            if call
                .closes
                .iter()
                .any(|close| stops.contains(&close.as_str()))
            {
                table.stop_closes = Some(state);
            }
            (call.opens, call.closes)
        } else {
            text_field(field)?;
            let open = delimiter(&scope, "open", field.open.as_ref())?;
            let close = delimiter(&scope, "close", field.close.as_ref())?;
            (open, close)
        };
        for open in &opens {
            if table.next(0, open).is_some() {
                return refuse(&scope, format_args!("an open {open:?} another field has"));
            }
            table.row(0, open, state);
        }
        for close in &closes {
            table.row(state, close, 0);
        }
    }
    if !outside_holds_content {
        table.row(0, &opener, 0);
    }
    table.derive_rows()?;

    let mut format = Format::new("response_template");
    for text in &table.terminals {
        format = format.terminal(text, text);
    }
    for (name, emits) in &table.states {
        format = format.state(name, *emits);
    }
    for &(from, on, to) in &table.rows {
        let (from, to) = (&table.states[from].0, &table.states[to].0);
        format = format.transition(from, &table.terminals[on], to);
    }
    if let Some(syntax) = table.syntax {
        format = format.calls(syntax);
    }
    if let Some(state) = table.stop_closes {
        format = format.closes_at_stop(&table.states[state].0);
    }
    if !outside_holds_content {
        format = format.marker_only(OUTSIDE);
    }
    format = format.opens_turn(&opener);
    match format.validate() {
        Ok(()) => Ok(format),
        Err(why) => refuse(TEMPLATE, why),
    }
}

/// `open` or `close` with its article.
fn marker(which: &str) -> &'static str {
    if which == "open" {
        "an open"
    } else {
        "a close"
    }
}

fn scope(field: &ResponseTemplateField) -> String {
    format!("Field '{}'", field.name)
}

/// The table as it is built: terminals by text, states, and rows `(from, terminal, to)`.
#[derive(Default)]
struct Table {
    terminals: Vec<String>,
    states: Vec<(String, Emits)>,
    rows: Vec<(usize, usize, usize)>,
    syntax: Option<CallSyntax>,
    stop_closes: Option<usize>,
}

impl Table {
    fn state(&mut self, name: &str, emits: Emits) -> usize {
        self.states.push((name.to_owned(), emits));
        self.states.len() - 1
    }

    fn terminal(&mut self, text: &str) -> usize {
        match self.terminals.iter().position(|t| t == text) {
            Some(index) => index,
            None => {
                self.terminals.push(text.to_owned());
                self.terminals.len() - 1
            }
        }
    }

    fn row(&mut self, from: usize, text: &str, to: usize) {
        let on = self.terminal(text);
        if self.next(from, text).is_none() {
            self.rows.push((from, on, to));
        }
    }

    fn next(&self, from: usize, text: &str) -> Option<usize> {
        self.rows
            .iter()
            .find(|&&(f, on, _)| f == from && self.terminals[on] == text)
            .map(|&(_, _, to)| to)
    }

    /// Where `text` takes the engine from `from` when it is a run of markers each of which has a
    /// row from where the one before it left, longest marker first.
    fn walk(&self, from: usize, text: &str) -> Option<usize> {
        if text.is_empty() {
            return Some(from);
        }
        self.rows
            .iter()
            .filter(|&&(f, on, _)| f == from && text.starts_with(self.terminals[on].as_str()))
            .max_by_key(|&&(_, on, _)| self.terminals[on].len())
            .and_then(|&(_, on, to)| self.walk(to, &text[self.terminals[on].len()..]))
    }

    /// A terminal with no row from a state, which begins with a marker that moves out of that
    /// state, moves where that marker and the ones after it move, as transformers reads the
    /// marker first. A state's marker inside a longer terminal anywhere else is refused: the
    /// table would read the longer terminal where transformers reads the marker.
    fn derive_rows(&mut self) -> Result<(), Unsupported> {
        let direct = self.rows.clone();
        for from in 0..self.states.len() {
            for on in 0..self.terminals.len() {
                if direct.iter().any(|&(f, o, _)| f == from && o == on) {
                    continue;
                }
                let text = &self.terminals[on];
                // The start anchor moves nothing; a marker holding it is not read through it.
                let moving = direct
                    .iter()
                    .filter(|&&(f, _, to)| f == from && (f, to) != (0, 0));
                for &(_, marker, _) in moving {
                    let marker = &self.terminals[marker];
                    if let Some(at) = text.find(marker.as_str()).filter(|&at| at > 0) {
                        return refuse(
                            TEMPLATE,
                            format_args!(
                                "the marker {text:?} holds {marker:?} at byte {at}, which moves \
                                 out of {:?}",
                                self.states[from].0
                            ),
                        );
                    }
                }
                if let Some(to) = self.walk(from, text) {
                    self.rows.push((from, on, to));
                }
            }
        }
        Ok(())
    }
}

/// A delimiter as its literal alternatives, and whether whitespace may come before them; or what
/// else it holds.
fn literals(pattern: &Pattern) -> Result<(bool, Vec<String>), String> {
    let mut atoms = pattern.shape()?;
    let spaces = atoms.first() == Some(&Atom::Spaces);
    if spaces {
        atoms.remove(0);
    }
    let texts = match atoms.as_slice() {
        [Atom::Lit(text)] => vec![text.clone()],
        [Atom::Alt(branches)] => branches
            .iter()
            .map(|branch| match branch.as_slice() {
                [Atom::Lit(text)] => Ok(text.clone()),
                _ => Err("an alternative that is not literal text".to_owned()),
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("more than literal text".into()),
    };
    Ok((spaces, texts))
}

/// A text field's open or close, as the literal markers it is.
fn delimiter(scope: &str, which: &str, d: Option<&Delimiter>) -> Result<Vec<String>, Unsupported> {
    let Some(d) = d else {
        return Ok(Vec::new());
    };
    match literals(&d.pattern) {
        Ok((_, texts)) => Ok(texts),
        Err(what) => refuse(scope, format_args!("{} with {what}", marker(which))),
    }
}

/// The checks every text field passes.
fn text_field(field: &ResponseTemplateField) -> Result<(), Unsupported> {
    let scope = scope(field);
    if field.content.parser != ContentParser::Text {
        return refuse(&scope, "content that is not text");
    }
    common(field)?;
    if field.repeats {
        return refuse(&scope, "a text field that repeats");
    }
    if field.transform.is_some() {
        return refuse(&scope, "a transform of a text field");
    }
    Ok(())
}

/// The checks every field passes.
fn common(field: &ResponseTemplateField) -> Result<(), Unsupported> {
    let scope = scope(field);
    if field.join.is_some() {
        return refuse(&scope, "join");
    }
    if field.transform_each {
        return refuse(&scope, "transform_each");
    }
    if !field.optional {
        return refuse(&scope, "optional: false");
    }
    Ok(())
}

/// The tool calls field, compiled.
struct Calls {
    syntax: CallSyntax,
    opens: Vec<String>,
    closes: Vec<String>,
    /// Whether an open inside a call starts the next one, as for the markers of a JSON or
    /// tagged call, which never hold one.
    reenters: bool,
}

fn tool_calls(field: &ResponseTemplateField) -> Result<Calls, Unsupported> {
    let scope = scope(field);
    common(field)?;
    if !field.repeats {
        return refuse(&scope, "tool calls that do not repeat");
    }
    let shape = |d: Option<&Delimiter>, which: &str| match d {
        None => refuse(&scope, format_args!("tool calls with no {which}")),
        Some(d) => d
            .pattern
            .shape()
            .or_else(|what| refuse(&scope, format_args!("{} with {what}", marker(which)))),
    };
    let open = shape(field.open.as_ref(), "open")?;
    let close = shape(field.close.as_ref(), "close")?;
    let transform = field.transform.as_ref();
    match field.content.parser {
        ContentParser::Json => {
            if transform != Some(&json_call()) {
                return refuse(&scope, "a transform of a JSON call other than its function");
            }
            let content = &field.content;
            if content.unquoted_keys || !content.string_delims.is_empty() || content.allow_non_json
            {
                return refuse(&scope, "a JSON dialect");
            }
            let open = literal_marker(&scope, "open", &open)?;
            let close = literal_marker(&scope, "close", &close)?;
            Ok(Calls {
                syntax: CallSyntax::Json,
                opens: open,
                closes: close,
                reenters: true,
            })
        }
        ContentParser::XmlInline => {
            if transform != Some(&named_call()) {
                return refuse(&scope, "a transform other than the name and the arguments");
            }
            let content = &field.content;
            if content.merge_duplicates {
                return refuse(&scope, "merge_duplicates");
            }
            let Some(tags) = &content.tag_pattern else {
                return refuse(&scope, "xml-inline with no tag_pattern");
            };
            let tags = tags
                .shape()
                .or_else(|what| refuse(&scope, format_args!("a tag_pattern with {what}")))?;
            if let Some(calls) = qwen_tags(&open, &close, &tags, content.value_parser.as_deref()) {
                return Ok(calls);
            }
            addressed(
                &scope,
                &open,
                &close,
                &tags,
                content.value_parser.as_deref(),
            )
        }
        _ => refuse(&scope, "tool calls whose content is not json or xml-inline"),
    }
}

/// An open or close of the tool calls that is literal markers, whitespace before them allowed.
fn literal_marker(scope: &str, which: &str, atoms: &[Atom]) -> Result<Vec<String>, Unsupported> {
    let atoms = atoms.strip_prefix(&[Atom::Spaces]).unwrap_or(atoms);
    match atoms {
        [Atom::Lit(text)] => Ok(vec![text.clone()]),
        [Atom::Alt(branches)] => branches
            .iter()
            .map(|branch| match branch.as_slice() {
                [Atom::Lit(text)] => Ok(text.clone()),
                _ => refuse(
                    scope,
                    format_args!("{} that is not literal text", marker(which)),
                ),
            })
            .collect(),
        _ => refuse(
            scope,
            format_args!("{} that is not literal text", marker(which)),
        ),
    }
}

fn lit(text: &str) -> Atom {
    Atom::Lit(text.to_owned())
}

fn capture(name: &str, body: Body) -> Atom {
    Atom::Capture {
        name: name.to_owned(),
        body,
    }
}

/// Whether a capture reads `[^...]+` of exactly these characters (in any order).
fn reads_none_of(atom: &Atom, name: &str, chars: &[char], space: bool) -> bool {
    match atom {
        Atom::Capture {
            name: n,
            body: Body::NoneOf { chars: c, space: s },
        } => {
            let mut c = c.clone();
            c.sort_unstable();
            c.dedup();
            let mut want = chars.to_vec();
            want.sort_unstable();
            n == name && c == want && *s == space
        }
        _ => false,
    }
}

/// The Qwen 3.5 tags, in the one spelling `CallSyntax::Tagged(Json, OwnLine)` reads:
/// `<tool_call>`, `<function=NAME>`, `<parameter=KEY>`, each value on a line of its own between
/// its tags, `</function>`, `</tool_call>`.
fn qwen_tags(
    open: &[Atom],
    close: &[Atom],
    tags: &[Atom],
    value: Option<&Content>,
) -> Option<Calls> {
    let open = open.strip_prefix(&[Atom::Spaces]).unwrap_or(open);
    let name_ok = matches!(open, [Atom::Lit(a), Atom::Spaces, Atom::Lit(b), name, Atom::Lit(c)]
        if a == "<tool_call>" && b == "<function=" && c == ">"
            && reads_none_of(name, "name", &['>', '\n'], false));
    let close_ok = close == [lit("</function>"), Atom::Spaces, lit("</tool_call>")];
    let tags_ok = matches!(tags, [Atom::Lit(a), key, Atom::Lit(b), Atom::Spaces, value, Atom::Spaces, Atom::Lit(c)]
        if a == "<parameter=" && b == ">" && c == "</parameter>"
            && reads_none_of(key, "key", &['>', '\n'], false)
            && *value == capture("value", Body::LazyAny));
    (name_ok && close_ok && tags_ok && value.is_none()).then(|| Calls {
        syntax: CallSyntax::Tagged(Spelling::Json, Placement::OwnLine),
        opens: vec!["<tool_call>".to_owned()],
        closes: vec!["</tool_call>".to_owned()],
        reenters: true,
    })
}

/// A call addressed to the tool: `RECIPIENT(?P<name>[^\s<]+)HEADER_CLOSE` opens it, and each
/// parameter is `K1(?P<key>[^"]+)K2\n(?P<value>.*?)\nV`, where the key's class stops at the first
/// character of `K2`; with `\n(?:V|(?=K1))` in place of `\nV`, the next key's tag closes a value
/// too.
fn addressed(
    scope: &str,
    open: &[Atom],
    close: &[Atom],
    tags: &[Atom],
    value: Option<&Content>,
) -> Result<Calls, Unsupported> {
    let [Atom::Lit(recipient), name, Atom::Lit(header_close)] = open else {
        return refuse(
            scope,
            "an open that is not a recipient, a name and a header's close",
        );
    };
    if !reads_none_of(name, "name", &['<'], true) {
        return refuse(scope, "a name that is not read as [^\\s<]+");
    }
    let (key_open, key, after_key, value_capture, rest) = match tags {
        [Atom::Lit(k1), key, Atom::Lit(k2), value, rest @ ..] => (k1, key, k2, value, rest),
        _ => return refuse(scope, "a tag_pattern that is not a key's tag and a value"),
    };
    let Some(key_close) = after_key.strip_suffix('\n').filter(|k2| !k2.is_empty()) else {
        return refuse(scope, "a key's tag not followed by a newline");
    };
    let stop = key_close.chars().next().unwrap_or_default();
    if !reads_none_of(key, "key", &[stop], false) {
        return refuse(scope, "a key not read up to its tag's close");
    }
    if *value_capture != capture("value", Body::LazyAny) {
        return refuse(scope, "a value not read as .*?");
    }
    let (value_close, next_key_closes) = match rest {
        [Atom::Lit(tail)] => match tail.strip_prefix('\n') {
            Some(v) if !v.is_empty() => (v.to_owned(), false),
            _ => return refuse(scope, "a value's close not after a newline"),
        },
        [Atom::Lit(newline), Atom::Alt(branches)] if newline == "\n" => match branches.as_slice() {
            [close, ahead] if *ahead == [Atom::Ahead(vec![lit(key_open)])] => {
                match close.as_slice() {
                    [Atom::Lit(v)] => (v.clone(), true),
                    _ => return refuse(scope, "a value's close that is not literal text"),
                }
            }
            _ => {
                return refuse(
                    scope,
                    "a value's close other than its tag or the next key's",
                )
            }
        },
        _ => return refuse(scope, "a value's close of another shape"),
    };
    match value {
        None => {}
        Some(content) if content.parser == ContentParser::Text && !content.strip => {}
        Some(_) => return refuse(scope, "a value_parser other than text with strip: false"),
    }
    let closes = literal_marker(scope, "close", close)?;
    let spelling = addressed::Spelling {
        header_close: Cow::Owned(header_close.clone()),
        key_open: Cow::Owned(key_open.clone()),
        key_close: Cow::Owned(key_close.to_owned()),
        value_close: Cow::Owned(value_close),
        next_key_closes,
    };
    Ok(Calls {
        syntax: CallSyntax::Addressed(spelling),
        opens: vec![recipient.clone()],
        closes,
        reenters: false,
    })
}
