//! The structural tags of a table whose calls are addressed to a tool ([`CallSyntax::Addressed`]):
//! a forced call, exactly one call, and a JSON answer, each written as the table reads the turn.
//!
//! A turn of such a table is a run of messages. In the synthetic spelling the tests use:
//!
//! ```text
//! <|channel|>analysis<|message|>THOUGHT<|end|>
//! <|start|>assistant to=get_weather<|channel|>commentary<|message|><parameter name="city">
//! Paris
//! </parameter><|call|>
//! ```
//!
//! (one line, the newlines around a value excepted). The tags are derived from the table's rows
//! ([`Format::addressed_tags`]), not written per model:
//! - **the thought first**: where a reasoning state is reached from the turn's own state through
//!   rows of markers only, the tag writes those markers, the thought as free text, and the row out
//!   of it from whose state markers lead to a call; the thought may be empty but is always
//!   written, as such a template renders it;
//! - **a call** is the markers of the rows into the arguments state (the recipient marker last),
//!   the tool's name, the spelling's header close, then the parameters: `key_open`, the key,
//!   `key_close`, a newline, the value, a newline and `value_close`, one after another with a
//!   newline between two, in the order of the schema's `properties`, required ones always and no
//!   undeclared ones (each value held at most 1 + ⌈log₂ n⌉ times among n optional ones);
//! - **between two calls**, the markers of a row out of the arguments state that is not a stop,
//!   up to the next call's own; **after the last**, the row out of it on a stop token
//!   ([`Format::stop_tokens`]), where the engine stops;
//! - **a JSON answer** is the markers from the thought's exit into the answer's state, then JSON
//!   of the schema, ended by the end-of-sequence token where the grammar ends;
//! - **tokens**: a marker token of the table ([`Format::marker_tokens`]) is a `token` element, so
//!   its spelling in ordinary tokens opens and closes nothing, and the engine stops on the stop
//!   token itself; free text excludes them, and a value written as text also excludes
//!   `value_close`;
//! - **values** as the parser types them back: a value that may be a string (following each
//!   `$ref` once, and an `allOf` of one schema) is text, a string enum one of its values, and any
//!   other value JSON of its schema, which carries the tool's root definitions only when it
//!   refers to them. JSON strings hold none of the words that end a value or the answer where the
//!   output is read (the rows out of the arguments and answer states, and `value_close`):
//!   xgrammar's `json_schema` has no exclude list, so the schema is rewritten, the pattern written
//!   once per document in a string definition that strings and keys refer to.
//!
//! The lengths of the tags' JSON text are worked out without building them
//! ([`AddressedTags::calls_len`] and its siblings), so a gateway can refuse a tag too large before
//! it builds one ([`AddressedTags::MAX_TAG_BYTES`]).

use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap, HashSet},
    rc::Rc,
};

use openai_protocol::common::Tool;
use serde_json::{json, Map, Value};

use super::{schema::has_local_reference, Bound, Grammar, Tag};
use crate::format::{CallSyntax, Emits, Format};

/// How deep `$ref`s, one-schema `allOf`s and branches are followed to tell whether a value may be
/// a string.
const MAX_REF_DEPTH: usize = 8;

/// The structural tags of an addressed table, derived from its rows; see the module
/// documentation.
#[derive(Clone, Debug)]
pub struct AddressedTags {
    /// The text that opens the turn: a prompt the tags follow ends with it.
    opener: String,
    /// The thought written first, if the table has one.
    thought: Option<Thought>,
    /// The text before a call's name: the markers into the arguments state.
    call_lead: String,
    header_close: String,
    key_open: String,
    key_close: String,
    value_close: String,
    /// The elements between two calls, and after the last.
    separator: Vec<Grammar>,
    last: Vec<Grammar>,
    /// The elements before a JSON answer, if the table has an answer state.
    answer_lead: Option<Vec<Grammar>>,
    /// The marker tokens, longest first for splitting.
    tokens: Vec<String>,
    /// What free text of a value excludes, and what a name, key or enum value may not hold.
    delimiters: Vec<String>,
    /// The pattern of a JSON string's text.
    pattern: Rc<str>,
    pieces: Pieces,
}

/// The thought: the elements that open it, what its text excludes, and where it ends.
#[derive(Clone, Debug)]
struct Thought {
    open: Vec<Grammar>,
    excludes: Vec<String>,
    end: Bound,
}

impl Format {
    /// The structural tags of this table's addressed calls, or why its rows give none, naming what
    /// is missing: calls that are not addressed, no turn opener, no row out of a call on a stop
    /// token or toward another call, a thought with no way out toward a call, or delimiters that
    /// a JSON string's pattern cannot leave out (each must begin with one character that none
    /// holds again).
    pub fn addressed_tags(&self) -> Result<AddressedTags, String> {
        let Some(CallSyntax::Addressed(spelling)) = self.call_syntax() else {
            return Err("the table's calls are not addressed to a tool".to_string());
        };
        let opener = self.turn_opener().ok_or("the table names no turn opener")?;
        let arguments = (0..self.states())
            .find(|&state| self.emits(state) == Emits::Arguments)
            .ok_or("the table has no arguments state")?;
        let mut tokens: Vec<String> = self.marker_token_texts().to_vec();
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        let split = |text: &str| split(text, &tokens);
        let holds_token = |text: &str| tokens.iter().any(|token| text.contains(token.as_str()));

        let exits = |state: usize| {
            self.transitions()
                .filter(move |&(source, _, _)| source == state)
                .map(|(_, on, target)| (self.terminal_text(on), target))
        };
        let path_text =
            |path: &[usize]| -> String { path.iter().map(|&on| self.terminal_text(on)).collect() };
        let last = exits(arguments)
            .find(|(text, _)| self.is_stop_token(text))
            .map(|(text, _)| text)
            .ok_or("no row out of a call is on a stop token")?;
        let joiner = exits(arguments)
            .filter(|(text, _)| !self.is_stop_token(text))
            .find_map(|(text, target)| {
                Some(format!(
                    "{text}{}",
                    path_text(&self.marker_path(target, arguments)?)
                ))
            })
            .ok_or("no row out of a call leads to another call")?;

        let start = 0;
        let reasoning = (0..self.states()).find(|&state| {
            self.emits(state) == Emits::Reasoning && self.marker_path(start, state).is_some()
        });
        let (thought, home, rest) = match reasoning {
            Some(reasoning) => {
                let open = self.marker_path(start, reasoning).unwrap_or_default();
                let (exit, home) = exits(reasoning)
                    .filter(|(text, _)| !self.is_stop_token(text))
                    .find(|&(_, target)| self.marker_path(target, arguments).is_some())
                    .ok_or("the thought has no row out toward a call")?;
                let (end, rest) = first_piece(exit, &tokens);
                let mut excludes = self.marker_token_texts().to_vec();
                for (text, _) in exits(reasoning) {
                    if !holds_token(text) && !excludes.iter().any(|known| known == text) {
                        excludes.push(text.to_string());
                    }
                }
                let thought = Thought {
                    open: split(&path_text(&open)),
                    excludes,
                    end,
                };
                (Some(thought), home, rest.to_string())
            }
            None => (None, start, String::new()),
        };
        let to_call = self
            .marker_path(home, arguments)
            .ok_or("no markers lead from the turn to a call")?;
        let call_lead = format!("{rest}{}", path_text(&to_call));
        let separator = joiner.strip_suffix(call_lead.as_str()).ok_or_else(|| {
            format!(
                "the markers between two calls ({joiner:?}) do not end with a call's own \
                 ({call_lead:?})"
            )
        })?;
        let answer = (0..self.states()).find_map(|state| {
            let answers = self.emits(state) == Emits::Content && !self.holds_only_markers(state);
            answers
                .then(|| self.marker_path(home, state))
                .flatten()
                .map(|path| (state, path))
        });
        let answer_lead = answer
            .as_ref()
            .map(|(_, path)| split(&format!("{rest}{}", path_text(path))));

        // What a value's text excludes, and what a name, key or enum value may not hold.
        let mut delimiters = vec![spelling.value_close.to_string()];
        delimiters.extend(self.marker_token_texts().iter().cloned());
        for (text, _) in exits(arguments) {
            if !holds_token(text) && !delimiters.iter().any(|known| known == text) {
                delimiters.push(text.to_string());
            }
        }
        // What a JSON string excludes: the words that end a value or the answer where the
        // output is read, each once, none holding another.
        let value_close = spelling.value_close.to_string();
        let mut words: Vec<&str> = exits(arguments).map(|(text, _)| text).collect();
        if let Some((state, _)) = answer {
            words.extend(exits(state).map(|(text, _)| text));
        }
        words.push(&value_close);
        let words: Vec<&str> = words
            .iter()
            .copied()
            .filter(|word| {
                !words
                    .iter()
                    .any(|other| other.len() < word.len() && word.contains(other))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let pattern: Rc<str> = excluding(&words)?.into();

        let mut tags = AddressedTags {
            opener: opener.to_string(),
            thought,
            call_lead,
            header_close: spelling.header_close.to_string(),
            key_open: spelling.key_open.to_string(),
            key_close: spelling.key_close.to_string(),
            value_close,
            separator: split(separator),
            last: split(last),
            answer_lead,
            tokens,
            delimiters,
            pattern: pattern.clone(),
            pieces: Pieces::default(),
        };
        tags.pieces = Pieces::of(&tags);
        Ok(tags)
    }

    /// The shortest run of rows from `from` to `to` whose states between hold only markers (a
    /// wrapper state, or one the template writes only markers in), none on a stop token: the
    /// markers the template writes to get there. Rows are taken in their order among equals.
    fn marker_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        let passes =
            |state: usize| self.emits(state) == Emits::Wrapper || self.holds_only_markers(state);
        let mut reached: Vec<Option<(usize, usize)>> = vec![None; self.states()];
        let mut seen = vec![false; self.states()];
        seen[from] = true;
        let mut frontier = vec![from];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &state in &frontier {
                for (source, on, target) in self.transitions() {
                    let text = self.terminal_text(on);
                    if source != state || self.is_stop_token(text) {
                        continue;
                    }
                    if target == to {
                        let mut path = vec![on];
                        let mut at = state;
                        while let Some((before, on)) = reached[at] {
                            path.push(on);
                            at = before;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    if !seen[target] && passes(target) {
                        seen[target] = true;
                        reached[target] = Some((state, on));
                        next.push(target);
                    }
                }
            }
            frontier = next;
        }
        None
    }
}

/// `text` as elements: each marker token among `tokens` (longest first where two begin at one
/// place) a token, the text between a constant string.
fn split(text: &str, tokens: &[String]) -> Vec<Grammar> {
    let mut out = Vec::new();
    let mut plain = String::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        match tokens
            .iter()
            .find(|token| !token.is_empty() && rest.starts_with(token.as_str()))
        {
            Some(token) => {
                if !plain.is_empty() {
                    out.push(Grammar::ConstString(std::mem::take(&mut plain)));
                }
                out.push(Grammar::Token(token.clone()));
                rest = &rest[token.len()..];
            }
            None => {
                plain.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    if !plain.is_empty() {
        out.push(Grammar::ConstString(plain));
    }
    out
}

/// The first piece of `text` as a tag's bound (its first marker token, or the text up to one),
/// and the rest of the text.
fn first_piece<'t>(text: &'t str, tokens: &[String]) -> (Bound, &'t str) {
    match split(text, tokens).into_iter().next() {
        Some(Grammar::Token(token)) => {
            let rest = &text[token.len()..];
            (Bound::Token(token), rest)
        }
        Some(Grammar::ConstString(plain)) => {
            let rest = &text[plain.len()..];
            (Bound::Text(plain), rest)
        }
        _ => (Bound::Text(String::new()), text),
    }
}

/// `text` after `elements`, run into the constant string they end with.
fn push_text(elements: &mut Vec<Grammar>, text: &str) {
    if let Some(Grammar::ConstString(last)) = elements.last_mut() {
        last.push_str(text);
    } else if !text.is_empty() {
        elements.push(const_string(text));
    }
}

/// The elements one after another; one element stands as it is.
fn sequence(mut elements: Vec<Grammar>) -> Grammar {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    Grammar::Sequence(elements)
}

/// One of the elements; one element stands as it is.
fn one_of(mut elements: Vec<Grammar>) -> Grammar {
    if elements.len() == 1 {
        return elements.remove(0);
    }
    Grammar::Or(elements)
}

fn optional(content: Grammar) -> Grammar {
    Grammar::Optional(Box::new(content))
}

fn star(content: Grammar) -> Grammar {
    Grammar::Star(Box::new(content))
}

fn const_string(text: &str) -> Grammar {
    Grammar::ConstString(text.to_string())
}

fn any_text(excludes: &[String]) -> Grammar {
    Grammar::AnyText {
        excludes: excludes.to_vec(),
    }
}

impl AddressedTags {
    /// The most bytes of JSON text a tag may take; a gateway refuses a request whose tag would
    /// take more before building it ([`Self::calls_len`]). A call holds each parameter's value
    /// once, or at most 1 + ⌈log₂ n⌉ times among n parameters none of which is required, but each
    /// JSON value carries the definitions it refers to, as a structural tag cannot refer to a
    /// schema written elsewhere: a tag grows with the values that refer to definitions times the
    /// definitions. Forced tags of agent tool sets take 0.05–2 MB; 4 MiB keeps about twice the
    /// largest, and a tag that size builds in under a second.
    ///
    /// It bounds the tag, not the engine's work to compile it. With xgrammar 0.2.3 that grows
    /// about with the square of the distinct parameters and fields the tag holds, and with the
    /// JSON values of distinct schemas that hold strings, each a document of its own that compiles
    /// its own copy of the string pattern; a bound on the compile belongs to the engine.
    pub const MAX_TAG_BYTES: usize = 4 << 20;

    /// The most parameters (keys of its schema's `properties`) a tool may declare for these tags;
    /// a gateway refuses a request with a tool that declares more
    /// ([`Self::check_parameter_counts`]). Tools declare tens; a tool's body compiles in a time
    /// that grows with the square of its parameters.
    pub const MAX_TOOL_PARAMETERS: usize = 1024;

    /// The most parameters the tools of one tag may declare together; a gateway refuses more with
    /// [`Self::check_parameter_counts`]. Agent tool sets declare up to about 1,100.
    pub const MAX_TOTAL_PARAMETERS: usize = 4096;

    /// The text that opens the turn the tags write: a prompt they follow must end with it, since
    /// every tag starts a new turn.
    pub fn generation_prompt(&self) -> &str {
        &self.opener
    }

    /// Refuses tools whose calls the tags cannot write so that the table reads them back: a name
    /// the addressed syntax does not read whole (empty, with whitespace or `<`, or holding a
    /// delimiter or the header's close), a parameters schema that is not an object, a required
    /// parameter it does not declare, a parameter name the key's tag cannot carry, an empty enum,
    /// and a string enum value holding a delimiter.
    pub fn check_tools(&self, tools: &[Tool]) -> Result<(), String> {
        let key_stop = self.key_close.chars().next();
        for tool in tools {
            let name = &tool.function.name;
            if !self.readable_name(name) {
                return Err(format!(
                    "tool name '{name}' cannot be read back from this model's call format \
                     (it must be non-empty, without whitespace or '<')"
                ));
            }
            let parameters = &tool.function.parameters;
            let object = parameters.as_object().filter(|schema| {
                schema
                    .get("type")
                    .is_none_or(|kind| kind.as_str() == Some("object"))
            });
            let Some(object) = object else {
                return Err(format!(
                    "tool '{name}': this model's call format takes an object parameters schema"
                ));
            };
            let declared = object.get("properties").and_then(Value::as_object);
            let required = object.get("required").and_then(Value::as_array);
            for key in required.into_iter().flatten() {
                if !key
                    .as_str()
                    .is_some_and(|key| declared.is_some_and(|d| d.contains_key(key)))
                {
                    return Err(format!(
                        "tool '{name}': required parameter {key} is not among its properties"
                    ));
                }
            }
            for (key, schema) in properties(object) {
                if schema
                    .get("enum")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
                {
                    return Err(format!(
                        "tool '{name}': parameter '{key}' has an empty enum"
                    ));
                }
                if key.is_empty()
                    || key_stop.is_some_and(|stop| key.contains(stop))
                    || self.holds_delimiter(key)
                {
                    return Err(format!(
                        "tool '{name}': parameter name '{key}' cannot be written in this \
                         model's call format"
                    ));
                }
                if string_enum(schema)
                    .is_some_and(|values| values.iter().any(|v| self.holds_delimiter(v)))
                {
                    return Err(format!(
                        "tool '{name}': a value of parameter '{key}' holds a delimiter of \
                         this model's call format"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Refuses a tool that declares more than [`Self::MAX_TOOL_PARAMETERS`] parameters, naming it
    /// and the limit, then tools that declare more than [`Self::MAX_TOTAL_PARAMETERS`] together.
    /// Only a tool's own parameters count, not the fields of a parameter's value.
    pub fn check_parameter_counts(tools: &[Tool]) -> Result<(), String> {
        let mut total: usize = 0;
        for tool in tools {
            let count = tool
                .function
                .parameters
                .get("properties")
                .and_then(Value::as_object)
                .map_or(0, Map::len);
            if count > Self::MAX_TOOL_PARAMETERS {
                return Err(format!(
                    "tool '{}' declares {count} parameters, over this model's limit of {} \
                     parameters per tool",
                    tool.function.name,
                    Self::MAX_TOOL_PARAMETERS
                ));
            }
            total = total.saturating_add(count);
        }
        if total > Self::MAX_TOTAL_PARAMETERS {
            return Err(format!(
                "the tools declare {total} parameters together, over this model's limit of \
                 {} parameters per request",
                Self::MAX_TOTAL_PARAMETERS
            ));
        }
        Ok(())
    }

    /// The tag for a forced tool choice: the thought, then one or more calls to `tools`, the
    /// separator between two and the stop after the last.
    pub fn calls(&self, tools: &[Tool]) -> Grammar {
        self.calls_of(&tools.iter().collect::<Vec<_>>())
    }

    pub(super) fn calls_of(&self, tools: &[&Tool]) -> Grammar {
        let mut elements: Vec<Grammar> = self.thought().into_iter().collect();
        elements.extend(self.call_run(tools));
        sequence(elements)
    }

    /// The tag for a forced choice of exactly one call (a named function, or parallel calls turned
    /// off): the thought, then one call to one of `tools`, then the stop.
    pub fn single_call(&self, tools: &[Tool]) -> Grammar {
        let mut elements: Vec<Grammar> = self.thought().into_iter().collect();
        elements.push(self.any_call(&tools.iter().collect::<Vec<_>>()));
        elements.extend(self.last.iter().cloned());
        sequence(elements)
    }

    /// The tag for a JSON `response_format`: the thought, then the answer's markers and JSON of
    /// `schema`, ended by the end-of-sequence token. With `tools` (choice `auto`) the model may
    /// call them instead, as an agent loop that wants a typed final answer needs. A table without
    /// an answer state gives the calls alone, or with no tools the thought alone.
    pub fn response_format(&self, schema: &Value, tools: &[Tool]) -> Grammar {
        let tools: Vec<&Tool> = tools.iter().collect();
        let mut elements: Vec<Grammar> = self.thought().into_iter().collect();
        let answer = self.answer(schema);
        let choice = match answer {
            Some(answer) if tools.is_empty() => Some(answer),
            Some(answer) => Some(one_of(vec![answer, sequence(self.call_run(&tools))])),
            None if tools.is_empty() => None,
            None => Some(sequence(self.call_run(&tools))),
        };
        elements.extend(choice);
        sequence(elements)
    }

    /// The length of [`Self::calls`]'s payload as JSON text (`{"format": …}`, as the request
    /// carries it), or more, worked out without building the tag: each parameter value's length
    /// once (a JSON value's without rewriting its schema), each tool's root definitions' once, and
    /// the copies the tag holds counted, so the work is about linear in `tools` and takes little
    /// memory. It is the exact length unless a schema's definitions take the names the tag's own
    /// would.
    pub fn calls_len(&self, tools: &[Tool]) -> usize {
        let pieces = &self.pieces;
        let run = self.call_run_len(tools);
        let count = usize::from(self.thought.is_some()) + run.len();
        let total = run.iter().fold(
            if self.thought.is_some() {
                pieces.thought
            } else {
                0
            },
            |sum, &len| sum.saturating_add(len),
        );
        pieces
            .payload
            .saturating_add(listed_len(pieces.sequence, count, total))
    }

    /// The length of [`Self::single_call`]'s payload, or more; see [`Self::calls_len`].
    pub fn single_call_len(&self, tools: &[Tool]) -> usize {
        let pieces = &self.pieces;
        let mut lens: Vec<usize> = Vec::new();
        if self.thought.is_some() {
            lens.push(pieces.thought);
        }
        lens.push(self.any_call_len(tools));
        lens.extend(pieces.last.iter().copied());
        pieces.payload.saturating_add(listed_len(
            pieces.sequence,
            lens.len(),
            lens.iter()
                .fold(0, |sum: usize, &len| sum.saturating_add(len)),
        ))
    }

    /// The length of [`Self::response_format`]'s payload, or more; see [`Self::calls_len`].
    pub fn response_format_len(&self, schema: &Value, tools: &[Tool]) -> usize {
        let pieces = &self.pieces;
        let answer = self.answer_lead.as_ref().map(|lead| {
            let json_value = pieces.json_value.saturating_add(self.rewrite_len(schema));
            let total = lead.iter().fold(json_value, |sum, e| {
                sum.saturating_add(json_len(&e.to_json()))
            });
            listed_len(pieces.sequence, lead.len() + 1, total)
        });
        let calls = (!tools.is_empty()).then(|| {
            let run = self.call_run_len(tools);
            let total = run
                .iter()
                .fold(0, |sum: usize, &len| sum.saturating_add(len));
            listed_len(pieces.sequence, run.len(), total)
        });
        let choice = match (answer, calls) {
            (Some(answer), Some(calls)) => {
                Some(listed_len(pieces.one_of, 2, answer.saturating_add(calls)))
            }
            (answer, calls) => answer.or(calls),
        };
        let mut lens: Vec<usize> = Vec::new();
        if self.thought.is_some() {
            lens.push(pieces.thought);
        }
        lens.extend(choice);
        pieces.payload.saturating_add(listed_len(
            pieces.sequence,
            lens.len(),
            lens.iter()
                .fold(0, |sum: usize, &len| sum.saturating_add(len)),
        ))
    }

    /// The name's capture (`[^\s<]+`) takes the whole name, and the header's close follows it.
    fn readable_name(&self, name: &str) -> bool {
        !name.is_empty()
            && !name.chars().any(|c| c == '<' || c.is_whitespace())
            && !name.contains(self.header_close.as_str())
            && !self.holds_delimiter(name)
    }

    fn holds_delimiter(&self, text: &str) -> bool {
        self.delimiters
            .iter()
            .any(|delimiter| text.contains(delimiter.as_str()))
    }

    /// The thought: its markers, then a tag of free text ended by the first piece of its exit.
    fn thought(&self) -> Option<Grammar> {
        let thought = self.thought.as_ref()?;
        let mut elements = thought.open.clone();
        elements.push(Grammar::Tag(Tag::new(
            "",
            any_text(&thought.excludes),
            thought.end.clone(),
        )));
        Some(sequence(elements))
    }

    /// The answer: its markers, then JSON of `schema`.
    fn answer(&self, schema: &Value) -> Option<Grammar> {
        let mut elements = self.answer_lead.clone()?;
        elements.push(self.json_value(schema));
        Some(sequence(elements))
    }

    /// One or more calls to `tools`: a call, then any number of the separator and a call, then
    /// the stop.
    fn call_run(&self, tools: &[&Tool]) -> Vec<Grammar> {
        let any_call = self.any_call(tools);
        let mut next = self.separator.clone();
        next.push(any_call.clone());
        let mut run = vec![any_call, star(sequence(next))];
        run.extend(self.last.iter().cloned());
        run
    }

    fn any_call(&self, tools: &[&Tool]) -> Grammar {
        one_of(tools.iter().map(|tool| self.call(tool)).collect())
    }

    /// The markers into the arguments state, the name and the header's close, then the
    /// parameters.
    fn call(&self, tool: &Tool) -> Grammar {
        let mut elements = self.call_opening(&tool.function.name);
        let empty = Map::new();
        let schema = tool.function.parameters.as_object().unwrap_or(&empty);
        elements.extend(self.parameters(schema));
        sequence(elements)
    }

    fn call_opening(&self, name: &str) -> Vec<Grammar> {
        let mut elements = split(&self.call_lead, &self.tokens);
        push_text(&mut elements, name);
        for element in split(&self.header_close, &self.tokens) {
            match element {
                Grammar::ConstString(text) => push_text(&mut elements, &text),
                token => elements.push(token),
            }
        }
        elements
    }

    /// The parameters in schema order, joined by `\n`: each required one, and each other one or
    /// nothing. `None` for a tool without parameters.
    ///
    /// Whether a parameter is written decides whether a `\n` comes between it and the next one
    /// written, so the body holds a parameter's value once in each place it may be written from,
    /// and the places are kept few:
    /// - with a required parameter, each optional one before the first required one is written
    ///   with the `\n` after it, or not at all, then the first required one, then each later one
    ///   with the `\n` before it (or, optional, not at all): each value once;
    /// - with none, the body is empty or [`one_or_more`] of them, which holds a value at most
    ///   1 + ⌈log₂ n⌉ times for n parameters, nested as deep.
    fn parameters(&self, schema: &Map<String, Value>) -> Option<Grammar> {
        let required = required_names(schema);
        let defs = Defs::of(schema);
        let params: Vec<Parameter> = properties(schema)
            .map(|(key, schema)| Parameter {
                tags: self,
                key,
                value: self.value(schema, &defs),
                required: required.contains(key.as_str()),
            })
            .collect();
        if params.is_empty() {
            return None;
        }
        let Some(first) = params.iter().position(|p| p.required) else {
            return Some(optional(one_or_more(&params)));
        };
        let mut elements: Vec<Grammar> = params[..first].iter().map(Parameter::before).collect();
        elements.push(params[first].first());
        elements.extend(params[first + 1..].iter().map(Parameter::after));
        Some(sequence(elements))
    }

    /// `key_open KEY key_close \n` value `\n value_close`, after a `\n` when another parameter
    /// precedes it.
    fn parameter(&self, key: &str, value: Grammar, after_another: bool) -> Grammar {
        let lead = if after_another { "\n" } else { "" };
        Grammar::Tag(Tag::new(
            format!("{lead}{}{key}{}\n", self.key_open, self.key_close),
            value,
            format!("\n{}", self.value_close),
        ))
    }

    /// A parameter value as the parser reads it back: a string enum one of its values, a value
    /// that may be a string text, and any other one JSON of its schema.
    fn value(&self, schema: &Value, defs: &Defs) -> Grammar {
        self.raw_value(schema, defs)
            .unwrap_or_else(|| self.json_value(&defs.attach(schema)))
    }

    fn raw_value(&self, schema: &Value, defs: &Defs) -> Option<Grammar> {
        Some(match raw(schema, defs)? {
            Raw::Enum(values) => one_of(values.iter().map(|v| const_string(v)).collect()),
            Raw::Text => self.raw_text(),
        })
    }

    /// Text that holds no delimiter, for a value that may be any string.
    fn raw_text(&self) -> Grammar {
        any_text(&self.delimiters)
    }

    /// JSON of `schema` whose strings, as written, hold none of the words that end a value or
    /// the answer ([`JsonStrings`]).
    fn json_value(&self, schema: &Value) -> Grammar {
        Grammar::JsonSchema {
            schema: JsonStrings::rewrite(schema, &self.pattern),
            style: None,
        }
    }
}

/// A parameter of a call: its name, its value's format, and whether the schema requires it.
struct Parameter<'a> {
    tags: &'a AddressedTags,
    key: &'a str,
    value: Grammar,
    required: bool,
}

impl Parameter<'_> {
    /// Written first.
    fn first(&self) -> Grammar {
        self.tags.parameter(self.key, self.value.clone(), false)
    }

    /// Before another one, with the `\n` after it; optional.
    fn before(&self) -> Grammar {
        optional(sequence(vec![self.first(), const_string("\n")]))
    }

    /// After another one, with the `\n` before it; optional unless required.
    fn after(&self) -> Grammar {
        let after = self.tags.parameter(self.key, self.value.clone(), true);
        if self.required {
            after
        } else {
            optional(after)
        }
    }
}

/// One or more of `params`, all optional, in order and joined by `\n`: with the first half, one or
/// more of it then each of the second half with the `\n` before it or not at all; without, one or
/// more of the second half.
fn one_or_more(params: &[Parameter]) -> Grammar {
    if let [only] = params {
        return only.first();
    }
    let (head, tail) = params.split_at(params.len().div_ceil(2));
    let mut with_head = vec![one_or_more(head)];
    with_head.extend(tail.iter().map(Parameter::after));
    one_of(vec![sequence(with_head), one_or_more(tail)])
}

/// The names `schema` lists as required, as a set: a long list is looked up once per parameter.
fn required_names(schema: &Map<String, Value>) -> HashSet<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn properties(schema: &Map<String, Value>) -> impl Iterator<Item = (&String, &Value)> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
}

/// How a value is written as text: one of a string enum's values, or text.
enum Raw<'a> {
    Enum(Vec<&'a String>),
    Text,
}

/// How a value of `schema` is written as text; `None` for JSON.
fn raw<'a>(schema: &'a Value, defs: &Defs) -> Option<Raw<'a>> {
    if let Some(values) = string_enum(schema) {
        return Some(Raw::Enum(values));
    }
    admits_string(schema, defs, 0).then_some(Raw::Text)
}

/// The keywords xgrammar 0.2.3 reads a type or a structure from; a schema with none of them is any
/// JSON to it.
const TYPED: [&str; 13] = [
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "properties",
    "additionalProperties",
    "unevaluatedProperties",
    "items",
    "prefixItems",
    "unevaluatedItems",
];

/// The keywords xgrammar 0.2.3 reads before a schema's `type`: a schema with one of them is what
/// that keyword gives, whatever its type says.
const READ_BEFORE_TYPE: [&str; 6] = ["$ref", "const", "enum", "anyOf", "oneOf", "allOf"];

/// The keywords for which a string keeps xgrammar's own rule.
const OWN_STRING_RULE: [&str; 6] = [
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "enum",
    "const",
];

/// The keywords that only annotate a schema.
const ANNOTATIONS: [&str; 8] = [
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
];

/// The names [`JsonStrings`] gives its two definitions in a document whose `$defs` takes neither.
const STRING_NAME: &str = "symphony_string";
const ANY_NAME: &str = "symphony_json";

/// Rewrites a schema for [`AddressedTags::json_value`] (xgrammar 0.2.3's reading of it), so that
/// its strings, as written, hold none of the words the pattern leaves out. The pattern is written
/// once, in a definition of a string of it that the document's strings and keys refer to:
/// xgrammar compiles a pattern again wherever it is written, but a definition once per document.
/// - a string schema refers to the definition, and a union type with a string is an `anyOf` of the
///   reference and the other types;
/// - a value of any type (`true`, a schema with no type or structure, an `allOf` of several
///   schemas) refers to one definition of any JSON whose strings and keys refer to the string;
/// - an object open to other keys (`additionalProperties` or `unevaluatedProperties` other than
///   false, or none of them nor `properties` nor `patternProperties`) takes them as
///   `additionalProperties` named by `propertyNames`, a reference to the string
///   ([`named_others`]);
/// - an array without `items` or `prefixItems` takes any JSON.
///
/// A string with its own `pattern`, `format`, length, `enum` or `const` keeps xgrammar's rule, a
/// schema xgrammar reads by a keyword before its type ([`READ_BEFORE_TYPE`]) keeps what that
/// keyword reads, and declared property names are the schema's own. Annotations
/// ([`ANNOTATIONS`]), which xgrammar does not read, are left out, so values of one schema are one
/// document, which xgrammar compiles once for the tag.
///
/// It holds the pattern and the names of the two definitions, the lengths those names give
/// ([`NameLens`]), and how many references to each it wrote. A clone shares the names.
#[derive(Clone)]
struct JsonStrings {
    pattern: Rc<str>,
    string: Rc<str>,
    any: Rc<str>,
    lens: NameLens,
    string_refs: usize,
    any_refs: usize,
}

/// The lengths of the JSON text of a reference to each of the two definitions and of each as a
/// `$defs` member (its key, a colon and its text), worked out once with their names, so that a
/// length counts a reference without writing a name that may be long.
#[derive(Clone, Copy, Debug, Default)]
struct NameLens {
    string_ref: usize,
    any_ref: usize,
    string_def: usize,
    any_def: usize,
}

impl NameLens {
    fn of(string: &str, any: &str, pattern: &str) -> Self {
        let member = |name: &str, definition: &Value| json_len(name) + 1 + json_len(definition);
        let [string_def, any_def] = definitions(string, any, pattern);
        Self {
            string_ref: json_len(&reference(string)),
            any_ref: json_len(&reference(any)),
            string_def: member(string, &string_def),
            any_def: member(any, &any_def),
        }
    }
}

impl JsonStrings {
    fn rewrite(schema: &Value, pattern: &Rc<str>) -> Value {
        let mut root = match schema {
            Value::Object(map) => map.clone(),
            Value::Bool(true) => Map::new(),
            _ => return schema.clone(),
        };
        // The root's definitions stay at the root, whatever it becomes.
        let mut definitions: Vec<(String, Value)> = ["$defs", "definitions"]
            .into_iter()
            .filter_map(|key| root.remove_entry(key))
            .collect();
        let own = match definitions.iter().find(|(key, _)| key == "$defs") {
            Some((_, Value::Object(own))) => Some(own),
            Some(_) => return schema.clone(),
            None => None,
        };
        let mut strings = Self::new(own, pattern, None);
        let mut out = Value::Object(root);
        strings.schema(&mut out);
        for (_, definitions) in &mut definitions {
            if let Value::Object(definitions) = definitions {
                definitions.values_mut().for_each(|d| strings.schema(d));
            }
        }
        let Value::Object(mut out) = out else {
            return out;
        };
        out.extend(definitions);
        let added = strings.added();
        if !added.is_empty() {
            let own = out.entry("$defs").or_insert_with(|| json!({}));
            if let Value::Object(own) = own {
                own.extend(added);
            }
        }
        Value::Object(out)
    }

    /// Names the two definitions apart from those in `own`, the document's `$defs`; `lens` are
    /// the lengths of the default names, when worked out already.
    fn new(own: Option<&Map<String, Value>>, pattern: &Rc<str>, lens: Option<NameLens>) -> Self {
        let unused = |name: &str| {
            let mut name = name.to_owned();
            while own.is_some_and(|own| own.contains_key(&name)) {
                name.push('_');
            }
            name
        };
        let (string, any) = (unused(STRING_NAME), unused(ANY_NAME));
        let lens = match lens {
            Some(lens) if string == STRING_NAME && any == ANY_NAME => lens,
            _ => NameLens::of(&string, &any, pattern),
        };
        Self {
            pattern: pattern.clone(),
            string: string.into(),
            any: any.into(),
            lens,
            string_refs: 0,
            any_refs: 0,
        }
    }

    /// The definitions the references written so far refer to, to add to the document's `$defs`.
    fn added(&self) -> Vec<(String, Value)> {
        let [string, any] = definitions(&self.string, &self.any, &self.pattern);
        let mut added = Vec::new();
        if self.string_refs > 0 || self.any_refs > 0 {
            added.push((self.string.to_string(), string));
        }
        if self.any_refs > 0 {
            added.push((self.any.to_string(), any));
        }
        added
    }

    /// [`Self::added`]'s members' lengths, from [`NameLens`].
    fn added_lens(&self) -> Vec<usize> {
        let mut added = Vec::new();
        if self.string_refs > 0 || self.any_refs > 0 {
            added.push(self.lens.string_def);
        }
        if self.any_refs > 0 {
            added.push(self.lens.any_def);
        }
        added
    }

    fn string(&mut self) -> Value {
        self.string_refs = self.string_refs.saturating_add(1);
        reference(&self.string)
    }

    fn any(&mut self) -> Value {
        self.any_refs = self.any_refs.saturating_add(1);
        reference(&self.any)
    }

    fn schema(&mut self, node: &mut Value) {
        let map = match node {
            Value::Bool(true) => {
                *node = self.any();
                return;
            }
            Value::Object(map) => map,
            _ => return,
        };
        map.retain(|key, _| !ANNOTATIONS.contains(&key.as_str()));
        for key in ["properties", "patternProperties", "$defs", "definitions"] {
            if let Some(Value::Object(members)) = map.get_mut(key) {
                members.values_mut().for_each(|m| self.schema(m));
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems", "items"] {
            if let Some(Value::Array(members)) = map.get_mut(key) {
                members.iter_mut().for_each(|m| self.schema(m));
            }
        }
        for key in [
            "items",
            "additionalProperties",
            "unevaluatedProperties",
            "unevaluatedItems",
        ] {
            if let Some(member) = map.get_mut(key).filter(|m| !m.is_array()) {
                self.schema(member);
            }
        }
        // xgrammar reads an allOf of several schemas as any JSON.
        let several = map
            .get("allOf")
            .and_then(Value::as_array)
            .is_some_and(|all| all.len() > 1);
        if several || !TYPED.iter().any(|key| map.contains_key(*key)) {
            *node = self.any();
            return;
        }
        if READ_BEFORE_TYPE.iter().any(|key| map.contains_key(*key)) {
            return;
        }
        let types: Vec<&str> = match map.get("type") {
            Some(Value::String(kind)) => vec![kind.as_str()],
            Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        let has = |keys: &[&str]| keys.iter().any(|key| map.contains_key(*key));
        let string = types.contains(&"string") && !has(&OWN_STRING_RULE);
        let object = types.contains(&"object")
            || (types.is_empty()
                && has(&[
                    "properties",
                    "additionalProperties",
                    "unevaluatedProperties",
                ]));
        let array = types.contains(&"array")
            || (types.is_empty() && has(&["items", "prefixItems", "unevaluatedItems"]));
        let free_array = array && !has(&["items", "prefixItems", "unevaluatedItems"]);
        if object {
            self.open_object(map);
        }
        if free_array {
            map.insert("items".to_owned(), self.any());
        }
        if string {
            // A string is the definition, and a union type with one an `anyOf` of it and the
            // other types (each of which xgrammar reads with the same keywords).
            let others = other_types(map.get("type"));
            if others.is_empty() {
                *node = self.string();
            } else {
                let others = others.into_iter().cloned().collect();
                map.insert("type".to_owned(), Value::Array(others));
                let rest = std::mem::take(map);
                *node = json!({"anyOf": [self.string(), rest]});
            }
        }
    }

    /// The other keys of an object, if it takes any, named by the string definition
    /// ([`named_others`]), or else by the pattern as a key of `patternProperties`: xgrammar writes
    /// the keys an `additionalProperties` (or an `unevaluatedProperties`) allows as any JSON
    /// string, and reads an object with no keyword about its keys as any JSON object.
    fn open_object(&mut self, map: &mut Map<String, Value>) {
        const OTHERS: [&str; 2] = ["additionalProperties", "unevaluatedProperties"];
        let closed = OTHERS
            .iter()
            .any(|key| map.get(*key) == Some(&Value::Bool(false)));
        if closed
            || map
                .get("patternProperties")
                .is_some_and(|patterns| !patterns.is_object())
        {
            return;
        }
        let keyed = ["properties", "patternProperties"]
            .iter()
            .any(|key| map.contains_key(*key));
        let others = OTHERS
            .map(|key| map.remove(key))
            .into_iter()
            .flatten()
            .next();
        let others = match others {
            Some(others) => others,
            None if keyed => return,
            None => self.any(),
        };
        match named_others(map) {
            Some(declared) => {
                if !declared {
                    map.insert("properties".to_owned(), json!({"": others.clone()}));
                }
                map.insert("additionalProperties".to_owned(), others);
                map.insert("propertyNames".to_owned(), self.string());
            }
            None => {
                let patterns = map.entry("patternProperties").or_insert_with(|| json!({}));
                if let Value::Object(patterns) = patterns {
                    patterns.insert(self.pattern.to_string(), others);
                }
            }
        }
        map.entry("type").or_insert_with(|| json!("object"));
    }
}

/// The entries of a `type` array other than `"string"`.
fn other_types(kinds: Option<&Value>) -> Vec<&Value> {
    match kinds {
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter(|kind| kind.as_str() != Some("string"))
            .collect(),
        _ => Vec::new(),
    }
}

/// How an open object's other keys are named once rewritten. xgrammar 0.2.3 names them by
/// `propertyNames` (here a reference to the string definition, compiled once) only beside
/// declared `properties` and without `patternProperties`; without declared properties it gives
/// them any value. So `Some(true)` for an object that declares properties, and `Some(false)` for
/// one that declares and requires none, which declares one named "" taking what the other keys
/// take: that changes nothing, as "" may be another key. `None` for the rest, rare, which take the
/// other keys as `patternProperties` of the pattern: an object with `patternProperties` of its
/// own, or that declares no properties but requires some or bounds their number, which xgrammar
/// would read otherwise beside a declared one.
fn named_others(map: &Map<String, Value>) -> Option<bool> {
    let empty_object = |key: &str| {
        map.get(key)
            .is_none_or(|v| v.as_object().is_some_and(Map::is_empty))
    };
    if !empty_object("patternProperties") {
        return None;
    }
    if map
        .get("properties")
        .and_then(Value::as_object)
        .is_some_and(|declared| !declared.is_empty())
    {
        return Some(true);
    }
    let requires = map
        .get("required")
        .is_some_and(|r| r.as_array().is_none_or(|r| !r.is_empty()));
    let counted = ["minProperties", "maxProperties"]
        .iter()
        .any(|key| map.contains_key(*key));
    (empty_object("properties") && !requires && !counted).then_some(false)
}

/// A reference to the definition `name` in a document's `$defs`.
fn reference(name: &str) -> Value {
    json!({"$ref": format!("#/$defs/{name}")})
}

/// The two definitions [`JsonStrings`] adds, named `string` and `any`: a string of `pattern`, and
/// any JSON whose strings and keys are such strings (an object as [`named_others`] gives it).
fn definitions(string: &str, any: &str, pattern: &str) -> [Value; 2] {
    let any = reference(any);
    [
        json!({"type": "string", "pattern": pattern}),
        json!({"anyOf": [
            reference(string),
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
            {"type": "array", "items": any},
            {"type": "object", "properties": {"": any}, "additionalProperties": any,
             "propertyNames": reference(string)},
        ]}),
    ]
}

/// A regex for what a JSON string holds between its quotes, as written, that holds none of
/// `words`; each word starts with one lead character (`<`, say) and holds no other. It reads
/// `(U|D<(L|QE))*D`, with `<` for the lead: U is a character other than `<`, `"`, `\` and the
/// controls, or an escape E; Q is what may follow a `<` toward a word without ending it; L is Q
/// and then a character that leads toward no word; D is `<` Q any number of times. So each `<`
/// either starts a run that leaves every word, or is followed by another `<` or by the end.
/// Words that do not share a lead, or hold it again, are refused.
fn excluding(words: &[&str]) -> Result<String, String> {
    const ESCAPE: &str = r#"\\(["\\/bfnrt]|u[0-9a-fA-F]{4})"#;
    let lead = words
        .first()
        .and_then(|word| word.chars().next())
        .ok_or("no word ends a value")?;
    let fits =
        |word: &&str| word.starts_with(lead) && word.chars().filter(|&c| c == lead).count() == 1;
    if lead == '"' || lead == '\\' || lead.is_control() || !words.iter().all(fits) {
        return Err(format!(
            "the words that end a value ({words:?}) do not each begin with one character that \
             none holds again"
        ));
    }
    let start = escaped(lead);
    let partial = toward(words, &lead.to_string(), false);
    let dangling = format!("({start}{partial})*");
    Ok(format!(
        "({char}|{ESCAPE}|{dangling}{start}({leave}|{partial}{ESCAPE}))*{dangling}",
        char = json_char(lead, &BTreeSet::new()),
        leave = toward(words, &lead.to_string(), true),
    ))
}

/// `c` as a regex outside a class.
fn escaped(c: char) -> String {
    if r"\.^$|?*+()[]{}".contains(c) {
        format!("\\{c}")
    } else {
        c.to_string()
    }
}

/// What may follow `prefix` toward one of `words` without ending it, as a regex; with `leave`,
/// followed by a character that leads toward none.
fn toward(words: &[&str], prefix: &str, leave: bool) -> String {
    let next: BTreeSet<char> = words
        .iter()
        .filter_map(|word| word.strip_prefix(prefix)?.chars().next())
        .collect();
    let lead = prefix.chars().next().unwrap_or('<');
    let mut branches: Vec<String> = next
        .iter()
        .filter_map(|&c| {
            let longer = format!("{prefix}{c}");
            (!words.contains(&longer.as_str()))
                .then(|| format!("{}{}", escaped(c), toward(words, &longer, leave)))
        })
        .collect();
    if leave {
        branches.push(json_char(lead, &next));
    } else if branches.is_empty() {
        return String::new();
    }
    let group = format!("({})", branches.join("|"));
    if leave {
        group
    } else {
        group + "?"
    }
}

/// A character of a JSON string as written, other than `lead`, an escape and `except`.
fn json_char(lead: char, except: &BTreeSet<char>) -> String {
    let in_class = |c: char| {
        if r"\]^-[".contains(c) {
            format!("\\{c}")
        } else {
            c.to_string()
        }
    };
    let except: String = except.iter().map(|&c| in_class(c)).collect();
    format!(r#"[^"\\\x00-\x1f{}{except}]"#, in_class(lead))
}

/// The values of a string `enum` (or a string `const`).
fn string_enum(schema: &Value) -> Option<Vec<&String>> {
    if let Some(Value::String(value)) = schema.get("const") {
        return Some(vec![value]);
    }
    let values: Vec<&String> = schema
        .get("enum")?
        .as_array()?
        .iter()
        .map(|v| match v {
            Value::String(s) => Some(s),
            _ => None,
        })
        .collect::<Option<_>>()?;
    (!values.is_empty()).then_some(values)
}

/// The root definitions a property's `$ref` points into, carried into each JSON value's schema
/// that refers into them.
struct Defs<'a> {
    groups: Vec<(&'a String, &'a Value)>,
    /// Whether a value of each `$ref` followed may be a string, worked out once per tool so the
    /// work stays linear in the schema's size however its references branch and repeat; `true`
    /// while it is worked out, so a cycle may be one, as a chain followed too deep may.
    strings: RefCell<HashMap<String, bool>>,
}

impl<'a> Defs<'a> {
    fn of(schema: &'a Map<String, Value>) -> Self {
        Self {
            groups: schema
                .iter()
                .filter(|(key, _)| matches!(key.as_str(), "$defs" | "definitions"))
                .collect(),
            strings: RefCell::default(),
        }
    }

    /// `schema` with the definitions if it refers inside its document: each value is a document
    /// of its own, and the definitions would otherwise be copied into every one.
    fn attach(&self, schema: &Value) -> Value {
        let mut schema = schema.clone();
        if !has_local_reference(&schema) {
            return schema;
        }
        if let Value::Object(map) = &mut schema {
            for (key, definitions) in &self.groups {
                map.entry(key.as_str())
                    .or_insert_with(|| (*definitions).clone());
            }
        }
        schema
    }

    fn resolve(&self, reference: &str) -> Option<&'a Value> {
        let (group, name) = reference.strip_prefix("#/")?.split_once('/')?;
        let (_, definitions) = self.groups.iter().find(|(key, _)| key.as_str() == group)?;
        definitions.get(name)
    }

    /// [`admits_string`] for the target of `reference`; true for one that is not there, which has
    /// no type.
    fn admits_string(&self, reference: &str, depth: usize) -> bool {
        if let Some(&known) = self.strings.borrow().get(reference) {
            return known;
        }
        let Some(target) = self.resolve(reference) else {
            return true;
        };
        self.strings.borrow_mut().insert(reference.to_owned(), true);
        let admits = admits_string(target, self, depth);
        self.strings
            .borrow_mut()
            .insert(reference.to_owned(), admits);
        admits
    }
}

/// Whether a value of `schema` may be a string, which the parser keeps as text: a string type, a
/// string `enum` or `const` member, an `anyOf`/`oneOf` branch that may be one, a `$ref` or an
/// `allOf` of one schema (as Pydantic wraps a described enum) to one that may be, or no type at
/// all.
fn admits_string(schema: &Value, defs: &Defs, depth: usize) -> bool {
    let Value::Object(map) = schema else {
        return true;
    };
    if depth > MAX_REF_DEPTH {
        return true;
    }
    let is_string = |v: &Value| v.as_str() == Some("string");
    match map.get("type") {
        Some(kind) if is_string(kind) => return true,
        Some(Value::Array(kinds)) if kinds.iter().any(is_string) => return true,
        _ => {}
    }
    if map.get("const").is_some_and(Value::is_string)
        || map
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(Value::is_string))
    {
        return true;
    }
    let mut branches = ["anyOf", "oneOf"]
        .iter()
        .filter_map(|key| map.get(*key)?.as_array())
        .flatten();
    if branches.any(|branch| admits_string(branch, defs, depth + 1)) {
        return true;
    }
    if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
        return defs.admits_string(reference, depth + 1);
    }
    if let Some(Value::Array(all)) = map.get("allOf") {
        if let [only] = all.as_slice() {
            return admits_string(only, defs, depth + 1);
        }
    }
    let typed = ["type", "const", "enum", "anyOf", "oneOf", "allOf", "not"];
    !typed.iter().any(|key| map.contains_key(*key))
}

// The lengths of the tags' JSON text, worked out without building them
// (`AddressedTags::calls_len` and its siblings). Each `*_len` function follows the builder it is
// named after, over lengths: the fixed pieces are the builders' own, built once per table
// ([`Pieces`]), and a JSON value's schema is walked as `JsonStrings` would rewrite it, without
// rewriting it.

/// The length of `value` as JSON text, counted as it is written out.
fn json_len<T: serde::Serialize + ?Sized>(value: &T) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    match serde_json::to_writer(&mut count, value) {
        Ok(()) => count.0,
        Err(_) => usize::MAX,
    }
}

/// The lengths of the pieces the tags are built of, each the length of the piece around a
/// placeholder less the placeholder's, worked out once per table.
#[derive(Clone, Debug, Default)]
struct Pieces {
    /// `sequence` and `one_of` of no elements.
    sequence: usize,
    one_of: usize,
    optional: usize,
    star: usize,
    payload: usize,
    json_value: usize,
    /// A parameter with an empty name, first and after another.
    parameter: usize,
    parameter_after: usize,
    thought: usize,
    separator: Vec<usize>,
    last: Vec<usize>,
    /// The `\n` after a parameter written before another.
    newline: usize,
    /// A `const_string` of the empty string, and a text value.
    const_string: usize,
    raw_text: usize,
    /// The pattern as a JSON string.
    pattern: usize,
    /// [`NameLens`] of [`STRING_NAME`] and [`ANY_NAME`], which most documents give the
    /// definitions.
    names: NameLens,
}

impl Pieces {
    fn of(tags: &AddressedTags) -> Self {
        let hole = Grammar::ConstString(String::new());
        let hole_len = json_len(&hole.to_json());
        let around = |piece: Grammar| json_len(&piece.to_json()) - hole_len;
        let lens = |elements: &[Grammar]| -> Vec<usize> {
            elements.iter().map(|e| json_len(&e.to_json())).collect()
        };
        Self {
            sequence: json_len(&Grammar::Sequence(Vec::new()).to_json()),
            one_of: json_len(&Grammar::Or(Vec::new()).to_json()),
            optional: around(optional(hole.clone())),
            star: around(star(hole.clone())),
            payload: json_len(&hole.payload()) - hole_len,
            json_value: json_len(
                &Grammar::JsonSchema {
                    schema: Value::Null,
                    style: None,
                }
                .to_json(),
            ) - 4,
            parameter: around(tags.parameter("", hole.clone(), false)),
            parameter_after: around(tags.parameter("", hole.clone(), true)),
            thought: tags.thought().map_or(0, |t| json_len(&t.to_json())),
            separator: lens(&tags.separator),
            last: lens(&tags.last),
            newline: json_len(&const_string("\n").to_json()),
            const_string: hole_len,
            raw_text: json_len(&tags.raw_text().to_json()),
            pattern: json_len(&*tags.pattern),
            names: NameLens::of(STRING_NAME, ANY_NAME, &tags.pattern),
        }
    }
}

/// The length of a sequence or `or` (`empty` long with no elements) of `count` elements of
/// `total` bytes together: one element stands as it is, more are listed with a comma between each
/// two.
fn listed_len(empty: usize, count: usize, total: usize) -> usize {
    match count {
        0 => empty,
        1 => total,
        _ => empty.saturating_add(total).saturating_add(count - 1),
    }
}

impl AddressedTags {
    /// [`Self::call_run`]'s elements' lengths.
    fn call_run_len(&self, tools: &[Tool]) -> Vec<usize> {
        let pieces = &self.pieces;
        let any_call = self.any_call_len(tools);
        let next = pieces
            .separator
            .iter()
            .fold(any_call, |sum, &len| sum.saturating_add(len));
        let next = listed_len(pieces.sequence, pieces.separator.len() + 1, next);
        let mut run = vec![any_call, pieces.star.saturating_add(next)];
        run.extend(pieces.last.iter().copied());
        run
    }

    fn any_call_len(&self, tools: &[Tool]) -> usize {
        let total = tools.iter().fold(0, |sum: usize, tool| {
            sum.saturating_add(self.call_len(tool))
        });
        listed_len(self.pieces.one_of, tools.len(), total)
    }

    fn call_len(&self, tool: &Tool) -> usize {
        let opening = self.call_opening(&tool.function.name);
        let empty = Map::new();
        let schema = tool.function.parameters.as_object().unwrap_or(&empty);
        let parameters = self.parameters_len(schema);
        let total = opening
            .iter()
            .fold(parameters.unwrap_or(0), |sum, element| {
                sum.saturating_add(json_len(&element.to_json()))
            });
        let count = opening.len() + usize::from(parameters.is_some());
        listed_len(self.pieces.sequence, count, total)
    }

    /// [`Self::parameters`]' length: each value's is worked out once, and each parameter counted
    /// as many times as the body holds it.
    fn parameters_len(&self, schema: &Map<String, Value>) -> Option<usize> {
        let pieces = &self.pieces;
        let required = required_names(schema);
        let defs = Defs::of(schema);
        let groups = RootGroups::of(&defs, self);
        // Each parameter's length written first, before another and after another
        // (`Parameter`): its name is written escaped, without the quotes of a JSON string.
        let params: Vec<ParameterLen> = properties(schema)
            .map(|(key, schema)| {
                let value = self.value_len(schema, &defs, &groups);
                let required = required.contains(key.as_str());
                let value = value.saturating_add(json_len(key.as_str()) - 2);
                let first = pieces.parameter.saturating_add(value);
                let before = listed_len(pieces.sequence, 2, first.saturating_add(pieces.newline));
                let after = pieces.parameter_after.saturating_add(value);
                let after = if required {
                    after
                } else {
                    pieces.optional.saturating_add(after)
                };
                ParameterLen {
                    first,
                    before: pieces.optional.saturating_add(before),
                    after,
                    required,
                }
            })
            .collect();
        if params.is_empty() {
            return None;
        }
        let Some(first) = params.iter().position(|p| p.required) else {
            return Some(
                pieces
                    .optional
                    .saturating_add(one_or_more_len(&params, pieces)),
            );
        };
        let total = params[..first]
            .iter()
            .map(|p| p.before)
            .chain(std::iter::once(params[first].first))
            .chain(params[first + 1..].iter().map(|p| p.after))
            .fold(0, usize::saturating_add);
        Some(listed_len(pieces.sequence, params.len(), total))
    }

    /// [`Self::value`]'s length; a string enum's is counted without building it.
    fn value_len(&self, schema: &Value, defs: &Defs, groups: &RootGroups) -> usize {
        let pieces = &self.pieces;
        match raw(schema, defs) {
            Some(Raw::Enum(values)) => {
                let total = values.iter().fold(0, |sum: usize, value| {
                    let text = json_len(value.as_str()).saturating_sub(2);
                    sum.saturating_add(pieces.const_string.saturating_add(text))
                });
                listed_len(pieces.one_of, values.len(), total)
            }
            Some(Raw::Text) => pieces.raw_text,
            None => pieces
                .json_value
                .saturating_add(self.document_len(schema, groups)),
        }
    }

    fn strings(&self, own: Option<&Map<String, Value>>) -> JsonStrings {
        JsonStrings::new(own, &self.pattern, Some(self.pieces.names))
    }

    /// The length of the JSON text of `JsonStrings::rewrite(&defs.attach(schema))` for the tool's
    /// `defs`, worked out without copying or rewriting anything: the root groups the document
    /// carries come from `groups`, each reference in them longer by as much as the document's
    /// names for the two definitions are.
    fn document_len(&self, schema: &Value, groups: &RootGroups) -> usize {
        let Value::Object(map) = schema else {
            return self.rewrite_len(schema);
        };
        // The document's groups, its own or the root's, as `attach` gives them.
        let attach = has_local_reference(schema);
        let document: Vec<(&str, Group)> = ["$defs", "definitions"]
            .into_iter()
            .filter_map(|key| match map.get(key) {
                Some(own) => Some((key, Group::Own(own))),
                None if attach => groups.get(key).map(|root| (key, Group::Root(root))),
                None => None,
            })
            .collect();
        let own = document
            .iter()
            .find(|(key, _)| *key == "$defs")
            .map(|(_, group)| group.value());
        if own.is_some_and(|own| !own.is_object()) {
            // `rewrite` leaves the document as it is.
            let mut out = ObjectLen::default();
            for (key, value) in map {
                out.add(key, json_len(value));
            }
            for (key, group) in &document {
                if let Group::Root(root) = group {
                    out.add(key, root.raw_len);
                }
            }
            return out.len();
        }
        let mut strings = match document.iter().find(|(key, _)| *key == "$defs") {
            Some((_, Group::Root(_))) => groups.named.clone(),
            _ => self.strings(own.and_then(Value::as_object)),
        };
        let mut out = strings.object_len(map, true, &self.pieces);
        let mut own_defs = None;
        for (key, group) in document {
            let len = match group {
                Group::Own(own) => strings.member_len(key, own, &self.pieces),
                Group::Root(root) => {
                    strings.string_refs = strings.string_refs.saturating_add(root.string_refs);
                    strings.any_refs = strings.any_refs.saturating_add(root.any_refs);
                    let longer = |refs: usize, name: usize, default: usize| {
                        refs.saturating_mul(name.saturating_sub(default))
                    };
                    root.len
                        .saturating_add(longer(
                            root.string_refs,
                            strings.string.len(),
                            groups.string_name,
                        ))
                        .saturating_add(longer(root.any_refs, strings.any.len(), groups.any_name))
                }
            };
            if key == "$defs" {
                own_defs = Some((len, group.value()));
            } else {
                out.add(key, len);
            }
        }
        out.add_defs(own_defs, &strings);
        out.len()
    }

    /// The length of `JsonStrings::rewrite(schema)`'s JSON text, worked out without rewriting
    /// anything.
    fn rewrite_len(&self, schema: &Value) -> usize {
        let empty = Map::new();
        let map = match schema {
            Value::Object(map) => map,
            Value::Bool(true) => &empty,
            _ => return json_len(schema),
        };
        let own = map.get("$defs");
        if own.is_some_and(|own| !own.is_object()) {
            return json_len(schema);
        }
        let mut strings = self.strings(own.and_then(Value::as_object));
        let mut out = strings.object_len(map, true, &self.pieces);
        let mut own_defs = None;
        for key in ["$defs", "definitions"] {
            if let Some(group) = map.get(key) {
                let len = strings.member_len(key, group, &self.pieces);
                if key == "$defs" {
                    own_defs = Some((len, group));
                } else {
                    out.add(key, len);
                }
            }
        }
        out.add_defs(own_defs, &strings);
        out.len()
    }
}

/// A parameter's lengths as [`Parameter`] writes it.
struct ParameterLen {
    first: usize,
    before: usize,
    after: usize,
    required: bool,
}

/// [`one_or_more`]'s length.
fn one_or_more_len(params: &[ParameterLen], pieces: &Pieces) -> usize {
    if let [only] = params {
        return only.first;
    }
    let (head, tail) = params.split_at(params.len().div_ceil(2));
    let with_head = tail.iter().fold(one_or_more_len(head, pieces), |sum, p| {
        sum.saturating_add(p.after)
    });
    let with_head = listed_len(pieces.sequence, 1 + tail.len(), with_head);
    listed_len(
        pieces.one_of,
        2,
        with_head.saturating_add(one_or_more_len(tail, pieces)),
    )
}

/// A tool's root `$defs` and `definitions` as the JSON values that carry them ([`Defs::attach`])
/// have them rewritten: each worked out once, with the names a document without `$defs` gives
/// the two definitions.
struct RootGroups<'a> {
    groups: Vec<RootGroup<'a>>,
    string_name: usize,
    any_name: usize,
    /// The names a document that carries the root `$defs` gives the two definitions, apart from
    /// the root's own: worked out once for the tool, not again for each value, as the root may
    /// hold many names.
    named: JsonStrings,
}

struct RootGroup<'a> {
    key: &'a str,
    value: &'a Value,
    /// Its length as given, and as rewritten.
    raw_len: usize,
    len: usize,
    /// The references to the two definitions the rewrite writes.
    string_refs: usize,
    any_refs: usize,
}

impl<'a> RootGroups<'a> {
    fn of(defs: &Defs<'a>, tags: &AddressedTags) -> Self {
        let names = tags.strings(None);
        let groups = defs
            .groups
            .iter()
            .map(|&(key, value)| {
                let mut strings = tags.strings(None);
                RootGroup {
                    key,
                    value,
                    raw_len: json_len(value),
                    len: strings.member_len(key, value, &tags.pieces),
                    string_refs: strings.string_refs,
                    any_refs: strings.any_refs,
                }
            })
            .collect();
        let own = defs
            .groups
            .iter()
            .find(|(key, _)| key.as_str() == "$defs")
            .and_then(|(_, own)| own.as_object());
        Self {
            groups,
            string_name: names.string.len(),
            any_name: names.any.len(),
            named: tags.strings(own),
        }
    }

    fn get(&self, key: &str) -> Option<&RootGroup<'a>> {
        self.groups.iter().find(|group| group.key == key)
    }
}

/// A `$defs` or `definitions` group of a JSON value's schema: the schema's own, or the tool's root
/// one it carries.
enum Group<'a> {
    Own(&'a Value),
    Root(&'a RootGroup<'a>),
}

impl<'a> Group<'a> {
    fn value(&self) -> &'a Value {
        match self {
            Group::Own(value) => value,
            Group::Root(root) => root.value,
        }
    }
}

/// A JSON object's text length, from its members'.
#[derive(Default)]
struct ObjectLen {
    members: usize,
    inner: usize,
}

impl ObjectLen {
    /// An object of `members` members whose text takes `len` bytes.
    fn of(len: usize, members: usize) -> Self {
        Self {
            members,
            inner: len.saturating_sub(2 + members.saturating_sub(1)),
        }
    }

    fn add(&mut self, key: &str, len: usize) {
        self.add_len(json_len(key), len);
    }

    /// A member whose key, as a JSON string, takes `key` bytes.
    fn add_len(&mut self, key: usize, len: usize) {
        self.add_member((key + 1).saturating_add(len));
    }

    /// A member whose key, colon and value take `len` bytes.
    fn add_member(&mut self, len: usize) {
        self.members += 1;
        self.inner = self.inner.saturating_add(len);
    }

    /// Two braces and a comma between each two members.
    fn len(&self) -> usize {
        self.inner
            .saturating_add(2 + self.members.saturating_sub(1))
    }

    /// A document's `$defs` (`own`: its length rewritten, and as given) with the definitions
    /// `strings` adds to it.
    fn add_defs(&mut self, own: Option<(usize, &Value)>, strings: &JsonStrings) {
        let added = strings.added_lens();
        if own.is_none() && added.is_empty() {
            return;
        }
        let mut defs = own.map_or_else(ObjectLen::default, |(len, value)| {
            ObjectLen::of(len, value.as_object().map_or(0, Map::len))
        });
        added.into_iter().for_each(|member| defs.add_member(member));
        self.add("$defs", defs.len());
    }
}

/// References to the two definitions, written by a part of a schema.
#[derive(Clone, Copy)]
struct Refs {
    string: usize,
    any: usize,
}

impl Refs {
    fn since(self, before: Refs) -> Refs {
        Refs {
            string: self.string.saturating_sub(before.string),
            any: self.any.saturating_sub(before.any),
        }
    }
}

impl JsonStrings {
    fn refs(&self) -> Refs {
        Refs {
            string: self.string_refs,
            any: self.any_refs,
        }
    }

    /// The length of [`Self::string`]'s reference, counted as written.
    fn string_len(&mut self) -> usize {
        self.string_refs = self.string_refs.saturating_add(1);
        self.lens.string_ref
    }

    /// The length of [`Self::any`]'s reference, counted as written.
    fn any_len(&mut self) -> usize {
        self.any_refs = self.any_refs.saturating_add(1);
        self.lens.any_ref
    }

    /// The length of `node`'s text as [`Self::schema`] rewrites it, worked out without rewriting
    /// it; the references it writes are counted all the same.
    fn schema_len(&mut self, node: &Value, pieces: &Pieces) -> usize {
        match node {
            Value::Bool(true) => self.any_len(),
            Value::Object(map) => self.object_len(map, false, pieces).len(),
            _ => json_len(node),
        }
    }

    /// The length of a member of a schema as [`Self::schema`] rewrites it.
    fn member_len(&mut self, key: &str, value: &Value, pieces: &Pieces) -> usize {
        match (key, value) {
            (
                "properties" | "patternProperties" | "$defs" | "definitions",
                Value::Object(members),
            ) => {
                let mut out = ObjectLen::default();
                for (key, member) in members {
                    let len = self.schema_len(member, pieces);
                    out.add(key, len);
                }
                out.len()
            }
            ("anyOf" | "oneOf" | "allOf" | "prefixItems" | "items", Value::Array(members)) => {
                let total = members.iter().fold(0, |sum: usize, member| {
                    sum.saturating_add(self.schema_len(member, pieces))
                });
                total.saturating_add(2 + members.len().saturating_sub(1))
            }
            (
                "items" | "additionalProperties" | "unevaluatedProperties" | "unevaluatedItems",
                _,
            ) => self.schema_len(value, pieces),
            _ => json_len(value),
        }
    }

    /// [`Self::schema_len`] of an object, as an object; at a document's root without its `$defs`
    /// and `definitions`, which `rewrite` takes out first.
    fn object_len(&mut self, map: &Map<String, Value>, root: bool, pieces: &Pieces) -> ObjectLen {
        let kept = |key: &str| {
            !(root && matches!(key, "$defs" | "definitions")) && !ANNOTATIONS.contains(&key)
        };
        let get = |key: &str| map.get(key).filter(|_| kept(key));
        // Each member's length, and the references it writes, in case the rewrite writes it
        // twice.
        let members: Vec<(&String, usize, Refs)> = map
            .iter()
            .filter(|(key, _)| kept(key))
            .map(|(key, value)| {
                let before = self.refs();
                let len = self.member_len(key, value, pieces);
                (key, len, self.refs().since(before))
            })
            .collect();
        let reference = |len: usize| ObjectLen::of(len, 1);
        let several = get("allOf")
            .and_then(Value::as_array)
            .is_some_and(|all| all.len() > 1);
        if several || !TYPED.iter().any(|key| get(key).is_some()) {
            return reference(self.any_len());
        }
        let mut out = ObjectLen::default();
        if READ_BEFORE_TYPE.iter().any(|key| get(key).is_some()) {
            for &(key, len, _) in &members {
                out.add(key, len);
            }
            return out;
        }
        let types: Vec<&str> = match get("type") {
            Some(Value::String(kind)) => vec![kind.as_str()],
            Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        let has = |keys: &[&str]| keys.iter().any(|key| get(key).is_some());
        let string = types.contains(&"string") && !has(&OWN_STRING_RULE);
        let object = types.contains(&"object")
            || (types.is_empty()
                && has(&[
                    "properties",
                    "additionalProperties",
                    "unevaluatedProperties",
                ]));
        let array = types.contains(&"array")
            || (types.is_empty() && has(&["items", "prefixItems", "unevaluatedItems"]));
        let free_array = array && !has(&["items", "prefixItems", "unevaluatedItems"]);
        // `open_object`: the other keys of an open object, and how they are named
        // (`named_others`); `None` where it leaves the object as it is.
        const OTHERS: [&str; 2] = ["additionalProperties", "unevaluatedProperties"];
        let open = object
            && !OTHERS
                .iter()
                .any(|key| get(key) == Some(&Value::Bool(false)))
            && get("patternProperties").is_none_or(Value::is_object);
        let given = OTHERS.iter().find_map(|other| {
            members
                .iter()
                .find(|(key, _, _)| key.as_str() == *other)
                .map(|&(_, len, refs)| (len, refs))
        });
        let others = match (open, given) {
            (false, _) => None,
            (true, Some(given)) => Some(given),
            (true, None) if has(&["properties", "patternProperties"]) => None,
            (true, None) => Some((self.any_len(), Refs { string: 0, any: 1 })),
        };
        let named = others.map(|others| (others, named_others(map)));
        // The pattern's entry in `patternProperties`. Were the pattern a key there already, it
        // would replace it; counted twice, the length is only longer.
        let patterns = |len: usize, members: usize, others: usize| {
            let mut patterns = ObjectLen::of(len, members);
            patterns.add_len(pieces.pattern, others);
            patterns.len()
        };
        // A union type with a string keeps its other types.
        let other_types = other_types(get("type"));
        let union = string && !other_types.is_empty();
        for &(key, len, _) in &members {
            match (key.as_str(), named) {
                // Taken out, and given again below where named.
                (other, Some(_)) if OTHERS.contains(&other) => {}
                ("properties", Some((_, Some(false)))) | ("propertyNames", Some((_, Some(_)))) => {}
                ("patternProperties", Some(((others, _), None))) => {
                    let count = get(key).and_then(Value::as_object).map_or(0, Map::len);
                    out.add(key, patterns(len, count, others));
                }
                ("type", _) if union => out.add(key, json_len(&other_types)),
                _ => out.add(key, len),
            }
        }
        if let Some(((others, refs), named)) = named {
            match named {
                Some(declared) => {
                    if !declared {
                        // `{"": others}`, a copy of the other keys' value.
                        let mut declared = ObjectLen::default();
                        declared.add("", others);
                        out.add("properties", declared.len());
                        self.string_refs = self.string_refs.saturating_add(refs.string);
                        self.any_refs = self.any_refs.saturating_add(refs.any);
                    }
                    out.add("additionalProperties", others);
                    let names = self.string_len();
                    out.add("propertyNames", names);
                }
                None if get("patternProperties").is_none() => {
                    out.add("patternProperties", patterns(2, 0, others));
                }
                None => {}
            }
            if get("type").is_none() {
                out.add("type", json_len("object"));
            }
        }
        if free_array {
            out.add("items", self.any_len());
        }
        if !string {
            return out;
        }
        let string = self.string_len();
        if !union {
            return reference(string);
        }
        // `{"anyOf": [string, rest]}`
        let any_of = string.saturating_add(out.len()).saturating_add(3);
        let mut wrapped = ObjectLen::default();
        wrapped.add("anyOf", any_of);
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use regex::Regex;

    use super::*;

    const WORDS: [&str; 4] = ["<|end|>", "<|call|>", "<|return|>", "</parameter>"];

    fn pattern() -> String {
        excluding(&WORDS).expect("one lead")
    }

    /// Whether `text` may stand between the quotes of a JSON string.
    fn json_text(text: &str) -> bool {
        Regex::new(r#"^([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u[0-9a-fA-F]{4}))*$"#)
            .expect("a regex")
            .is_match(text)
    }

    #[test]
    fn the_string_pattern_leaves_out_exactly_the_delimiters() {
        let pattern = Regex::new(&format!("^({})$", pattern())).expect("a regex");
        let expected = |text: &str| json_text(text) && !WORDS.iter().any(|w| text.contains(w));
        let check = |text: &str| assert_eq!(pattern.is_match(text), expected(text), "{text:?}");

        // Every text of up to four characters from the delimiters' own, and texts of their
        // pieces.
        let alphabet = [
            '<', '|', 'e', 'n', 'd', 'c', 'a', 'l', 'r', 't', 'u', '>', '/', 'x',
        ];
        let mut texts = vec![String::new()];
        for _ in 0..4 {
            texts = texts
                .iter()
                .flat_map(|t| alphabet.iter().map(move |c| format!("{t}{c}")))
                .collect();
            texts.iter().for_each(|t| check(t));
        }
        let pieces = [
            "<",
            "|",
            "e",
            "n",
            "d",
            ">",
            "c",
            "l",
            "x",
            "<|",
            "<|e",
            "<|end",
            "<|end|",
            "|>",
            "<|c",
            "<|cal",
            "<|call|",
            "<|r",
            "<|retur",
            "<|return|",
            "</",
            "</par",
            "</param",
            "</parameter",
            "r>",
            "\\n",
            "\"",
            "\\",
            "é",
        ];
        let mut seed: u64 = 1;
        for _ in 0..20_000 {
            let mut text = String::new();
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            for _ in 0..(seed >> 61) {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                text.push_str(pieces[(seed >> 33) as usize % pieces.len()]);
            }
            check(&text);
        }
        // Each delimiter, each one cut short, broken off or doubled.
        for word in WORDS {
            for (at, _) in word.char_indices().skip(1) {
                let (head, tail) = word.split_at(at);
                for text in [
                    format!("{head}x{tail}"),
                    format!("{head}{word}"),
                    format!("{head}<{tail}"),
                    format!("{head}\\n{tail}"),
                    format!("{head}\"{tail}"),
                    head.to_owned(),
                    format!("a{head}"),
                    format!("{head}{head}"),
                ] {
                    check(&text);
                }
            }
            for text in [
                word.to_owned(),
                format!("a{word}b"),
                format!("<{word}"),
                format!("{word}{word}"),
            ] {
                assert!(!pattern.is_match(&text), "{text:?}");
            }
        }
        for text in [
            "</div> a < b || c <|en <|call|",
            "\\u003c|call|>",
            "<\\/parameter>",
            "é ✓ \\\" \\\\",
        ] {
            assert!(pattern.is_match(text), "{text:?}");
        }
        for text in ["\"", "\\", "\\x", "a\nb", "\u{1}"] {
            assert!(!pattern.is_match(text), "{text:?}");
        }
    }

    /// Another lead works the same way; words without one lead, or that hold it twice, are
    /// refused.
    #[test]
    fn the_pattern_takes_any_one_lead() {
        let words = ["[END]", "[CALL]"];
        let pattern =
            Regex::new(&format!("^({})$", excluding(&words).expect("one lead"))).expect("a regex");
        for text in ["[END", "[ENDx]", "a [CAL] b", "[[", "x]"] {
            assert!(pattern.is_match(text), "{text:?}");
        }
        for text in ["[END]", "a[CALL]", "[[END]"] {
            assert!(!pattern.is_match(text), "{text:?}");
        }
        assert!(excluding(&["<a>", "[b]"]).is_err());
        assert!(excluding(&["<a<b>"]).is_err());
        assert!(excluding(&["\"a"]).is_err());
        assert!(excluding(&[]).is_err());
    }

    #[test]
    fn json_strings_take_the_pattern() {
        let pattern = pattern();
        let shared: Rc<str> = pattern.as_str().into();
        let string = json!({"$ref": "#/$defs/symphony_string"});
        let any = json!({"$ref": "#/$defs/symphony_json"});
        let string_def = json!({"type": "string", "pattern": pattern});
        // An object whose keys are strings of the definition, taking `value`.
        let named = |value: &Value| {
            json!({"type": "object", "properties": {"": value}, "additionalProperties": value,
                   "propertyNames": string})
        };
        let any_def = json!({"anyOf": [
            string, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
            {"type": "array", "items": any}, named(&any),
        ]});
        let rewrite = |schema: &Value| JsonStrings::rewrite(schema, &shared);

        // Nothing to hold: unchanged, no definitions added.
        for schema in [
            json!({"type": "integer"}),
            json!({"type": "object", "properties": {"a": {"type": "integer"}},
                   "additionalProperties": false}),
            json!({"type": "array", "prefixItems": [{"type": "boolean"}]}),
            json!({"type": "object", "additionalProperties": {"type": "integer"},
                   "unevaluatedProperties": false}),
            json!(false),
        ] {
            assert_eq!(rewrite(&schema), schema);
        }

        // Strings refer to the definition, a union type through an `anyOf` with its other
        // types; one with its own rule keeps it, and one xgrammar reads by another keyword first
        // is that keyword's.
        let schema = json!({"type": "object", "properties": {
            "plain": {"type": "string", "description": "a city"},
            "nullable": {"type": "string", "nullable": true, "x-kind": "city"},
            "union": {"type": ["string", "null"], "description": "or none"},
            "both": {"type": ["string", "object", "string"]},
            "short": {"type": "string", "maxLength": 3},
            "dated": {"type": "string", "format": "date"},
            "fixed": {"type": "string", "enum": ["x"]},
            "either": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "first": {"oneOf": [{"type": "integer"}], "type": "string"},
            "list": {"type": "array", "items": {"type": "string"}},
        }, "required": ["plain"]});
        let expected = json!({"type": "object", "properties": {
            "plain": string,
            "nullable": string,
            "union": {"anyOf": [string, {"type": ["null"]}]},
            "both": {"anyOf": [string, {"type": ["object"], "properties": {"": any},
                                        "additionalProperties": any,
                                        "propertyNames": string}]},
            "short": {"type": "string", "maxLength": 3},
            "dated": {"type": "string", "format": "date"},
            "fixed": {"type": "string", "enum": ["x"]},
            "either": {"anyOf": [string, {"type": "null"}]},
            "first": {"oneOf": [{"type": "integer"}], "type": "string"},
            "list": {"type": "array", "items": string},
        }, "required": ["plain"], "$defs": {
            "symphony_string": string_def, "symphony_json": any_def}});
        assert_eq!(rewrite(&schema), expected);

        // Any JSON, and the other keys of an open object.
        let defs = json!({"symphony_string": string_def, "symphony_json": any_def});
        let int = json!({"type": "integer"});
        for (schema, expected) in [
            (json!(true), json!({"$ref": "#/$defs/symphony_json"})),
            (json!({"description": "anything"}), any.clone()),
            (json!({"type": "object"}), named(&any)),
            (
                json!({"type": "array"}),
                json!({"type": "array", "items": any}),
            ),
            (
                json!({"type": "object", "properties": {"a": {"type": "integer"}},
                       "additionalProperties": true, "required": ["a"]}),
                json!({"type": "object", "properties": {"a": {"type": "integer"}},
                       "additionalProperties": any, "propertyNames": string,
                       "required": ["a"]}),
            ),
            (
                json!({"type": "object", "properties": {}, "additionalProperties": true}),
                named(&any),
            ),
            (
                json!({"additionalProperties": {"type": "integer"}}),
                json!({"properties": {"": int}, "additionalProperties": int,
                       "propertyNames": string, "type": "object"}),
            ),
            // Its own patterns, or keys it requires without declaring them: the pattern names
            // the other keys.
            (
                json!({"type": "object", "patternProperties": {"^x": int},
                       "additionalProperties": true}),
                json!({"type": "object", "patternProperties": {"^x": int, &pattern: any}}),
            ),
            (
                json!({"type": "object", "required": ["k"]}),
                json!({"type": "object", "required": ["k"],
                       "patternProperties": {&pattern: any}}),
            ),
            (
                json!({"allOf": [{"type": "object"}, {"description": "x"}]}),
                any.clone(),
            ),
        ] {
            let mut expected = expected;
            let has_any = expected.to_string().contains("symphony_json");
            expected["$defs"] = if has_any {
                defs.clone()
            } else {
                json!({"symphony_string": string_def})
            };
            assert_eq!(rewrite(&schema), expected, "{schema}");
        }

        // The root's definitions are rewritten and kept, beside a name that is taken.
        let schema = json!({"$ref": "#/$defs/Name", "$defs": {
            "Name": {"type": "string"}, "symphony_string": {"type": "integer"},
        }});
        let expected = json!({"$ref": "#/$defs/Name", "$defs": {
            "Name": {"$ref": "#/$defs/symphony_string_"},
            "symphony_string": {"type": "integer"},
            "symphony_string_": string_def,
        }});
        assert_eq!(rewrite(&schema), expected);
    }

    /// The pattern is written once, in the string definition, wherever a document holds strings
    /// or keys; only an object that names its other keys with patterns of its own, or declares no
    /// keys but requires or counts some, holds it as a key of its own.
    #[test]
    fn the_pattern_is_written_once() {
        let shared: Rc<str> = pattern().into();
        let written = serde_json::to_string(&*shared).expect("a string");
        let copies = |member: &Value| {
            let properties: Map<String, Value> =
                (0..50).map(|i| (format!("f{i}"), member.clone())).collect();
            let schema = json!({"type": "object", "properties": properties});
            JsonStrings::rewrite(&schema, &shared)
                .to_string()
                .matches(&written)
                .count()
        };
        for member in [
            json!({"type": ["string", "null"], "description": "or none"}),
            json!({"type": "string", "x-nullable": true}),
            json!({"type": "object"}),
            json!({"type": "object", "properties": {"a": {}}, "additionalProperties": true}),
            json!({"additionalProperties": {"type": "integer"}}),
            json!({"type": ["object", "string"], "properties": {"a": {"type": "string"}}}),
            json!(true),
        ] {
            assert_eq!(copies(&member), 1, "{member}");
        }
        for member in [
            json!({"type": "object", "patternProperties": {"^a": {}}, "additionalProperties": {}}),
            json!({"type": "object", "required": ["k"]}),
            json!({"additionalProperties": {"type": "integer"}, "minProperties": 1}),
        ] {
            // One for each object, and the definition where a value refers to it.
            assert!((50..=51).contains(&copies(&member)), "{member}");
        }
    }

    /// Markers split into their tokens, the text between them run together.
    #[test]
    fn markers_split_into_their_tokens() {
        let tokens = ["<|a|>".to_string(), "<|ab|>".to_string()];
        assert_eq!(
            split("x<|a|><|ab|>y z<|a", &tokens),
            vec![
                Grammar::ConstString("x".to_string()),
                Grammar::Token("<|a|>".to_string()),
                Grammar::Token("<|ab|>".to_string()),
                Grammar::ConstString("y z<|a".to_string()),
            ]
        );
        assert_eq!(split("", &tokens), Vec::<Grammar>::new());
        assert_eq!(
            first_piece("<|a|>rest", &tokens),
            (Bound::Token("<|a|>".to_string()), "rest")
        );
        assert_eq!(
            first_piece("text<|a|>", &tokens),
            (Bound::Text("text".to_string()), "<|a|>")
        );
    }
}
