//! A response template compiled into a symphony table.
//!
//! symphony reads a model's output with a table ([`Format`]): literal markers, the states the
//! text between them falls into, and the rows that move between states. It never runs a regular
//! expression while it streams. [`compile`] reads a template's patterns through the reader's tree
//! of each one and lowers the template's constructs into such a table, so that the table gives
//! what transformers' whole-text reading of the same output gives (`parse_response(output,
//! prefix=prompt, tools=tools)`), at every chunking, apart from the differences a stream cannot
//! avoid and the API's own policies (listed in the README).
//!
//! What a template becomes:
//!
//! - **Anchors.** `start_anchor`, one literal or several, and a `start_anchor_pattern` that is a
//!   finite set of literals, become the table's turn anchors, in the order the template's pattern
//!   tries them: the prompt's replay starts after the last one, as transformers truncates the
//!   prompt past it.
//! - **Fields.** The field with no open is the text outside every other field: `content` as text
//!   is a content state, trimmed unless `strip: false`; any other name, or no such field, is a
//!   framing state, whose text the template discards (transformers' null sink). An explicit
//!   `thinking` or `reasoning_content` is a reasoning state, `content` a content state (each
//!   trimmed unless `strip: false`), and a field of another name a framing state. Each open of a
//!   field is a row into its state from the outside, each close a row back.
//! - **Delimiters.** A delimiter's marker is a literal: an open's first literal after optional
//!   whitespace, a close's last literal before it. Whitespace the delimiter takes beside its
//!   marker is text of the state next to it, which is trimmed, so trimming drops what the pattern
//!   takes; it must not be able to take the first character of another marker that moves that
//!   state. Alternatives (`(?:A|B)`, a literal list) are markers of their own.
//! - **Markers that hold markers.** A close made of markers (`<|end|>` then the anchor's text)
//!   moves where those markers move in turn, a marker that leads into a framing state dropping the
//!   rest. A marker that would hide a marker its state moves on is refused.
//! - **Stops.** The texts of the tokens the model stops on are not fed to a stream. A close of the
//!   outside text that is one adds no row: the stream's stop ends the message. Every marker that
//!   is one is named a stop token of the table.
//! - **Values.** The table reads its calls' values with [`TRANSFORMERS`], the crate's own port of
//!   transformers' typing.
//!
//! Everything else is refused by name ([`Refusals`], each with a stable [`Construct`] code), and
//! the compile goes on after a refused construct, so that one compile lists every construct a
//! template uses that no table reads. Tool calls are refused in this version: their arguments are
//! not lowered yet ([`Construct::TaggedParameters`], [`Construct::JsonArguments`],
//! [`Construct::JsonEnvelope`]).

mod lower;
mod refusal;
mod values;

use std::fmt::Write as _;

use serde_json::Value;
use symphony::{shape::Slot, Emits, Format};

use self::{
    lower::{
        anchor_literals, literals, lower, pattern, split_close, split_open, Branches, Captures,
        Item,
    },
    refusal::Refused,
};
pub use self::{
    refusal::{Construct, Refusal, Refusals},
    values::{Transformers, TRANSFORMERS},
};
use crate::{
    content_parsers::{placeholder, ContentParser},
    py,
    pyre::Node,
    response_templates::{Delimiter, ResponseTemplate, ResponseTemplateField, Template},
};

/// A table compiled from a template, and a one-line description of what it reads, for logs and
/// the pinned outcome lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compiled {
    pub format: Format,
    pub summary: String,
}

/// Compile `template` into a symphony table. `stops` are the texts of the tokens the model stops
/// on (its end-of-sequence and stop tokens), which the stop decoder hides from the stream: a
/// close of the outside text that is one adds no row, and every marker that is one is named a
/// stop token of the table. Refused, with every construct the compile does not lower, when the
/// template uses one.
pub fn compile(template: &ResponseTemplate, stops: &[&str]) -> Result<Compiled, Refusals> {
    let mut compiler = Compiler {
        template: &template.0,
        stops,
        refusals: Vec::new(),
        table: Table::default(),
        slack_after: Vec::new(),
        slack_before: Vec::new(),
    };
    compiler.anchors();
    compiler.outside();
    compiler.fields();
    compiler.defaults();
    if !compiler.refusals.is_empty() {
        return Err(Refusals(compiler.refusals));
    }
    compiler.table.derive();
    let mut refused = compiler.table.hidden_markers();
    refused.extend(compiler.slack_before_markers());
    if !refused.is_empty() {
        return Err(Refusals(
            refused.into_iter().map(|r| r.scoped(TEMPLATE)).collect(),
        ));
    }
    let format = compiler.table.format(stops);
    if let Err(why) = format.validate() {
        return Err(Refusals(vec![
            Refused::new(Construct::Table, why).scoped(TEMPLATE)
        ]));
    }
    let summary = compiler.table.summary(&format);
    Ok(Compiled { format, summary })
}

const TEMPLATE: &str = "response_template";

/// What a field is in a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Reasoning,
    Content,
    Calls,
    /// A field the message has no place for: its text is dropped.
    Other,
}

impl Kind {
    fn of(name: &str) -> Self {
        match name {
            "thinking" | "reasoning_content" => Self::Reasoning,
            "content" => Self::Content,
            "tool_calls" => Self::Calls,
            _ => Self::Other,
        }
    }
}

fn scope(field: &ResponseTemplateField) -> String {
    format!("Field '{}'", field.name)
}

struct Compiler<'t> {
    template: &'t Template,
    stops: &'t [&'t str],
    refusals: Vec<Refusal>,
    table: Table,
    /// Delimiters whose whitespace after the marker is text of the state they lead into: that
    /// state, the whitespace, and where it stands.
    slack_after: Vec<(usize, Vec<Item>, String)>,
    /// Delimiters whose whitespace before the marker is text of the state the marker moves: that
    /// state, the whitespace, where it stands, and the marker.
    slack_before: Vec<(usize, Vec<Item>, String, String)>,
}

impl Compiler<'_> {
    fn refuse(&mut self, scope: &str, refused: Refused) {
        self.refusals.push(refused.scoped(scope));
    }

    fn refuse_all(&mut self, scope: &str, refused: Vec<Refused>) {
        for one in refused {
            self.refuse(scope, one);
        }
    }

    /// The turn anchors: the literals of `start_anchor`, or of a `start_anchor_pattern` that is a
    /// finite set of them.
    fn anchors(&mut self) {
        let anchor = &self.template.start_anchor;
        let lowered = match branches(anchor, &Captures::default()) {
            Ok(branches) => branches,
            Err(refused) => return self.refuse_all(TEMPLATE, refused),
        };
        match anchor_literals(&lowered) {
            Ok(texts) => self.table.anchors = texts,
            Err(refused) => self.refuse(TEMPLATE, refused),
        }
    }

    /// The outside state, from the field with no open.
    fn outside(&mut self) {
        let implicit = self.template.implicit.map(|i| &self.template.fields[i]);
        let (emits, framing, trims) = match implicit {
            None => (Emits::Content, true, false),
            Some(field) => match Kind::of(&field.name) {
                Kind::Content => {
                    let ok = self.text_field(field);
                    (Emits::Content, false, ok && field.content.strips())
                }
                Kind::Other => (Emits::Content, true, false),
                Kind::Reasoning | Kind::Calls => {
                    self.refuse(
                        &scope(field),
                        Refused::new(
                            Construct::ImplicitField,
                            format!("the field with no open is {:?}", field.name),
                        ),
                    );
                    (Emits::Content, true, false)
                }
            },
        };
        self.table.states.push(State {
            name: "0:outside".to_owned(),
            field: implicit.map(|field| field.name.clone()),
            emits,
            framing,
            trims,
        });
        if let Some(field) = implicit {
            self.implicit_closes(field);
        }
    }

    /// The checks a text field passes: text content, no transform, no repeats.
    fn text_field(&mut self, field: &ResponseTemplateField) -> bool {
        let scope = scope(field);
        let before = self.refusals.len();
        if field.content.parser != ContentParser::Text {
            self.refuse(
                &scope,
                Refused::new(
                    Construct::TextFieldContent,
                    "a text field whose content is not text",
                ),
            );
        }
        if field.transform.is_some() {
            self.refuse(
                &scope,
                Refused::new(Construct::TextFieldTransform, "a transform of a text field"),
            );
        }
        if field.repeats {
            self.refuse(
                &scope,
                Refused::new(Construct::RepeatedTextField, "a text field that repeats"),
            );
        }
        self.refusals.len() == before
    }

    /// The closes of the field with no open: a close that is a stop text adds nothing, since the
    /// stream's stop ends the message; any other is a marker, read as a row back into the outside
    /// that drops it when the outside is content, and as dropped text when it is framing.
    fn implicit_closes(&mut self, field: &ResponseTemplateField) {
        let Some(close) = &field.close else {
            return;
        };
        let scope = scope(field);
        let closes = match self.closes(&scope, close, 0) {
            Some(closes) => closes,
            None => return,
        };
        for close in closes {
            if self.stops.contains(&close.as_str()) {
                continue;
            }
            let on = self.table.terminal(&close);
            if !self.table.states[0].framing {
                self.table.row(0, on, 0);
            }
        }
    }

    /// Every explicit field: its state, and its rows.
    fn fields(&mut self) {
        let mut seen: Vec<(Kind, String)> = Vec::new();
        if let Some(field) = self.template.implicit.map(|i| &self.template.fields[i]) {
            seen.push((Kind::of(&field.name), field.name.clone()));
        }
        for field in self.template.fields.iter().filter(|f| f.open.is_some()) {
            let kind = Kind::of(&field.name);
            if kind != Kind::Other {
                if let Some((_, other)) = seen.iter().find(|(k, _)| *k == kind) {
                    let other = other.clone();
                    self.refuse(
                        &scope(field),
                        Refused::new(
                            Construct::SharedField,
                            format!("{:?} and {other:?} are one field of a message", field.name),
                        ),
                    );
                }
                seen.push((kind, field.name.clone()));
            }
            match kind {
                Kind::Calls => self.calls(field),
                Kind::Reasoning | Kind::Content | Kind::Other => self.text_region(field, kind),
            }
        }
    }

    /// A field whose text is a text value (the thought, the content) or dropped: a state with a
    /// row in on each of its opens and out on each of its closes.
    fn text_region(&mut self, field: &ResponseTemplateField, kind: Kind) {
        let scope = scope(field);
        let (emits, framing, trims) = match kind {
            Kind::Reasoning | Kind::Content => {
                let ok = self.text_field(field);
                let emits = if kind == Kind::Reasoning {
                    Emits::Reasoning
                } else {
                    Emits::Content
                };
                (emits, false, ok && field.content.strips())
            }
            Kind::Calls | Kind::Other => (Emits::Content, true, false),
        };
        let state = self.table.states.len();
        self.table.states.push(State {
            name: format!("{state}:{}", field.name),
            field: Some(field.name.clone()),
            emits,
            framing,
            trims,
        });
        if let Some(open) = &field.open {
            if let Some(opens) = self.opens(&scope, open, state) {
                for marker in opens {
                    let on = self.table.terminal(&marker);
                    if let Some(to) = self.table.row(0, on, state) {
                        let other = self.table.states[to].field.clone().unwrap_or_default();
                        self.refuse(
                            &scope,
                            Refused::new(
                                Construct::SharedOpenMarker,
                                format!("the open {marker:?} also opens {other:?}"),
                            ),
                        );
                    }
                }
            }
        }
        if let Some(close) = &field.close {
            if let Some(closes) = self.closes(&scope, close, state) {
                for marker in closes {
                    let on = self.table.terminal(&marker);
                    self.table.row(state, on, 0);
                }
            }
        }
    }

    /// The markers of a text field's open into `state`: each alternative's first literal, with
    /// only whitespace beside it, which the outside before it and `state` after it trim.
    fn opens(&mut self, scope: &str, open: &Delimiter, state: usize) -> Option<Vec<String>> {
        if open
            .regex
            .as_ref()
            .is_some_and(|regex| zero_width(&regex.node))
        {
            self.refuse(
                scope,
                Refused::new(
                    Construct::ZeroWidthOpen,
                    "an open that is a lookahead as a whole",
                ),
            );
            return None;
        }
        let branches = match branches(open, &Captures::default()) {
            Ok(branches) => branches,
            Err(refused) => {
                self.refuse_all(scope, refused);
                return None;
            }
        };
        let mut markers = Vec::new();
        for branch in &branches {
            let open = match split_open(branch) {
                Ok(open) => open,
                Err(refused) => {
                    self.refuse(scope, refused);
                    continue;
                }
            };
            if !open.header.iter().all(Item::is_slack) {
                self.refuse(
                    scope,
                    Refused::new(
                        Construct::TextFieldDelimiter,
                        format!(
                            "the open {:?} is followed by more than whitespace",
                            open.marker
                        ),
                    ),
                );
            }
            if self.slack_beside(scope, &open.slack, 0, "before the open") {
                self.slack_before.push((
                    0,
                    open.slack.clone(),
                    scope.to_owned(),
                    open.marker.clone(),
                ));
            }
            if self.slack_beside(scope, &open.header, state, "after the open") {
                self.slack_after
                    .push((state, open.header.clone(), scope.to_owned()));
            }
            markers.push(open.marker);
        }
        self.alternative_order(scope, &markers);
        Some(markers)
    }

    /// The markers of a close out of `state` into the outside: each alternative's last literal,
    /// with only whitespace beside it, which `state` before it and the outside after it trim.
    fn closes(&mut self, scope: &str, close: &Delimiter, state: usize) -> Option<Vec<String>> {
        let branches = match branches(close, &Captures::default()) {
            Ok(branches) => branches,
            Err(refused) => {
                self.refuse_all(scope, refused);
                return None;
            }
        };
        let mut markers = Vec::new();
        for branch in &branches {
            let close = match split_close(branch) {
                Ok(close) => close,
                Err(refused) => {
                    self.refuse(scope, refused);
                    continue;
                }
            };
            if !close.trailer.iter().all(Item::is_slack) {
                self.refuse(
                    scope,
                    Refused::new(
                        Construct::TextFieldDelimiter,
                        format!("the close {:?} follows more than whitespace", close.marker),
                    ),
                );
            }
            if self.slack_beside(scope, &close.trailer, state, "before the close") {
                self.slack_before.push((
                    state,
                    close.trailer.clone(),
                    scope.to_owned(),
                    close.marker.clone(),
                ));
            }
            // A stream never shows what follows a stop text.
            let stop = self.stops.contains(&close.marker.as_str());
            if !stop && self.slack_beside(scope, &close.slack, 0, "after the close") {
                self.slack_after
                    .push((0, close.slack.clone(), scope.to_owned()));
            }
            markers.push(close.marker);
        }
        self.alternative_order(scope, &markers);
        Some(markers)
    }

    /// Whitespace a delimiter takes beside `state`, which must trim it or drop it. Returns
    /// whether there is any.
    fn slack_beside(&mut self, scope: &str, slack: &[Item], state: usize, place: &str) -> bool {
        let slack: Vec<&Item> = slack.iter().filter(|item| item.is_slack()).collect();
        if slack.is_empty() {
            return false;
        }
        let kept = &self.table.states[state];
        if !kept.trims && !kept.framing {
            let name = kept
                .field
                .clone()
                .unwrap_or_else(|| "the outside".to_owned());
            self.refuse(
                scope,
                Refused::new(
                    Construct::UntrimmedSlack,
                    format!("whitespace {place} is taken from {name:?}, which keeps it"),
                ),
            );
        }
        true
    }

    /// An alternative whose marker begins a later alternative's marker: the pattern takes the
    /// first, where a table takes the longest.
    fn alternative_order(&mut self, scope: &str, markers: &[String]) {
        for (at, earlier) in markers.iter().enumerate() {
            if let Some(later) = markers[at + 1..]
                .iter()
                .find(|later| later.len() > earlier.len() && later.starts_with(earlier.as_str()))
            {
                let later = later.clone();
                self.refuse(
                    scope,
                    Refused::new(
                        Construct::AlternativeOrder,
                        format!("{earlier:?} is tried before {later:?}, which it begins"),
                    ),
                );
            }
        }
    }

    /// The tool calls. Their arguments are not lowered yet: every form is refused by name, with
    /// what else the field uses that no table reads.
    fn calls(&mut self, field: &ResponseTemplateField) {
        let scope = scope(field);
        if !field.repeats {
            self.refuse(
                &scope,
                Refused::new(
                    Construct::NonRepeatingCalls,
                    "tool calls that do not repeat: one region holds every call",
                ),
            );
        }
        let named = if field.transform_each {
            self.refuse(
                &scope,
                Refused::new(Construct::TransformEach, "tool calls given as a list"),
            );
            None
        } else {
            match call_transform(field.transform.as_ref()) {
                Ok(named) => Some(named),
                Err(refused) => {
                    self.refuse(&scope, refused);
                    None
                }
            }
        };
        if let Some(named) = &named {
            let arguments = arguments(field, named);
            self.refuse(&scope, arguments);
        }
        let content = &field.content;
        if content.parser == ContentParser::Json && content.json_dialect() {
            self.refuse(
                &scope,
                Refused::new(
                    Construct::JsonDialect,
                    "JSON with unquoted keys or its own string delimiters",
                ),
            );
        }
        if content.parser == ContentParser::XmlInline && content.merges_duplicates() {
            self.refuse(
                &scope,
                Refused::new(Construct::MergeDuplicates, "merge_duplicates"),
            );
        }
        self.call_delimiters(field, named.as_ref());
    }

    /// What a call's open and close use that no table reads, and whether the open captures the
    /// transform's name.
    fn call_delimiters(&mut self, field: &ResponseTemplateField, named: Option<&Named>) {
        let scope = scope(field);
        let captures = match (named, field.open.as_ref().and_then(|d| d.regex.as_ref())) {
            (Some(Named::Group(group)), Some(regex)) => {
                let numbers: Vec<usize> = regex
                    .names
                    .iter()
                    .filter(|(name, _)| name == group)
                    .map(|(_, number)| *number)
                    .collect();
                match numbers.as_slice() {
                    [number] => Captures(vec![(*number, Slot::Name)]),
                    _ => {
                        self.refuse(
                            &scope,
                            Refused::new(
                                Construct::HeaderName,
                                format!("the name {group:?} is no group the open captures once"),
                            ),
                        );
                        Captures::default()
                    }
                }
            }
            (Some(Named::Group(group)), None) => {
                self.refuse(
                    &scope,
                    Refused::new(
                        Construct::HeaderName,
                        format!("the name {group:?} is no group of a literal open"),
                    ),
                );
                Captures::default()
            }
            _ => Captures::default(),
        };
        if let Some(open) = &field.open {
            match branches(open, &captures) {
                Ok(branches) => {
                    for branch in &branches {
                        match split_open(branch) {
                            Ok(open) => {
                                if let Err(error) = pattern(&open.header).check() {
                                    self.refuse(
                                        &scope,
                                        Refused::new(
                                            Construct::UnboundedHeaderRun,
                                            format!("the header after {:?}: {error}", open.marker),
                                        ),
                                    );
                                }
                            }
                            Err(refused) => self.refuse(&scope, refused),
                        }
                    }
                }
                Err(refused) => self.refuse_all(&scope, refused),
            }
        }
        if let Some(close) = &field.close {
            match branches(close, &Captures::default()) {
                Ok(branches) => {
                    for branch in &branches {
                        match split_close(branch) {
                            Ok(close) => {
                                if let Err(error) = pattern(&close.trailer).check() {
                                    self.refuse(
                                        &scope,
                                        Refused::new(
                                            Construct::UnboundedHeaderRun,
                                            format!(
                                                "the trailer before {:?}: {error}",
                                                close.marker
                                            ),
                                        ),
                                    );
                                }
                            }
                            Err(refused) => self.refuse(&scope, refused),
                        }
                    }
                }
                Err(refused) => self.refuse_all(&scope, refused),
            }
        }
    }

    /// `defaults` other than the role: a default with a value for the thought, the content or the
    /// calls is in transformers' message, whether the template has such a field or not, where the
    /// stream shows nothing.
    fn defaults(&mut self) {
        for (key, value) in &self.template.defaults {
            if key == "role" || is_empty(value) {
                continue;
            }
            if Kind::of(key) != Kind::Other {
                self.refuse(
                    TEMPLATE,
                    Refused::new(Construct::Default, format!("a default for {key:?}")),
                );
            }
        }
    }

    /// Whitespace a delimiter takes beside its marker, where another marker that moves the state
    /// the whitespace is read in may begin. After the marker, the pattern's greedy run would take
    /// that marker's first character, and the marker would not be read there. Before the marker,
    /// the pattern's match would start in the whitespace, before that marker, and win where a
    /// table reads the other marker first.
    fn slack_before_markers(&self) -> Vec<Refused> {
        let after = self
            .slack_after
            .iter()
            .map(|(state, slack, scope)| (*state, slack, scope, None));
        let before = self
            .slack_before
            .iter()
            .map(|(state, slack, scope, own)| (*state, slack, scope, Some(own.as_str())));
        let mut out = Vec::new();
        for (state, slack, scope, own) in after.chain(before) {
            for &(from, on, _) in &self.table.rows {
                let marker = self.table.terminals[on].as_str();
                if from != state || own == Some(marker) {
                    continue;
                }
                let Some(first) = marker.chars().next() else {
                    continue;
                };
                if slack.iter().any(|item| item.takes(first)) {
                    let place = match own {
                        Some(own) => format!("before {own:?}"),
                        None => "after a marker".to_owned(),
                    };
                    out.push(Refused::new(
                        Construct::SlackBeforeMarker,
                        format!("{scope}: whitespace {place} can take the start of {marker:?}"),
                    ));
                }
            }
        }
        out
    }
}

/// A delimiter's alternatives.
fn branches(delimiter: &Delimiter, captures: &Captures) -> Result<Branches, Vec<Refused>> {
    match &delimiter.regex {
        Some(regex) => lower(regex, captures),
        None => Ok(literals(&delimiter.texts)),
    }
}

/// Python's `_is_empty` in `finalize`: `None`, or an empty list, dict or str.
fn is_empty(value: &Value) -> bool {
    matches!(value, Value::Null)
        || matches!(value, Value::String(_) | Value::Array(_) | Value::Object(_))
            && !py::truthy(value)
}

/// Where a call's name comes from, as its transform says.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Named {
    /// A group the open captures.
    Group(String),
    /// The template's own text.
    Literal(String),
    /// The arguments' own object (`"function": "{content}"`).
    Envelope,
}

/// Why a call's arguments, written as `field`'s content, are not lowered: no form is yet.
fn arguments(field: &ResponseTemplateField, named: &Named) -> Refused {
    match (field.content.parser, named) {
        (ContentParser::Json, Named::Envelope) => Refused::new(
            Construct::JsonEnvelope,
            "a call as one JSON object holding its name and arguments",
        ),
        (ContentParser::Json, Named::Group(_) | Named::Literal(_)) => Refused::new(
            Construct::JsonArguments,
            "a call named by its open or the template, its arguments JSON",
        ),
        (ContentParser::XmlInline, _) => Refused::new(
            Construct::TaggedParameters,
            "arguments as tagged parameters",
        ),
        (ContentParser::KvLines, _) => {
            Refused::new(Construct::KeyValueLines, "arguments as key-value lines")
        }
        (
            ContentParser::Text | ContentParser::Int | ContentParser::Float | ContentParser::Bool,
            _,
        ) => Refused::new(
            Construct::CallContent,
            "tool calls whose content is a text or a number: a call needs a name and arguments",
        ),
    }
}

/// The name a call transform reads: `{"function": {"name": ..., "arguments": "{content}"}}`, the
/// name a group of the open or the template's text; or `{"function": "{content}"}`, an object
/// that names the call. Other keys (`"type": "function"`) have no place in the API's call and are
/// not read.
fn call_transform(transform: Option<&Value>) -> Result<Named, Refused> {
    let refuse = |detail: &str| Err(Refused::new(Construct::CallTransform, detail));
    let Some(Value::Object(transform)) = transform else {
        return refuse("tool calls with no transform, or one that is not an object");
    };
    match transform.get("function") {
        Some(Value::String(content)) if placeholder(content) == Some("content") => {
            Ok(Named::Envelope)
        }
        Some(Value::Object(function)) => {
            let arguments = function.get("arguments").and_then(Value::as_str);
            match arguments.and_then(placeholder) {
                Some("content") => {}
                Some(path) if path.contains('.') => {
                    return Err(Refused::new(
                        Construct::DottedPath,
                        format!("the arguments at {{{path}}}"),
                    ))
                }
                _ => return refuse("arguments that are not the content"),
            }
            match function.get("name") {
                Some(Value::String(name)) => match placeholder(name) {
                    Some(path) if path.contains('.') => Err(Refused::new(
                        Construct::DottedPath,
                        format!("the name at {{{path}}}"),
                    )),
                    Some(group) => Ok(Named::Group(group.to_owned())),
                    None => Ok(Named::Literal(name.clone())),
                },
                _ => refuse("a call with no name"),
            }
        }
        Some(Value::String(content)) => match placeholder(content) {
            Some(path) if path.contains('.') => Err(Refused::new(
                Construct::DottedPath,
                format!("the call at {{{path}}}"),
            )),
            _ => refuse("a function that is not the content"),
        },
        _ => refuse("a transform with no function"),
    }
}

/// A state of the table as it is built.
#[derive(Clone, Debug)]
struct State {
    name: String,
    /// The template's field, if the state is one's.
    field: Option<String>,
    emits: Emits,
    framing: bool,
    trims: bool,
}

/// The table as it is built: markers by text, states, rows `(from, marker, to)`, anchors.
#[derive(Default)]
struct Table {
    terminals: Vec<String>,
    states: Vec<State>,
    rows: Vec<(usize, usize, usize)>,
    anchors: Vec<String>,
}

impl Table {
    fn terminal(&mut self, text: &str) -> usize {
        match self.terminals.iter().position(|t| t == text) {
            Some(index) => index,
            None => {
                self.terminals.push(text.to_owned());
                self.terminals.len() - 1
            }
        }
    }

    fn next(&self, from: usize, on: usize) -> Option<usize> {
        self.rows
            .iter()
            .find(|&&(f, o, _)| f == from && o == on)
            .map(|&(_, _, to)| to)
    }

    /// Add the row `from + on = to`, unless one for `from + on` is there: then the first row
    /// stands, as the pattern added first wins a tie in transformers (an open before a close),
    /// and the state it leads to is returned when it is another.
    fn row(&mut self, from: usize, on: usize, to: usize) -> Option<usize> {
        match self.next(from, on) {
            Some(existing) => (existing != to).then_some(existing),
            None => {
                self.rows.push((from, on, to));
                None
            }
        }
    }

    /// A marker with no row from a state, which begins with a marker that moves the state, moves
    /// where that marker and the ones after it move, as transformers reads the first marker and
    /// goes on from its end: `<|end|><|start|>assistant`, in a state that `<|end|>` closes, closes
    /// it, and the rest is dropped where the state after it drops its text.
    fn derive(&mut self) {
        let direct = self.rows.clone();
        for from in 0..self.states.len() {
            for on in 0..self.terminals.len() {
                if direct.iter().any(|&(f, o, _)| f == from && o == on) {
                    continue;
                }
                if let Some(to) = self.walk(&direct, from, &self.terminals[on]) {
                    self.rows.push((from, on, to));
                }
            }
        }
    }

    /// Where `text` takes the engine from `from` when it begins with a marker that moves `from`,
    /// read marker by marker, the longest first; a rest that begins with no marker is text of the
    /// state reached, which is fine only where that state drops its text and no marker that moves
    /// it could begin inside that rest.
    fn walk(&self, direct: &[(usize, usize, usize)], from: usize, text: &str) -> Option<usize> {
        let (mut state, mut rest, mut steps) = (from, text, 0);
        loop {
            if rest.is_empty() {
                return (steps > 0).then_some(state);
            }
            let step = direct
                .iter()
                .filter(|&&(f, on, _)| f == state && rest.starts_with(self.terminals[on].as_str()))
                .max_by_key(|&&(_, on, _)| self.terminals[on].len());
            match step {
                Some(&(_, on, to)) => {
                    rest = &rest[self.terminals[on].len()..];
                    state = to;
                    steps += 1;
                }
                None => {
                    if steps == 0 || !self.states[state].framing {
                        return None;
                    }
                    let read = direct
                        .iter()
                        .any(|&(f, on, _)| f == state && runs_into(rest, &self.terminals[on], 0));
                    return (!read).then_some(state);
                }
            }
        }
    }

    /// Markers that hide a marker their state moves on. The engine's scanner reads the longest
    /// marker at the earliest place, whatever the state; a marker with no row from a state is that
    /// state's text. Where such a marker holds a marker that moves the state, or runs into one, the
    /// table would read it as text where the template reads the other.
    fn hidden_markers(&self) -> Vec<Refused> {
        let mut out = Vec::new();
        for from in 0..self.states.len() {
            let moving: Vec<&str> = self
                .rows
                .iter()
                .filter(|&&(f, _, _)| f == from)
                .map(|&(_, on, _)| self.terminals[on].as_str())
                .collect();
            for (on, text) in self.terminals.iter().enumerate() {
                if self.next(from, on).is_some() {
                    continue;
                }
                let hides = moving.iter().find(|marker| {
                    (text.len() > marker.len() && text.starts_with(**marker))
                        || runs_into(text, marker, 1)
                });
                if let Some(marker) = hides {
                    out.push(Refused::new(
                        Construct::MarkerInsideMarker,
                        format!(
                            "in {:?}, the marker {text:?} hides {marker:?}, which moves it",
                            self.states[from].name
                        ),
                    ));
                }
            }
        }
        out
    }

    /// The table as a symphony format.
    fn format(&self, stops: &[&str]) -> Format {
        let mut format = Format::new("response_template").reads_values_with(&TRANSFORMERS);
        for text in &self.terminals {
            format = format.terminal(text, text);
        }
        for state in &self.states {
            format = format.state(&state.name, state.emits);
        }
        for &(from, on, to) in &self.rows {
            format = format.transition(
                &self.states[from].name,
                &self.terminals[on],
                &self.states[to].name,
            );
        }
        for state in &self.states {
            if state.framing {
                format = format.framing(&state.name);
            }
            if state.trims {
                format = format.trims(&state.name);
            }
        }
        for anchor in &self.anchors {
            format = format.turn_anchor(anchor);
        }
        format.stop_tokens(stops.iter().copied())
    }

    /// A one-line description: the anchors, then each state with what it emits, its markers in
    /// and out, and its rows of trimming or framing, then the stop tokens.
    fn summary(&self, format: &Format) -> String {
        let mut out = format!("anchors {:?}", self.anchors);
        for (index, state) in self.states.iter().enumerate() {
            let markers = |into: bool| -> Vec<&str> {
                self.rows
                    .iter()
                    .filter(|&&(from, _, to)| {
                        from != to && if into { to == index } else { from == index }
                    })
                    .map(|&(_, on, _)| self.terminals[on].as_str())
                    .collect()
            };
            let emits = match state.emits {
                Emits::Content => "content",
                Emits::Reasoning => "reasoning",
                Emits::Arguments => "calls",
                Emits::Wrapper => "wrapper",
            };
            let _ = write!(out, "; {} {emits}", state.name);
            if state.framing {
                out.push_str(" framing");
            }
            if state.trims {
                out.push_str(" trimmed");
            }
            if index > 0 {
                let _ = write!(out, " in {:?} out {:?}", markers(true), markers(false));
            }
            let dropped: Vec<&str> = self
                .rows
                .iter()
                .filter(|&&(from, _, to)| from == index && to == index)
                .map(|&(_, on, _)| self.terminals[on].as_str())
                .collect();
            if !dropped.is_empty() {
                let _ = write!(out, " drops {dropped:?}");
            }
        }
        let stops: Vec<&str> = self
            .terminals
            .iter()
            .map(String::as_str)
            .filter(|text| format.is_stop_token(text))
            .collect();
        if !stops.is_empty() {
            let _ = write!(out, "; stop tokens {stops:?}");
        }
        out
    }
}

/// Whether `node` is a lookahead as a whole (`(?=[\s\S]*?</think>)`), inside groups that capture
/// nothing.
fn zero_width(node: &Node) -> bool {
    match node {
        Node::Around {
            behind: false,
            negated: false,
            ..
        } => true,
        Node::Group(None, inner) => zero_width(inner),
        Node::Cat(nodes) => matches!(nodes.as_slice(), [only] if zero_width(only)),
        _ => false,
    }
}

/// Whether `marker` occurs in `text` at a character offset of `from` or more, or begins in it
/// there and runs past its end.
fn runs_into(text: &str, marker: &str, from: usize) -> bool {
    text.char_indices().skip(from).any(|(at, _)| {
        let rest = &text[at..];
        rest.starts_with(marker) || (rest.len() < marker.len() && marker.starts_with(rest))
    })
}

#[cfg(test)]
mod tests;
