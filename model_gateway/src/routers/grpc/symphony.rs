//! The regular pipeline's driver for `smg-symphony`: a model whose parser setting names a symphony
//! table has its output read by that table and rendered by the crate's adapters, in place of the
//! reasoning and tool parsers.
//!
//! Selection rides on the settings that already pick parsers. A tool or reasoning parser name of
//! the form `symphony:<table>`, from a model card (worker label or `WorkerSpec`) or from
//! `--tool-call-parser` / `--reasoning-parser`, selects the row of that name in [`TABLES`]. The card
//! comes first: a card that names either parser decides alone, and the flags select a table only
//! for a model whose card names neither. Model-id detection never yields such a name, so a model
//! selects a table only when its configuration says so. A row also names the parsers the gateway
//! takes for the table's family without symphony, and the resolver answers with them wherever a
//! parser is looked up by name: preparation (tool constraints, `skip_special_tokens`, the stop
//! decoder) stays what it is for the family, and only the reading of the output changes.
//!
//! Request building arms a [`SymphonySpec`] on the response spec. The Chat and Messages response
//! paths, streamed and whole, branch where they parse: each choice gets a [`Choice`], which feeds
//! the prompt, the decoded deltas and the end of the stream to an [`Engine`] and applies the
//! request's [`Policy`] to the events, and the adapters render the events. The envelope around the
//! parse (role and finish chunks, usage, stop decoding, prefill/decode, metrics) is unchanged. The
//! Responses API runs on the Chat pipeline and follows it.
//!
//! One request shape stays with the legacy parsers on a selected table: a call forced through a
//! JSON-schema constraint, which a family without a structural tag gets. The model then writes the
//! bare arguments, which no table reads.

use std::{collections::HashMap, sync::Arc};

use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    chat::{ChatChoice, ChatStreamChoice},
    common::{ChatLogProbs, Tool, ToolChoice, ToolChoiceValue},
    messages::{self, MessageStreamEvent},
};
use serde_json::Value;
use symphony::{
    adapt, CallSyntax, Declared, Engine, EngineFinish, Event, Events, FinishReason, Format, Input,
    Parser, Text,
};
use tracing::warn;

use super::{
    spec::{ChatResponseSpec, MessagesResponseSpec},
    utils,
};

/// The parser-name prefix that selects a symphony table.
pub(crate) const PREFIX: &str = "symphony:";

/// A table a parser setting can select.
#[derive(Debug)]
pub(crate) struct Table {
    /// The name after [`PREFIX`].
    pub name: &'static str,
    /// Builds the table.
    pub format: fn() -> Format,
    /// The tool parser the gateway takes for the family without symphony. It builds the tool
    /// constraints.
    pub tool_parser: Option<&'static str>,
    /// The reasoning parser the gateway takes for the family without symphony. It decides
    /// whether special tokens are kept.
    pub reasoning_parser: Option<&'static str>,
}

/// The selectable tables, each with the parser names in the crate's README "SMG today" column:
/// groups the README marks ready that this driver is tested with (DeepSeek V4.1, Hy4, Qwen3 with
/// JSON calls). Another ready table is one more row.
pub(crate) static TABLES: &[Table] = &[
    Table {
        name: "deepseek_v4.1",
        format: symphony::deepseek_v4_1,
        tool_parser: Some("deepseek_v41"),
        reasoning_parser: Some("deepseek_v41"),
    },
    Table {
        name: "hy4",
        format: symphony::hy4,
        tool_parser: Some("hy_v4"),
        reasoning_parser: Some("hy_v4"),
    },
    Table {
        name: "qwen3",
        format: qwen3_json,
        tool_parser: Some("qwen"),
        reasoning_parser: Some("qwen3"),
    },
];

fn qwen3_json() -> Format {
    symphony::qwen3(CallSyntax::Json)
}

/// The table a parser name selects, if it names one.
pub(crate) fn table(name: &str) -> Option<&'static Table> {
    let name = name.strip_prefix(PREFIX)?;
    TABLES.iter().find(|table| table.name == name)
}

/// Whether a parser name selects a table, for the checks that refuse unknown parser names.
pub(crate) fn known(name: &str) -> bool {
    table(name).is_some()
}

/// The request's say over the events, applied before the adapters render them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Policy {
    /// The request lets the model call tools. Otherwise a call is text: what the model wrote of
    /// it, without the markers, is content (with them, when reasoning is not separated).
    pub calls: bool,
    /// Reasoning is reported apart from the content. Otherwise the output is read as SGLang reads
    /// it with `separate_reasoning: false`, where no reasoning parser runs: every byte the model
    /// wrote that is not a returned call is content as written, the format's markers included,
    /// so a request that lets the model call no tool gets the raw text. Calls the request lets
    /// the model make are still returned, and each takes its own bytes out of the text: what its
    /// events carry, the markers and whitespace inside it, the marker that opened it and the one
    /// that ended it, as the table names them ([`Format::call_terminals`]). Every other marker
    /// stays where the model wrote it, the ones of the regions around the call and of a block the
    /// calls sit in among them.
    pub separate_reasoning: bool,
    /// Only the first call is returned (`parallel_tool_calls: false`,
    /// `disable_parallel_tool_use: true`). The calls after it are dropped; without reasoning
    /// separation they are text as written, every byte of them, as a call the request does not
    /// let the model make is.
    pub first_call_only: bool,
    /// Calls made earlier in the conversation, for the id formats that number calls.
    pub history_tool_calls_count: usize,
}

/// What request building hands the response paths for a request a table reads.
#[derive(Debug)]
pub(crate) struct SymphonySpec {
    table: &'static Table,
    declared: Declared,
    prompt: String,
    policy: Policy,
    /// The ids of the table's markers that end a call and are one token each in the model's
    /// vocabulary: a stop on one of them may close the call the stream stopped in, and a stop on
    /// any other token closes none ([`Self::finish_of`]). `None` when no tokenizer looked them
    /// up.
    call_stops: Option<Vec<u32>>,
}

impl SymphonySpec {
    /// A spec for `table` over `prompt`, the rendered prompt the engine is given, with the
    /// request's `tools`, which type the arguments of a tagged call.
    pub(crate) fn new(
        table: &'static Table,
        tools: &[Tool],
        prompt: impl Into<String>,
        policy: Policy,
    ) -> Self {
        Self {
            table,
            declared: Declared::of(tools),
            prompt: prompt.into(),
            policy,
            call_stops: None,
        }
    }

    /// The spec with the ids of the table's call-ending markers that `tokenizer` has as one
    /// token each ([`Self::finish_of`]).
    fn with_call_stops(mut self, tokenizer: Option<&dyn Tokenizer>) -> Self {
        self.call_stops = tokenizer.map(|tokenizer| {
            let format = (self.table.format)();
            let (_, ends) = format.call_terminals();
            ends.into_iter()
                .filter_map(|text| tokenizer.token_to_id(text))
                .collect()
        });
        self
    }

    /// The finish a choice's end takes for the gateway's finish reason `finish` and the stop the
    /// choice matched, if any. A stop on a stop sequence of the request's (`matched_stop` is the
    /// matched string) is [`STOP_SEQUENCE`]; a stop on a token that is not one of the table's
    /// call-ending markers (`matched_stop` is its id) is [`STOP_TOKEN`]; any other finish is
    /// `finish`. Neither of the two completes a call: only a call's own stop token closes the
    /// call the stream stopped in, where the table's stop row says the stop closes it.
    pub(crate) fn finish_of<'a>(&self, finish: &'a str, matched_stop: Option<&Value>) -> &'a str {
        match matched_stop {
            Some(Value::String(_)) if finish == "stop" => STOP_SEQUENCE,
            Some(Value::Number(id)) if finish == "stop" && !self.ends_calls(id) => STOP_TOKEN,
            _ => finish,
        }
    }

    /// Whether the stop token `id` is one of the table's call-ending markers, or may be, when no
    /// tokenizer looked them up.
    fn ends_calls(&self, id: &serde_json::Number) -> bool {
        self.call_stops.as_ref().is_none_or(|stops| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .is_some_and(|id| stops.contains(&id))
        })
    }

    /// The spec for a Chat request, or `None` when its tool constraint keeps it on the legacy
    /// parsers. `tokenizer` is the model's, which names the stop tokens that end a call.
    pub(crate) fn for_chat(
        table: &'static Table,
        spec: &ChatResponseSpec,
        parallel_tool_calls: Option<bool>,
        prompt: &str,
        tool_constraint: Option<&(String, String)>,
        tokenizer: Option<&dyn Tokenizer>,
    ) -> Option<Arc<Self>> {
        if forces_json(tool_constraint) {
            return None;
        }
        let policy = Policy {
            calls: spec.tools.is_some()
                && !matches!(
                    spec.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                ),
            separate_reasoning: spec.separate_reasoning,
            first_call_only: parallel_tool_calls == Some(false),
            history_tool_calls_count: spec.history_tool_calls_count,
        };
        let tools = spec.tools.as_deref().unwrap_or_default();
        Some(Arc::new(
            Self::new(table, tools, prompt, policy).with_call_stops(tokenizer),
        ))
    }

    /// The spec for a Messages request, or `None` when its tool constraint keeps it on the
    /// legacy parsers. Reasoning is always a thinking block. `tokenizer` is the model's, which
    /// names the stop tokens that end a call.
    pub(crate) fn for_messages(
        table: &'static Table,
        spec: &MessagesResponseSpec,
        prompt: &str,
        tool_constraint: Option<&(String, String)>,
        tokenizer: Option<&dyn Tokenizer>,
    ) -> Option<Arc<Self>> {
        if forces_json(tool_constraint) {
            return None;
        }
        let policy = Policy {
            calls: spec.has_tools && !matches!(spec.tool_choice, Some(messages::ToolChoice::None)),
            separate_reasoning: true,
            first_call_only: matches!(
                spec.tool_choice,
                Some(
                    messages::ToolChoice::Auto {
                        disable_parallel_tool_use: Some(true)
                    } | messages::ToolChoice::Any {
                        disable_parallel_tool_use: Some(true)
                    } | messages::ToolChoice::Tool {
                        disable_parallel_tool_use: Some(true),
                        ..
                    }
                )
            ),
            history_tool_calls_count: spec.history_tool_calls_count,
        };
        Some(Arc::new(
            Self::new(table, &spec.chat_tools, prompt, policy).with_call_stops(tokenizer),
        ))
    }
}

/// Whether the tool constraint forces bare JSON arguments, which no table reads.
fn forces_json(tool_constraint: Option<&(String, String)>) -> bool {
    tool_constraint.is_some_and(|(kind, _)| kind == "json_schema")
}

/// One choice of a response a table reads: an engine fed the prompt, the decoded deltas and the
/// end of the stream, with the request's policy applied to the events it emits.
pub(crate) struct Choice {
    engine: Engine,
    events: Events,
    policy: Policy,
    model: String,
    /// The returned calls that ended with complete arguments, the calls made.
    calls: u32,
    /// A returned call ended with arguments that are not a whole JSON object: the turn made no
    /// call, whatever the other calls did ([`Self::calls`]).
    cut: bool,
    /// The arguments of each returned call that has not ended, as far as they arrived.
    arguments: HashMap<u32, String>,
    /// The table's markers that open a call and the ones that end it.
    opens: Vec<String>,
    ends: Vec<String>,
    /// Without reasoning separation, with calls: the markers that arrived since the last event
    /// that was not one, held until the next event says whether the last of them opened a call
    /// (and goes with it); the others are content.
    markers: Vec<Text>,
    /// A call is open: the markers and whitespace inside it are its own.
    inside: bool,
    /// A call ended without the marker that ended it among its own bytes: the whitespace and the
    /// first ending marker that follow it are its own.
    closing: bool,
    /// Without reasoning separation: the call open now is one that only the first call being
    /// returned drops, so its bytes are content as written.
    shown: bool,
    /// The calls that have started and not ended: a stop on a token that ends no call leaves
    /// such a call cut ([`STOP_TOKEN`]).
    open_calls: u32,
    /// The engine refused an input. It is fed nothing more but the end, and later text is
    /// content, so that nothing is lost.
    failed: bool,
    ended: bool,
}

impl Choice {
    /// A choice that has read the prompt and nothing else. `model` names the model in call ids.
    pub(crate) fn new(spec: &SymphonySpec, model: &str) -> Self {
        let format = (spec.table.format)();
        let (opens, ends) = format.call_terminals();
        let owned = |texts: Vec<&str>| texts.into_iter().map(str::to_string).collect();
        let (opens, ends) = (owned(opens), owned(ends));
        let mut choice = Self {
            engine: Engine::new(format, spec.declared.clone()),
            events: Events::new(),
            policy: spec.policy,
            model: model.to_string(),
            calls: 0,
            cut: false,
            arguments: HashMap::new(),
            opens,
            ends,
            markers: Vec::new(),
            inside: false,
            closing: false,
            shown: false,
            open_calls: 0,
            failed: false,
            ended: false,
        };
        choice.feed(Input::Prompt {
            token_ids: &[],
            text: &spec.prompt,
        });
        choice
    }

    /// The events of a whole output that ended with `finish`, the gateway's finish reason as
    /// [`SymphonySpec::finish_of`] gives it.
    pub(crate) fn whole(spec: &SymphonySpec, model: &str, text: &str, finish: &str) -> Vec<Event> {
        let mut choice = Self::new(spec, model);
        let mut events = choice.delta(text);
        events.extend(choice.end(Some(finish)));
        events
    }

    /// The events one decoded delta completes.
    pub(crate) fn delta(&mut self, text: &str) -> Vec<Event> {
        if self.ended {
            return Vec::new();
        }
        if self.failed {
            self.events.push_content(Text::uncounted(text));
        } else {
            // Token ids and their spans are not threaded through yet; the engine then leaves
            // its text uncounted.
            self.feed(Input::Delta {
                token_ids: &[],
                text,
                spans: &[],
            });
        }
        self.take()
    }

    /// The events the end of the stream releases, `Finish` last. `finish` is the gateway's
    /// finish reason for the choice, a stop-decoder match already folded in, as
    /// [`SymphonySpec::finish_of`] gives it, and `None` when the stream ended without one. Only
    /// the first call does anything.
    pub(crate) fn end(&mut self, finish: Option<&str>) -> Vec<Event> {
        if self.ended {
            return Vec::new();
        }
        self.ended = true;
        // A stop on a token that ends no call leaves the call it stopped inside cut; outside a
        // call it is the model's stop, as any other.
        let finish = match finish {
            Some(STOP_TOKEN) if self.open_calls == 0 => Some("stop"),
            finish => finish,
        };
        self.feed(Input::End {
            finish: engine_finish(finish),
        });
        self.take()
    }

    /// The calls this choice has made so far: the returned calls that ended with complete
    /// arguments, or none once a returned call ended with arguments that are not a whole JSON
    /// object. A call the end of the stream cut, before or after its first argument, is returned
    /// with what arrived but is not made, and a turn that holds one is no tool turn, however many
    /// calls before it were whole: it turns no `stop` into `tool_calls` and no Messages turn into
    /// `tool_use`, so a client runs none of its calls.
    pub(crate) fn calls(&self) -> u32 {
        if self.cut {
            0
        } else {
            self.calls
        }
    }

    /// Whether a returned call ended with arguments that are not a whole JSON object.
    pub(crate) fn cut(&self) -> bool {
        self.cut
    }

    fn feed(&mut self, input: Input<'_>) {
        if let Err(error) = self.engine.feed(input, &mut self.events) {
            warn!(%error, "symphony parser stopped; the rest of the output is content");
            self.failed = true;
        }
    }

    /// The events emitted since the last call, with the policy applied: a call the request did
    /// not ask for is content, reasoning the request does not separate is content (with the
    /// markers around it, see [`Policy::separate_reasoning`]), calls after the first are dropped
    /// when only the first is returned (content as written without reasoning separation, see
    /// [`Policy::first_call_only`]), every call returned gets its id from the gateway's scheme, a
    /// returned call that ends before the end of the stream without arguments gets `{}`, and a
    /// returned call counts as made once it ends with complete arguments, unless one of the
    /// returned calls did not ([`Self::calls`]).
    fn take(&mut self) -> Vec<Event> {
        let policy = self.policy;
        let raw = !policy.separate_reasoning;
        let dropped = |index: u32| policy.first_call_only && index > 0;
        let mut out = Events::new();
        for event in self.events.drain() {
            match &event {
                Event::ToolCallStart { .. } => self.open_calls += 1,
                Event::ToolCallEnd { .. } => self.open_calls = self.open_calls.saturating_sub(1),
                _ => {}
            }
            match event {
                // Without reasoning separation, a call only the first call being returned drops
                // is text as written: the markers before it, the one that opened it among them,
                // and every byte of it, the markers inside it included.
                Event::ToolCallStart { index, source, .. }
                    if policy.calls && raw && dropped(index) =>
                {
                    self.closing = false;
                    for marker in self.markers.drain(..) {
                        out.push_content(marker);
                    }
                    out.push_content(source);
                    self.inside = true;
                    self.shown = true;
                }
                Event::ToolCallArguments { index, source, .. }
                    if policy.calls && raw && dropped(index) =>
                {
                    out.push_content(source);
                }
                Event::ToolCallEnd { index, source } if policy.calls && raw && dropped(index) => {
                    out.push_content(source);
                    self.inside = false;
                    self.shown = false;
                    self.closing = false;
                }
                // A returned call's events. Without reasoning separation the marker that opened
                // it is its own and the markers before that one are content.
                Event::ToolCallStart {
                    index,
                    name,
                    source,
                    ..
                } if policy.calls => {
                    if raw {
                        self.open_call(&mut out);
                    }
                    if dropped(index) {
                        continue;
                    }
                    let id = utils::generate_tool_call_id(
                        &self.model,
                        &name,
                        index as usize,
                        policy.history_tool_calls_count,
                    );
                    self.arguments.insert(index, String::new());
                    out.push(Event::ToolCallStart {
                        index,
                        id,
                        name,
                        source,
                    });
                }
                Event::ToolCallArguments {
                    index,
                    json,
                    source,
                } if policy.calls => {
                    if dropped(index) {
                        continue;
                    }
                    self.arguments.entry(index).or_default().push_str(&json);
                    out.push(Event::ToolCallArguments {
                        index,
                        json,
                        source,
                    });
                }
                Event::ToolCallEnd { index, source } if policy.calls => {
                    // A call whose end carries the marker that ended it has taken it; otherwise
                    // the marker follows, and the whitespace before it is the call's too.
                    self.inside = false;
                    self.closing = !self
                        .ends
                        .iter()
                        .any(|end| source.text.ends_with(end.as_str()));
                    if dropped(index) {
                        continue;
                    }
                    let mut arguments = self.arguments.remove(&index).unwrap_or_default();
                    // A table renders only the bytes the model wrote, so a call written without
                    // arguments would have none; the API's arguments are a JSON object, `{}`
                    // here as under the legacy parsers. A call the end of the stream cut keeps
                    // what arrived, nothing included: its arguments never look complete.
                    if !self.ended && arguments.is_empty() {
                        arguments.push_str("{}");
                        out.push(Event::ToolCallArguments {
                            index,
                            json: arguments.clone(),
                            source: Text::default(),
                        });
                    }
                    if serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).is_ok() {
                        self.calls += 1;
                    } else {
                        self.cut = true;
                    }
                    out.push(Event::ToolCallEnd { index, source });
                }
                Event::Dropped { text, .. } if raw => {
                    if !policy.calls || self.shown {
                        out.push_content(text);
                    } else if self.inside {
                        // A marker or the template's whitespace inside a call is the call's.
                    } else if self.closing && text.text.trim().is_empty() {
                        // The whitespace between a call and the marker that ends it.
                    } else if self.closing && self.ends.contains(&text.text) {
                        self.closing = false;
                    } else {
                        self.closing = false;
                        self.markers.push(text);
                    }
                }
                Event::ReasoningStart | Event::ReasoningEnd if raw => {}
                event => {
                    if raw {
                        self.closing = false;
                        for marker in self.markers.drain(..) {
                            out.push_content(marker);
                        }
                    }
                    match event {
                        Event::ToolCallStart { source, .. }
                        | Event::ToolCallArguments { source, .. }
                        | Event::ToolCallEnd { source, .. } => out.push_content(source),
                        Event::Reasoning(text) if raw => out.push_content(text),
                        Event::Finish {
                            reason,
                            reasoning_tokens,
                            ..
                        } => out.push(Event::Finish {
                            reason: match reason {
                                FinishReason::Other(other)
                                    if other == STOP_SEQUENCE || other == STOP_TOKEN =>
                                {
                                    FinishReason::Stop
                                }
                                // A backend that names its own finish `tool_calls` makes no tool
                                // turn of one with a call cut short either.
                                FinishReason::Other(other) if other == "tool_calls" && self.cut => {
                                    FinishReason::Stop
                                }
                                reason => reason,
                            },
                            tool_calls: self.calls(),
                            reasoning_tokens,
                        }),
                        event => out.push(event),
                    }
                }
            }
        }
        out.drain()
    }

    /// A call starts, without reasoning separation: the last marker held is the one that opened
    /// it when the table opens calls with it, and goes with the call; the markers before it are
    /// the text's, and so is a last one that opens no call.
    fn open_call(&mut self, out: &mut Events) {
        if self
            .markers
            .last()
            .is_some_and(|marker| self.opens.contains(&marker.text))
        {
            self.markers.pop();
        }
        for marker in self.markers.drain(..) {
            out.push_content(marker);
        }
        self.inside = true;
        self.closing = false;
    }
}

/// What [`SymphonySpec::finish_of`] hands a choice's end for a stop on a stop sequence of the
/// request's, and the engine as a finish it does not interpret: a stop sequence never completes a
/// call, so the table must not close the call it cut, as it closes one at the stop token where
/// its stop row says so. The `Finish` the choice reports is `Stop` again.
pub(crate) const STOP_SEQUENCE: &str = "stop_sequence";

/// What [`SymphonySpec::finish_of`] hands a choice's end for a stop on a token that is not one of
/// the table's call-ending markers, the answer's stop token written inside a call among them.
/// When a call is open the choice hands the engine a finish it does not interpret, since only a
/// call's own stop token closes the call the stream stopped in: the call stays cut, with what
/// arrived, and the turn is no tool turn ([`Choice::calls`]). Outside a call the engine gets the
/// stop as it is. The `Finish` the choice reports is `Stop` either way.
pub(crate) const STOP_TOKEN: &str = "stop_token";

/// The engine's finish for the gateway's finish reason.
fn engine_finish(finish: Option<&str>) -> EngineFinish {
    match finish {
        Some("stop") => EngineFinish::Stop,
        Some("length") => EngineFinish::Length,
        Some("abort") | None => EngineFinish::Abort,
        Some(other) => EngineFinish::Other(other.to_string()),
    }
}

/// The Chat stream choices for a choice's events, without the finishing choice, which the stream
/// sends from its own finish bookkeeping. The chunk's logprobs go with its first content, or with
/// its first reasoning when it has no content, where the legacy stream puts them.
pub(crate) fn chat_deltas(
    index: u32,
    events: &[Event],
    mut logprobs: Option<ChatLogProbs>,
) -> Vec<ChatStreamChoice> {
    let mut choices: Vec<ChatStreamChoice> = adapt::chat::deltas(index, events)
        .into_iter()
        .filter(|choice| choice.finish_reason.is_none())
        .collect();
    let carrier = choices
        .iter()
        .position(|choice| choice.delta.content.is_some())
        .or_else(|| {
            choices
                .iter()
                .position(|choice| choice.delta.reasoning_content.is_some())
        });
    if let Some(at) = carrier {
        choices[at].logprobs = logprobs.take();
    }
    choices
}

/// The Chat message for a whole output's events.
pub(crate) fn chat_message(index: u32, events: &[Event]) -> ChatChoice {
    adapt::chat::message(index, events)
}

/// The content blocks and stop reason of a whole output's events, for the Messages API.
///
/// The whole message and the stream ([`MessagesStream`]) render the same events, each by the
/// adapter's rules (`symphony::adapt::messages`), and agree for the shapes models write most: one
/// thought, then text, then calls. Elsewhere they differ by that adapter's policy, which this
/// driver keeps as it is: the whole message gathers each kind into one block in the API's order
/// (thinking, text, tool_use), where the stream keeps the order the model wrote, gives each
/// reasoning region a thinking block of its own and lets go of whitespace between blocks; and a
/// call whose arguments are not a whole JSON value (one the end of the stream cut) has `{}` as its
/// whole `input`, where the stream has already sent the fragments that arrived as `partial_json`.
/// Either way such a call is not a call made, so the turn is not `tool_use`, and neither is a turn
/// that holds one beside whole calls ([`Choice::calls`]). The stream also lets text another kind
/// of block closed a call's `tool_use` block with take the call's later fragments, so a call the
/// model interrupted with text out of place streams as an input that is not whole JSON, while
/// the whole message gives every fragment to its call; the stream then makes no call either
/// ([`MessagesStream::calls`]), and the whole message keeps the call, whose input is whole.
pub(crate) fn messages_output(events: &[Event]) -> adapt::messages::Output {
    adapt::messages::output(events)
}

/// A Messages stream's choice: its events as `content_block_*` events, one block open at a time;
/// see [`messages_output`] for where the stream and the whole message differ.
pub(crate) struct MessagesStream {
    choice: Choice,
    blocks: adapt::messages::Stream,
    /// The input each `tool_use` block has streamed, by block index.
    inputs: HashMap<u32, String>,
}

impl MessagesStream {
    /// A stream that has read the prompt.
    pub(crate) fn new(spec: &SymphonySpec, model: &str) -> Self {
        Self {
            choice: Choice::new(spec, model),
            blocks: adapt::messages::Stream::new(),
            inputs: HashMap::new(),
        }
    }

    /// The block events one decoded delta completes.
    pub(crate) fn delta(&mut self, text: &str) -> Vec<MessageStreamEvent> {
        let events = self.choice.delta(text);
        self.render(&events)
    }

    /// The block events the end of the stream releases; the open block is stopped. `finish` is
    /// as for [`Choice::end`].
    pub(crate) fn end(&mut self, finish: Option<&str>) -> Vec<MessageStreamEvent> {
        let events = self.choice.end(finish);
        self.render(&events)
    }

    /// The calls this stream has made so far, as [`Choice::calls`] counts them, or none while a
    /// `tool_use` block it streamed holds an input that is not a whole JSON object, as a block
    /// another block closed before its call's last fragment does: the client's input is what the
    /// stream sent.
    pub(crate) fn calls(&self) -> u32 {
        if self.cut() {
            0
        } else {
            self.choice.calls()
        }
    }

    /// Whether a call this stream returned is not whole: cut short, or streamed as an input that
    /// is not a whole JSON object.
    pub(crate) fn cut(&self) -> bool {
        self.choice.cut()
            || self
                .inputs
                .values()
                .any(|input| serde_json::from_str::<serde_json::Map<String, Value>>(input).is_err())
    }

    fn render(&mut self, events: &[Event]) -> Vec<MessageStreamEvent> {
        let mut out = Vec::new();
        for event in events {
            // The stop reason is the stream's own, decided with the stop sequence it knows.
            let _ = self.blocks.feed(event, &mut out);
        }
        for event in &out {
            match event {
                MessageStreamEvent::ContentBlockStart {
                    index,
                    content_block: messages::ContentBlock::ToolUse { .. },
                } => {
                    self.inputs.insert(*index, String::new());
                }
                MessageStreamEvent::ContentBlockDelta {
                    index,
                    delta: messages::ContentBlockDelta::InputJsonDelta { partial_json },
                } => {
                    if let Some(input) = self.inputs.get_mut(index) {
                        input.push_str(partial_json);
                    }
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        chat::ChatCompletionRequest, messages::CreateMessageRequest, model_card::ModelCard,
    };
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::{json, Value};
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{
        routers::grpc::utils::ParserResolver,
        worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
    };

    /// Kimi's call ids are deterministic, `functions.<name>:<n>`, so outputs compare whole.
    const KIMI: &str = "kimi-k2";
    const PROMPT: &str = "<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n";
    const CALL: &str =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    /// A call written without an arguments member.
    const BARE_CALL: &str = "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>";

    fn qwen3() -> &'static Table {
        table("symphony:qwen3").expect("the qwen3 row")
    }

    fn tool(name: &str) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}
            }}}
        }))
        .expect("tool")
    }

    fn policy() -> Policy {
        Policy {
            calls: true,
            separate_reasoning: true,
            first_call_only: false,
            history_tool_calls_count: 0,
        }
    }

    fn spec(policy: Policy) -> SymphonySpec {
        spec_with(qwen3(), policy)
    }

    fn spec_with(table: &'static Table, policy: Policy) -> SymphonySpec {
        SymphonySpec::new(table, &[tool("get_weather")], PROMPT, policy)
    }

    fn message(spec: &SymphonySpec, output: &str, finish: &str) -> Value {
        let events = Choice::whole(spec, KIMI, output, finish);
        serde_json::to_value(chat_message(0, &events)).expect("serializable")
    }

    fn configured(tool: Option<&str>, reasoning: Option<&str>) -> ParserResolver {
        ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            tool.map(str::to_string),
            reasoning.map(str::to_string),
            None,
        )
    }

    #[test]
    fn a_parser_name_selects_a_table_only_with_the_prefix() {
        assert_eq!(table("symphony:qwen3").map(|t| t.name), Some("qwen3"));
        assert_eq!(table("symphony:hy4").map(|t| t.name), Some("hy4"));
        assert_eq!(
            table("symphony:deepseek_v4.1").map(|t| t.name),
            Some("deepseek_v4.1")
        );
        for name in [
            "qwen3",
            "qwen",
            "symphony:",
            "symphony:nope",
            "Symphony:qwen3",
        ] {
            assert!(table(name).is_none(), "{name}");
            assert!(!known(name), "{name}");
        }
    }

    /// The check every row passes, a new one included: its table builds and runs, its name is
    /// unique, and the parsers it names for its family are registered.
    #[test]
    fn every_table_runs_and_names_registered_parsers() {
        let tools = ToolParserFactory::new();
        let reasoning = ReasoningParserFactory::new();
        for (position, row) in TABLES.iter().enumerate() {
            let format = (row.format)();
            assert_eq!(format.validate(), Ok(()), "{}", row.name);
            let mut choice = Choice::new(&SymphonySpec::new(row, &[], "", policy()), "m");
            let events = choice.end(Some("stop"));
            assert!(
                matches!(events.last(), Some(Event::Finish { .. })),
                "{}",
                row.name
            );
            assert!(
                TABLES[..position]
                    .iter()
                    .all(|other| other.name != row.name),
                "{} is listed twice",
                row.name
            );
            if let Some(name) = row.tool_parser {
                assert!(tools.registry().has_parser(name), "{}: {name}", row.name);
            }
            if let Some(name) = row.reasoning_parser {
                assert!(
                    reasoning.registry().has_parser(name),
                    "{}: {name}",
                    row.name
                );
            }
        }
    }

    #[test]
    fn a_selected_table_resolves_both_names_to_its_family_parsers() {
        for (tool, reasoning) in [
            (Some("symphony:qwen3"), None),
            (None, Some("symphony:qwen3")),
            (Some("symphony:qwen3"), Some("deepseek_r1")),
            (Some("json"), Some("symphony:qwen3")),
        ] {
            let resolver = configured(tool, reasoning);
            assert_eq!(resolver.symphony("m").map(|t| t.name), Some("qwen3"));
            assert_eq!(resolver.tool_parser("m").as_deref(), Some("qwen"));
            assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("qwen3"));
        }
        // The tool parser's name wins when both select a table.
        let resolver = configured(Some("symphony:hy4"), Some("symphony:qwen3"));
        assert_eq!(resolver.symphony("m").map(|t| t.name), Some("hy4"));
        assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("hy_v4"));
    }

    #[test]
    fn names_without_a_table_resolve_as_before() {
        let resolver = configured(Some("qwen"), Some("qwen3"));
        assert!(resolver.symphony("m").is_none());
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("qwen"));
        assert_eq!(resolver.reasoning_parser("m").as_deref(), Some("qwen3"));
        // Model-id detection never selects a table.
        let resolver = configured(None, None);
        for model in [
            "Qwen/Qwen3-8B",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "tencent/Hy4-preview",
        ] {
            assert!(resolver.symphony(model).is_none(), "{model}");
            assert_eq!(resolver.tool_parser(model), None);
        }
        // A name that selects no table is left as written; startup refuses it.
        let resolver = configured(Some("symphony:nope"), None);
        assert!(resolver.symphony("m").is_none());
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("symphony:nope"));
    }

    #[test]
    fn a_model_card_selects_a_table_over_the_flag() {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("http://w1:8000")
            .model(ModelCard::new("m").with_tool_parser("symphony:hy4"))
            .worker_type(WorkerType::Regular)
            .build();
        registry.register(Arc::new(worker));
        let resolver = ParserResolver::new(registry, Some("qwen".to_string()), None, None);
        assert_eq!(resolver.symphony("m").map(|t| t.name), Some("hy4"));
        assert_eq!(resolver.tool_parser("m").as_deref(), Some("hy_v4"));
        // Another model keeps the flag's legacy parser.
        assert!(resolver.symphony("other").is_none());
        assert_eq!(resolver.tool_parser("other").as_deref(), Some("qwen"));
    }

    #[test]
    fn a_model_card_that_names_a_parser_keeps_the_flags_from_selecting_a_table() {
        let registry = Arc::new(WorkerRegistry::new());
        for card in [
            ModelCard::new("legacy").with_tool_parser("hermes"),
            ModelCard::new("table").with_reasoning_parser("symphony:hy4"),
        ] {
            let worker = BasicWorkerBuilder::new(format!("http://{}:8000", card.id))
                .model(card)
                .worker_type(WorkerType::Regular)
                .build();
            registry.register(Arc::new(worker));
        }
        let resolver =
            ParserResolver::new(registry, None, Some("symphony:qwen3".to_string()), None);
        // The card's legacy tool parser stands; the flag's table name, which selects nothing for
        // this model, stands for its family's reasoning parser.
        assert!(resolver.symphony("legacy").is_none());
        assert_eq!(resolver.tool_parser("legacy").as_deref(), Some("hermes"));
        assert_eq!(
            resolver.reasoning_parser("legacy").as_deref(),
            Some("qwen3")
        );
        // The card's table wins over the flag's, whatever side each names.
        assert_eq!(resolver.symphony("table").map(|t| t.name), Some("hy4"));
        assert_eq!(resolver.tool_parser("table").as_deref(), Some("hy_v4"));
        // A model without a card takes the flag's table.
        assert_eq!(resolver.symphony("other").map(|t| t.name), Some("qwen3"));
        assert_eq!(resolver.tool_parser("other").as_deref(), Some("qwen"));
    }

    #[test]
    fn a_selected_table_gets_its_family_s_tool_constraints() {
        // Preparation builds tool constraints from the resolved tool-parser name, so a model that
        // selects a table is constrained exactly as under the family's own parser names.
        let registry = ToolParserFactory::new();
        let choices: Vec<ToolChoice> = [
            json!("required"),
            json!({
                "type": "function", "function": {"name": "get_weather"}
            }),
        ]
        .into_iter()
        .map(|choice| serde_json::from_value(choice).expect("tool choice"))
        .collect();
        for row in TABLES {
            let name = format!("{PREFIX}{}", row.name);
            let selected = configured(Some(&name), None).tool_parser("m");
            assert_eq!(selected.as_deref(), row.tool_parser, "{}", row.name);
            for (choice, reasoning) in choices.iter().flat_map(|c| [(c, false), (c, true)]) {
                let constraint = |name: Option<&str>| {
                    registry
                        .registry()
                        .generate_tool_constraint(name, &[tool("get_weather")], choice, reasoning)
                        .expect("constraint")
                        .map(|constraint| constraint.to_tuple())
                };
                assert_eq!(
                    constraint(selected.as_deref()),
                    constraint(row.tool_parser),
                    "{}",
                    row.name
                );
            }
        }
    }

    fn chat_spec(request: Value) -> ChatResponseSpec {
        let request: ChatCompletionRequest = serde_json::from_value(request).expect("request");
        ChatResponseSpec::from(&request)
    }

    #[test]
    fn a_json_schema_tool_constraint_keeps_the_request_on_the_legacy_parsers() {
        // The family's own registry decides: Qwen has no structural tag, so a forced call is a
        // JSON schema; Hy4 has one, so the table reads the forced call.
        let registry = ToolParserFactory::new();
        let forced = serde_json::from_value(json!({
            "type": "function", "function": {"name": "get_weather"}
        }))
        .expect("tool choice");
        let spec = chat_spec(json!({
            "model": "m", "messages": [], "tools": [tool("get_weather")],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }));
        for (row, kind, armed) in [
            ("symphony:qwen3", "json_schema", false),
            ("symphony:hy4", "structural_tag", true),
        ] {
            let row = table(row).expect("row");
            let constraint = registry
                .registry()
                .generate_tool_constraint(row.tool_parser, &[tool("get_weather")], &forced, false)
                .expect("constraint")
                .expect("a forced call is constrained")
                .to_tuple();
            assert_eq!(constraint.0, kind, "{}", row.name);
            assert_eq!(
                SymphonySpec::for_chat(row, &spec, None, PROMPT, Some(&constraint), None).is_some(),
                armed,
                "{}",
                row.name
            );
        }
        assert!(SymphonySpec::for_chat(qwen3(), &spec, None, PROMPT, None, None).is_some());
    }

    #[test]
    fn a_chat_request_sets_the_policy() {
        let armed = |request: Value, parallel: Option<bool>| {
            SymphonySpec::for_chat(qwen3(), &chat_spec(request), parallel, PROMPT, None, None)
                .expect("armed")
                .policy
        };
        let tools = json!([tool("get_weather")]);
        let base = json!({"model": "m", "messages": []});
        let with = |extra: Value| {
            let mut request = base.clone();
            for (key, value) in extra.as_object().expect("object") {
                request[key] = value.clone();
            }
            request
        };
        assert_eq!(
            armed(with(json!({"tools": tools})), None),
            Policy {
                calls: true,
                separate_reasoning: true,
                first_call_only: false,
                history_tool_calls_count: 0,
            }
        );
        assert!(!armed(base.clone(), None).calls);
        assert!(!armed(with(json!({"tools": tools, "tool_choice": "none"})), None).calls);
        assert!(!armed(with(json!({"separate_reasoning": false})), None).separate_reasoning);
        assert!(armed(with(json!({"tools": tools})), Some(false)).first_call_only);
        assert!(!armed(with(json!({"tools": tools})), Some(true)).first_call_only);
    }

    #[test]
    fn a_messages_request_sets_the_policy() {
        let armed = |tool_choice: Value| {
            let mut request = json!({
                "model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
            });
            if !tool_choice.is_null() {
                request["tool_choice"] = tool_choice;
            }
            let request: CreateMessageRequest = serde_json::from_value(request).expect("request");
            SymphonySpec::for_messages(
                qwen3(),
                &MessagesResponseSpec::from(&request),
                PROMPT,
                None,
                None,
            )
            .expect("armed")
            .policy
        };
        let auto = armed(Value::Null);
        assert!(auto.calls && auto.separate_reasoning && !auto.first_call_only);
        assert!(!armed(json!({"type": "none"})).calls);
        for choice in [
            json!({"type": "auto", "disable_parallel_tool_use": true}),
            json!({"type": "any", "disable_parallel_tool_use": true}),
            json!({"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true}),
        ] {
            assert!(armed(choice.clone()).first_call_only, "{choice}");
        }
        assert!(!armed(json!({"type": "any", "disable_parallel_tool_use": false})).first_call_only);
    }

    #[test]
    fn a_whole_output_renders_reasoning_content_and_calls() {
        let output = format!("<think>\nCheck the weather.\n</think>\n\nLet me look.\n{CALL}");
        assert_eq!(
            message(&spec(policy()), &output, "stop"),
            json!({
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nLet me look.\n",
                    "refusal": null,
                    "annotations": [],
                    "reasoning_content": "\nCheck the weather.\n",
                    "tool_calls": [{
                        "id": "functions.get_weather:0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                    }],
                },
                "logprobs": null,
                "finish_reason": "tool_calls",
            })
        );
    }

    #[test]
    fn call_ids_come_from_the_gateway_scheme() {
        let history = Policy {
            history_tool_calls_count: 3,
            ..policy()
        };
        let events = Choice::whole(&spec(history), KIMI, &format!("{CALL}{CALL}"), "stop");
        let ids: Vec<_> = chat_message(0, &events)
            .message
            .tool_calls
            .expect("calls")
            .into_iter()
            .map(|call| call.id)
            .collect();
        assert_eq!(ids, ["functions.get_weather:3", "functions.get_weather:4"]);

        let events = Choice::whole(&spec(policy()), "Qwen/Qwen3-8B", CALL, "stop");
        let id = &chat_message(0, &events).message.tool_calls.expect("calls")[0].id;
        assert!(id.starts_with("call_") && id.len() == 29, "{id}");
    }

    #[test]
    fn a_call_the_request_did_not_ask_for_is_content_without_its_markers() {
        let no_calls = Policy {
            calls: false,
            ..policy()
        };
        let reply = message(&spec(no_calls), &format!("Sure.\n{CALL}"), "stop");
        assert_eq!(
            reply["message"]["content"],
            "Sure.\n\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"
        );
        assert!(reply["message"]["tool_calls"].is_null());
        assert_eq!(reply["finish_reason"], "stop");
    }

    #[test]
    fn without_reasoning_separation_the_output_is_the_raw_text_less_returned_calls() {
        let joined = Policy {
            separate_reasoning: false,
            ..policy()
        };
        // No call: every byte the model wrote, the markers included.
        let output = "<think>\nplan\n</think>\n\nanswer";
        let reply = message(&spec(joined), output, "stop");
        assert_eq!(reply["message"]["content"], output);
        assert!(reply["message"]["reasoning_content"].is_null());
        // A returned call takes its text out, the markers that open and close it included.
        let output = format!("<think>\nplan\n</think>\n\nLet me look.\n{CALL}\nDone.");
        let reply = message(&spec(joined), &output, "stop");
        assert_eq!(
            reply["message"]["content"],
            "<think>\nplan\n</think>\n\nLet me look.\n\nDone."
        );
        assert_eq!(
            reply["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\": \"Paris\"}"
        );
        assert_eq!(reply["finish_reason"], "tool_calls");
        // A request that lets the model call no tool gets the call's text too, as written.
        let raw = Policy {
            calls: false,
            ..joined
        };
        let output = format!("<think>\nplan\n</think>\n\nSure.\n{CALL}");
        let reply = message(&spec(raw), &output, "stop");
        assert_eq!(reply["message"]["content"], output);
        assert!(reply["message"]["tool_calls"].is_null());
        // The same streamed, at every chunking.
        for (policy, output) in [
            (
                joined,
                format!("<think>\nplan\n</think>\n\nLet me look.\n{CALL}\nDone."),
            ),
            (raw, format!("<think>\nplan\n</think>\n\nSure.\n{CALL}")),
        ] {
            let whole = message(&spec(policy), &output, "stop");
            for pieces in cuts(&output) {
                let mut choice = Choice::new(&spec(policy), KIMI);
                let mut choices = Vec::new();
                for piece in &pieces {
                    choices.extend(chat_deltas(0, &choice.delta(piece), None));
                }
                let end = choice.end(Some("stop"));
                choices.extend(chat_deltas(0, &end, None));
                let finished = end
                    .iter()
                    .find_map(|event| adapt::chat::delta(0, event)?.finish_reason);
                assert_eq!(fold(&choices, finished), whole, "{pieces:?}");
            }
        }
    }

    #[test]
    fn without_reasoning_separation_a_call_takes_only_its_own_bytes_out() {
        // A returned call takes out what its events carry, the whitespace and tags inside it,
        // the marker that opened it and the one that ended it; the block the calls sit in, and
        // every other marker, stay in the raw text where the model wrote them.
        let joined = Policy {
            separate_reasoning: false,
            ..policy()
        };
        for (row, call, raw) in [
            (
                "symphony:hy4",
                format!(
                    "{HY4_CALL}<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>\
                     Paris</arg_value:opensource>\n<arg_key:opensource>days</arg_key:opensource>\
                     <arg_value:opensource>3</arg_value:opensource></tool_call:opensource>\
                     </tool_calls:opensource>"
                ),
                "Sure.<tool_calls:opensource></tool_calls:opensource>Done.",
            ),
            (
                "symphony:deepseek_v4.1",
                format!(
                    "{DSML_CALL}<｜DSML｜ parameter name=\"city\" string=\"true\">Paris\
                     </｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"days\" string=\"false\">3\
                     </｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
                ),
                "Sure.<｜DSML｜ calls>\n\n</｜DSML｜ calls>Done.",
            ),
            (
                "symphony:qwen3",
                "\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \
                 \"Paris\", \"days\": 3}}\n</tool_call>\n"
                    .to_string(),
                "Sure.\n\nDone.",
            ),
        ] {
            let spec = SymphonySpec::new(table(row).expect("row"), &[typed_tool()], PROMPT, joined);
            let output = format!("Sure.{call}Done.");
            let reply = message(&spec, &output, "stop");
            assert_eq!(reply["message"]["content"], raw, "{row}: {reply}");
            assert_eq!(
                reply["message"]["tool_calls"][0]["function"]["arguments"],
                "{\"city\": \"Paris\", \"days\": 3}",
                "{row}"
            );
            // The same at every chunking.
            for pieces in cuts(&output) {
                let mut choice = Choice::new(&spec, KIMI);
                let mut events: Vec<Event> = Vec::new();
                for piece in &pieces {
                    events.extend(choice.delta(piece));
                }
                events.extend(choice.end(Some("stop")));
                assert_eq!(
                    chat_message(0, &events).message.content.as_deref(),
                    Some(raw),
                    "{row} {pieces:?}"
                );
            }
        }
    }

    #[test]
    fn a_turn_with_a_call_cut_short_beside_a_whole_one_makes_no_call() {
        // Two calls, the second cut inside its value: both are returned as they arrived, the
        // first whole, but the turn is no tool turn, whatever ended it, a backend's own
        // `tool_calls` finish included, so that a client runs neither.
        let output = format!(
            "{CALL}\n<tool_call>\n{{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"Lon"
        );
        for (finish, expected) in [
            ("abort", "abort"),
            ("length", "length"),
            ("stop", "stop"),
            (STOP_SEQUENCE, "stop"),
            ("tool_calls", "stop"),
        ] {
            let reply = message(&spec(policy()), &output, finish);
            assert_eq!(reply["finish_reason"], expected, "{finish}");
            let calls = reply["message"]["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 2, "{finish}");
            assert_eq!(calls[0]["function"]["arguments"], "{\"city\": \"Paris\"}");
            assert_eq!(calls[1]["function"]["arguments"], "{\"city\": \"Lon");
            for pieces in cuts(&output) {
                let mut choice = Choice::new(&spec(policy()), KIMI);
                for piece in &pieces {
                    let _ = choice.delta(piece);
                }
                let end = choice.end(Some(finish));
                assert_eq!(choice.calls(), 0, "{finish} {pieces:?}");
                assert!(choice.cut(), "{finish} {pieces:?}");
                assert!(
                    end.iter()
                        .any(|event| matches!(event, Event::Finish { tool_calls: 0, .. })),
                    "{finish} {pieces:?}"
                );
            }
        }
        // Both whole: two calls made.
        let reply = message(&spec(policy()), &format!("{CALL}\n{CALL}"), "stop");
        assert_eq!(reply["finish_reason"], "tool_calls");
    }

    #[test]
    fn a_messages_stream_whose_call_text_interrupted_makes_no_call() {
        // Text out of place between a call's arguments closes its tool_use block in the stream,
        // and the later fragments become text: the streamed input is not whole JSON, so the
        // stream makes no call, where the whole message gives every fragment to the call.
        let hy4 = SymphonySpec::new(
            table("symphony:hy4").expect("row"),
            &[typed_tool()],
            PROMPT,
            policy(),
        );
        let output = format!(
            "{HY4_CALL}<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Paris\
             </arg_value:opensource>stray words<arg_key:opensource>days</arg_key:opensource>\
             <arg_value:opensource>3</arg_value:opensource></tool_call:opensource>\
             </tool_calls:opensource>"
        );
        let whole = messages_output(&Choice::whole(&hy4, KIMI, &output, "stop"));
        assert_eq!(whole.stop_reason, Some(messages::StopReason::ToolUse));
        let mut stream = MessagesStream::new(&hy4, KIMI);
        let mut events = stream.delta(&output);
        events.extend(stream.end(Some("stop")));
        assert!(stream.cut());
        assert_eq!(stream.calls(), 0);
        let blocks = fold_blocks(&events);
        assert!(
            blocks.iter().any(|block| block["type"] == "tool_use"),
            "{blocks:?}"
        );
    }

    #[test]
    fn a_stop_sequence_ends_the_output_without_closing_a_call() {
        // A stop sequence (the matched string) is handed to the table as a finish it does not
        // interpret, a stop token (its id) as `Stop` when no tokenizer named the stop tokens
        // that end a call; the finish the client reads is `stop`.
        let unnamed = spec(policy());
        assert_eq!(
            unnamed.finish_of("stop", Some(&json!("Par"))),
            STOP_SEQUENCE
        );
        assert_eq!(unnamed.finish_of("stop", Some(&json!(151675))), "stop");
        assert_eq!(unnamed.finish_of("stop", None), "stop");
        assert_eq!(unnamed.finish_of("length", Some(&json!("Par"))), "length");
        assert_eq!(
            engine_finish(Some(STOP_SEQUENCE)),
            EngineFinish::Other(STOP_SEQUENCE.to_string())
        );
        let cut = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"";
        let reply = message(&spec(policy()), cut, STOP_SEQUENCE);
        assert_eq!(reply["finish_reason"], "stop");
        assert_eq!(
            reply["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\": \""
        );
        let events = Choice::whole(&spec(policy()), KIMI, cut, STOP_SEQUENCE);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                ..
            }
        )));
        // After a call that ended, the turn still made it.
        let reply = message(&spec(policy()), &format!("{CALL}\nand then"), STOP_SEQUENCE);
        assert_eq!(reply["finish_reason"], "tool_calls");
    }

    /// A table whose call ends at `<|im_end|>`, a token of the mock tokenizer's vocabulary, and
    /// whose stream's stop closes nothing: the call-ending stop token is looked up by the spec.
    fn ends_at_im_end() -> Format {
        Format::new("ends at im_end")
            .terminal("open", "<tool_call>")
            .terminal("close", "<|im_end|>")
            .state("content", symphony::Emits::Content)
            .state("call", symphony::Emits::Arguments)
            .transition("content", "open", "call")
            .transition("call", "close", "content")
            .calls(CallSyntax::Json)
    }

    static ENDS_AT_IM_END: Table = Table {
        name: "ends_at_im_end",
        format: ends_at_im_end,
        tool_parser: None,
        reasoning_parser: None,
    };

    #[test]
    fn a_stop_token_that_ends_no_call_is_a_finish_that_closes_none() {
        // The tokenizer names the table's call-ending markers that are one token each; a stop on
        // any other token, the answer's stop token written inside a call among them, reaches the
        // table as a finish it does not interpret, so the call it stopped in stays cut. The
        // client reads `stop` either way.
        let tokenizer = llm_tokenizer::MockTokenizer::new();
        let spec = SymphonySpec::new(&ENDS_AT_IM_END, &[tool("get_weather")], "", policy())
            .with_call_stops(Some(&tokenizer as &dyn Tokenizer));
        assert_eq!(spec.call_stops, Some(vec![1002]));
        assert_eq!(spec.finish_of("stop", Some(&json!(1002))), "stop");
        assert_eq!(spec.finish_of("stop", Some(&json!(999))), STOP_TOKEN);
        assert_eq!(spec.finish_of("stop", Some(&json!(-1))), STOP_TOKEN);
        assert_eq!(spec.finish_of("stop", Some(&json!("Par"))), STOP_SEQUENCE);
        assert_eq!(spec.finish_of("stop", None), "stop");
        assert_eq!(spec.finish_of("length", Some(&json!(999))), "length");
        assert_eq!(
            engine_finish(Some(STOP_TOKEN)),
            EngineFinish::Other(STOP_TOKEN.to_string())
        );
        // A table whose call-ending markers the vocabulary lacks: every stop token ends no call.
        let qwen = spec_with(qwen3(), policy()).with_call_stops(Some(&tokenizer as &dyn Tokenizer));
        assert_eq!(qwen.call_stops, Some(vec![]));
        assert_eq!(qwen.finish_of("stop", Some(&json!(1002))), STOP_TOKEN);
        // Without a tokenizer every stop token may end a call, as before.
        assert_eq!(
            spec_with(qwen3(), policy()).finish_of("stop", Some(&json!(1002))),
            "stop"
        );
        // The finish the client reads, and the call the stop token cut.
        let cut = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par";
        let events = Choice::whole(&spec, KIMI, cut, STOP_TOKEN);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::Finish {
                reason: FinishReason::Stop,
                tool_calls: 0,
                ..
            }
        )));
        let reply = serde_json::to_value(chat_message(0, &events)).expect("serializable");
        assert_eq!(reply["finish_reason"], "stop");
        assert_eq!(
            reply["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\": \"Par"
        );
    }

    #[test]
    fn without_reasoning_separation_the_calls_parallel_calls_off_drops_are_text() {
        // Only the first call is returned; without reasoning separation every byte of the calls
        // after it is content as written, their markers included, as for a call the request
        // does not let the model make.
        let raw_first = Policy {
            separate_reasoning: false,
            first_call_only: true,
            ..policy()
        };
        let output = format!("Sure.\n{CALL}\n{CALL}\nDone.");
        let reply = message(&spec(raw_first), &output, "stop");
        assert_eq!(
            reply["message"]["content"],
            format!("Sure.\n\n{CALL}\nDone."),
            "{reply}"
        );
        let calls = reply["message"]["tool_calls"]
            .as_array()
            .expect("the first call");
        assert_eq!(calls.len(), 1);
        assert_eq!(reply["finish_reason"], "tool_calls");
        for pieces in cuts(&output) {
            let mut choice = Choice::new(&spec(raw_first), KIMI);
            let mut events: Vec<Event> = Vec::new();
            for piece in &pieces {
                events.extend(choice.delta(piece));
            }
            events.extend(choice.end(Some("stop")));
            let message = chat_message(0, &events).message;
            assert_eq!(
                message.content.as_deref(),
                Some(format!("Sure.\n\n{CALL}\nDone.").as_str()),
                "{pieces:?}"
            );
            assert_eq!(message.tool_calls.map(|calls| calls.len()), Some(1));
        }
        // With reasoning separated, the dropped call is not shown, as before.
        let first = Policy {
            first_call_only: true,
            ..policy()
        };
        let reply = message(&spec(first), &output, "stop");
        assert_eq!(reply["message"]["content"], "Sure.\n\n\nDone.", "{reply}");
    }

    #[test]
    fn only_the_first_call_is_returned_when_parallel_calls_are_off() {
        let first = Policy {
            first_call_only: true,
            ..policy()
        };
        let mut choice = Choice::new(&spec(first), KIMI);
        let mut events = choice.delta(&format!("{CALL}\n{CALL}"));
        events.extend(choice.end(Some("stop")));
        assert_eq!(choice.calls(), 1);
        let calls = chat_message(0, &events).message.tool_calls.expect("calls");
        assert_eq!(calls.len(), 1);
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::Finish { tool_calls: 1, .. })));
    }

    #[test]
    fn the_end_reason_reaches_the_table() {
        let cut = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par";
        let length = message(&spec(policy()), cut, "length");
        assert_eq!(length["finish_reason"], "length");
        assert_eq!(
            length["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\": \"Par"
        );
        // A call cut short is returned as it arrived but not made: a stop stays a stop.
        assert_eq!(
            message(&spec(policy()), cut, "stop")["finish_reason"],
            "stop"
        );
        assert_eq!(
            message(&spec(policy()), "hi", "abort")["finish_reason"],
            "abort"
        );
        assert_eq!(
            message(&spec(policy()), "hi", "content_filter")["finish_reason"],
            "content_filter"
        );
        assert_eq!(engine_finish(None), EngineFinish::Abort);
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let thinking = SymphonySpec::new(qwen3(), &[], format!("{PROMPT}<think>\n"), policy());
        let reply = message(&thinking, "plan\n</think>\n\nanswer", "stop");
        assert_eq!(reply["message"]["reasoning_content"], "plan\n");
        assert_eq!(reply["message"]["content"], "\n\nanswer");
        // A marker quoted in an earlier turn moves nothing.
        let quoted = SymphonySpec::new(
            qwen3(),
            &[],
            "<|im_start|>user\nWhat does <think> do?<|im_end|>\n<|im_start|>assistant\n",
            policy(),
        );
        assert_eq!(
            message(&quoted, "It opens a thought.", "stop")["message"]["content"],
            "It opens a thought."
        );
    }

    #[test]
    fn malformed_text_is_content() {
        let reply = message(
            &spec(policy()),
            "<tool_call>\n{\"name\": \"get_weather\"} trailing\n</tool_call>",
            "stop",
        );
        assert!(reply["message"]["content"]
            .as_str()
            .is_some_and(|content| content.contains("trailing")));
    }

    #[test]
    fn a_call_without_arguments_has_an_empty_object() {
        let with_member =
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>";
        for output in [BARE_CALL, with_member] {
            let reply = message(&spec(policy()), output, "stop");
            let calls = reply["message"]["tool_calls"].as_array().expect("calls");
            assert_eq!(calls.len(), 1, "{output}");
            assert_eq!(calls[0]["function"]["arguments"], "{}", "{output}");
        }
        // Only a returned call gets one: a call that is content keeps the model's bytes.
        let no_calls = Policy {
            calls: false,
            ..policy()
        };
        let reply = message(&spec(no_calls), BARE_CALL, "stop");
        assert_eq!(reply["message"]["content"], "\n{\"name\": \"get_weather\"}");
        let output = messages_output(&Choice::whole(&spec(policy()), KIMI, BARE_CALL, "stop"));
        assert_eq!(
            serde_json::to_value(&output.content).expect("blocks")[0]["input"],
            json!({})
        );
    }

    /// A tool with a string and an integer parameter: a tagged call streams the string and
    /// writes the integer at its close.
    fn typed_tool() -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}
            }}}
        }))
        .expect("tool")
    }

    fn typed_spec(row: &str) -> SymphonySpec {
        SymphonySpec::new(table(row).expect("row"), &[typed_tool()], PROMPT, policy())
    }

    const HY4_CALL: &str = "<tool_calls:opensource><tool_call:opensource>get_weather";
    const DSML_CALL: &str = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n";

    /// Calls the end of the stream cuts before their first argument fragment: inside a key or
    /// an integer value of a keyed call, right after a DSML invoke's tag or inside its integer
    /// parameter. (The JSON syntax names no call before its arguments begin or its object
    /// closes; cut there, its bytes are content.)
    fn cut_before_arguments() -> Vec<(&'static str, String)> {
        vec![
            ("symphony:hy4", format!("{HY4_CALL}<arg_key:opensource>da")),
            (
                "symphony:hy4",
                format!(
                    "{HY4_CALL}<arg_key:opensource>days</arg_key:opensource>\
                     <arg_value:opensource>3"
                ),
            ),
            ("symphony:deepseek_v4.1", DSML_CALL.to_string()),
            (
                "symphony:deepseek_v4.1",
                format!("{DSML_CALL}<｜DSML｜ parameter name=\"days\" string=\"false\">3"),
            ),
        ]
    }

    #[test]
    fn a_call_the_end_cuts_before_its_arguments_stays_open_and_is_not_a_call_made() {
        // The call is returned as written, with no arguments: `{}` would read as a complete
        // call. Since nothing says the model made it, a stop stays a stop.
        for (row, output) in cut_before_arguments() {
            let spec = typed_spec(row);
            for finish in ["length", "stop", "abort"] {
                let reply = message(&spec, &output, finish);
                let calls = &reply["message"]["tool_calls"];
                assert_eq!(
                    calls.as_array().map(Vec::len),
                    Some(1),
                    "{row} {finish}: {reply}"
                );
                assert_eq!(calls[0]["function"]["name"], "get_weather");
                assert_eq!(
                    calls[0]["function"]["arguments"], "",
                    "{row} {finish}: {reply}"
                );
                assert_eq!(reply["finish_reason"], finish, "{row} {finish}: {output}");
                let mut choice = Choice::new(&spec, KIMI);
                choice.delta(&output);
                choice.end(Some(finish));
                assert_eq!(choice.calls(), 0, "{row} {finish}: {output}");
                // Streamed at every chunking, the same message: no chunking gives `{}`.
                for pieces in cuts(&output) {
                    let mut choice = Choice::new(&spec, KIMI);
                    let mut choices = Vec::new();
                    for piece in &pieces {
                        choices.extend(chat_deltas(0, &choice.delta(piece), None));
                    }
                    let end = choice.end(Some(finish));
                    choices.extend(chat_deltas(0, &end, None));
                    let finished = end
                        .iter()
                        .find_map(|event| adapt::chat::delta(0, event)?.finish_reason);
                    assert_eq!(fold(&choices, finished), reply, "{row} {pieces:?}");
                }
            }
        }
    }

    #[test]
    fn a_call_cut_after_its_first_argument_keeps_what_arrived_and_is_not_a_call_made() {
        // Its arguments are not a complete object: no `stop` or abort turns it into a call made.
        for (row, output, arguments) in [
            (
                "symphony:qwen3",
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par"
                    .to_string(),
                "{\"city\": \"Par",
            ),
            (
                "symphony:hy4",
                format!(
                    "{HY4_CALL}<arg_key:opensource>city</arg_key:opensource>\
                     <arg_value:opensource>Par"
                ),
                "{\"city\": \"Par",
            ),
        ] {
            let spec = typed_spec(row);
            for finish in ["length", "stop", "abort"] {
                let reply = message(&spec, &output, finish);
                assert_eq!(
                    reply["message"]["tool_calls"][0]["function"]["arguments"], arguments,
                    "{row} {finish}"
                );
                assert_eq!(reply["finish_reason"], finish, "{row} {finish}");
                let mut choice = Choice::new(&spec, KIMI);
                choice.delta(&output);
                let events = choice.end(Some(finish));
                assert_eq!(choice.calls(), 0, "{row} {finish}");
                assert!(events
                    .iter()
                    .any(|event| matches!(event, Event::Finish { tool_calls: 0, .. })));
            }
        }
    }

    #[test]
    fn a_call_that_ends_without_arguments_has_an_empty_object_on_every_table() {
        // Before the end of the stream: the table's own object, or the driver's `{}` for a
        // JSON call written without an arguments member.
        for (row, output) in [
            ("symphony:qwen3", BARE_CALL.to_string()),
            (
                "symphony:hy4",
                format!("{HY4_CALL}</tool_call:opensource></tool_calls:opensource>"),
            ),
            (
                "symphony:deepseek_v4.1",
                format!("{DSML_CALL}</｜DSML｜ invoke>\n</｜DSML｜ calls>"),
            ),
        ] {
            let spec = typed_spec(row);
            for (finish, reason) in [("stop", "tool_calls"), ("length", "length")] {
                let reply = message(&spec, &output, finish);
                assert_eq!(
                    reply["message"]["tool_calls"][0]["function"]["arguments"], "{}",
                    "{row} {finish}"
                );
                assert_eq!(reply["finish_reason"], reason, "{row} {finish}");
                let mut choice = Choice::new(&spec, KIMI);
                choice.delta(&output);
                choice.end(Some(finish));
                assert_eq!(choice.calls(), 1, "{row} {finish}");
            }
        }
    }

    #[test]
    fn the_end_is_taken_once_and_closes_the_choice() {
        let mut choice = Choice::new(&spec(policy()), KIMI);
        assert!(!choice.delta("hi").is_empty());
        assert!(matches!(
            choice.end(Some("stop")).last(),
            Some(Event::Finish { .. })
        ));
        assert!(choice.end(Some("stop")).is_empty());
        assert!(choice.delta("more").is_empty());
    }

    /// The outputs the chunking tests cut at every place.
    fn corpus() -> Vec<(String, &'static str, Policy)> {
        let no_calls = Policy {
            calls: false,
            ..policy()
        };
        let joined = Policy {
            separate_reasoning: false,
            ..policy()
        };
        let first = Policy {
            first_call_only: true,
            ..policy()
        };
        vec![
            (
                format!("<think>\nplan\n</think>\n\nLet me look.\n{CALL}"),
                "stop",
                policy(),
            ),
            (format!("{CALL}\n{CALL}"), "stop", policy()),
            (format!("{CALL}\n{CALL}"), "stop", first),
            (format!("Sure.\n{CALL}"), "stop", no_calls),
            (
                "<think>\nplan\n</think>\n\nanswer".to_string(),
                "stop",
                joined,
            ),
            (CALL[..CALL.len() - 20].to_string(), "length", policy()),
            ("answer <tool_".to_string(), "stop", policy()),
            (
                format!("{BARE_CALL}\n{CALL}\n{BARE_CALL}"),
                "stop",
                policy(),
            ),
            (format!("{BARE_CALL}\n{BARE_CALL}"), "stop", first),
        ]
    }

    /// Every cut into two deltas, and one delta per character.
    fn cuts(output: &str) -> Vec<Vec<&str>> {
        let mut cuts: Vec<Vec<&str>> = output
            .char_indices()
            .map(|(at, _)| vec![&output[..at], &output[at..]])
            .collect();
        cuts.push(
            output
                .char_indices()
                .map(|(at, c)| &output[at..at + c.len_utf8()])
                .collect(),
        );
        cuts
    }

    /// A Chat stream folded into the message it describes.
    fn fold(choices: &[ChatStreamChoice], finish: Option<String>) -> Value {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<Value> = Vec::new();
        for choice in choices {
            assert!(
                choice.finish_reason.is_none(),
                "the envelope sends the finish"
            );
            content.push_str(choice.delta.content.as_deref().unwrap_or_default());
            reasoning.push_str(
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .unwrap_or_default(),
            );
            for call in choice.delta.tool_calls.iter().flatten() {
                let function = call.function.as_ref().expect("function");
                match &call.id {
                    Some(id) => calls.push(json!({
                        "id": id, "type": "function",
                        "function": {"name": function.name, "arguments": ""},
                    })),
                    None => {
                        let arguments = &mut calls[call.index as usize]["function"]["arguments"];
                        *arguments = Value::String(format!(
                            "{}{}",
                            arguments.as_str().unwrap_or_default(),
                            function.arguments.as_deref().unwrap_or_default()
                        ));
                    }
                }
            }
        }
        // The message's always-present fields, as the whole message carries them.
        let mut message = json!({"role": "assistant", "refusal": null, "annotations": []});
        message["content"] = if content.trim().is_empty() {
            Value::Null
        } else {
            Value::String(content)
        };
        message["reasoning_content"] = if reasoning.is_empty() {
            Value::Null
        } else {
            Value::String(reasoning)
        };
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        json!({"index": 0, "message": message, "logprobs": null, "finish_reason": finish})
    }

    #[test]
    fn a_chat_stream_says_what_the_whole_message_says_at_every_chunking() {
        for (output, finish, policy) in corpus() {
            let spec = spec(policy);
            let whole = message(&spec, &output, finish);
            for pieces in cuts(&output) {
                let mut choice = Choice::new(&spec, KIMI);
                let mut choices = Vec::new();
                let mut finished = None;
                for piece in &pieces {
                    choices.extend(chat_deltas(0, &choice.delta(piece), None));
                }
                let end = choice.end(Some(finish));
                choices.extend(chat_deltas(0, &end, None));
                for event in &end {
                    if let Some(choice) = adapt::chat::delta(0, event) {
                        finished = finished.or(choice.finish_reason);
                    }
                }
                assert_eq!(fold(&choices, finished), whole, "{pieces:?}");
            }
        }
    }

    #[test]
    fn a_chunk_s_logprobs_ride_on_its_first_content() {
        let logprobs = ChatLogProbs::Detailed {
            content: None,
            refusal: None,
        };
        let mut choice = Choice::new(&spec(policy()), KIMI);
        let choices = chat_deltas(
            0,
            &choice.delta("<think>plan</think>one"),
            Some(logprobs.clone()),
        );
        let with: Vec<_> = choices
            .iter()
            .map(|choice| (choice.delta.content.is_some(), choice.logprobs.is_some()))
            .collect();
        assert_eq!(with, [(false, false), (true, true)]);
        // A chunk with no content carries them on its reasoning, as the legacy stream does.
        let mut choice = Choice::new(&spec(policy()), KIMI);
        let choices = chat_deltas(0, &choice.delta("<think>plan"), Some(logprobs.clone()));
        let with: Vec<_> = choices
            .iter()
            .map(|choice| {
                (
                    choice.delta.reasoning_content.is_some(),
                    choice.logprobs.is_some(),
                )
            })
            .collect();
        assert_eq!(with, [(true, true)]);
    }

    /// A Messages stream folded into the blocks it describes.
    fn fold_blocks(events: &[MessageStreamEvent]) -> Vec<Value> {
        let mut blocks: Vec<Value> = Vec::new();
        let mut arguments: Vec<String> = Vec::new();
        for event in events {
            match event {
                MessageStreamEvent::ContentBlockStart { content_block, .. } => {
                    blocks.push(serde_json::to_value(content_block).expect("block"));
                    arguments.push(String::new());
                }
                MessageStreamEvent::ContentBlockDelta { delta, .. } => {
                    let block = blocks.last_mut().expect("an open block");
                    match delta {
                        messages::ContentBlockDelta::TextDelta { text } => {
                            let joined =
                                format!("{}{text}", block["text"].as_str().unwrap_or_default());
                            block["text"] = Value::String(joined);
                        }
                        messages::ContentBlockDelta::ThinkingDelta { thinking } => {
                            let joined = format!(
                                "{}{thinking}",
                                block["thinking"].as_str().unwrap_or_default()
                            );
                            block["thinking"] = Value::String(joined);
                        }
                        messages::ContentBlockDelta::InputJsonDelta { partial_json } => {
                            arguments
                                .last_mut()
                                .expect("arguments")
                                .push_str(partial_json);
                        }
                        other => panic!("unexpected delta {other:?}"),
                    }
                }
                MessageStreamEvent::ContentBlockStop { .. } => {
                    let block = blocks.last_mut().expect("an open block");
                    if block["type"] == "tool_use" {
                        block["input"] = serde_json::from_str(arguments.last().expect("arguments"))
                            .unwrap_or_else(|_| json!({}));
                    }
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        blocks
    }

    #[test]
    fn a_messages_stream_builds_the_whole_message_s_blocks_at_every_chunking() {
        // One reasoning region, then text, then nothing but calls: the stream's blocks are the
        // fold's (the adapter's contract; text between calls is where the two differ).
        for (output, finish) in [
            (
                format!("<think>\nplan\n</think>\n\nLet me look.\n{CALL}"),
                "stop",
            ),
            (format!("Sure.\n{CALL}{CALL}"), "stop"),
            ("answer".to_string(), "length"),
            (format!("{BARE_CALL}{CALL}"), "stop"),
        ] {
            let spec = spec(policy());
            let whole = messages_output(&Choice::whole(&spec, KIMI, &output, finish));
            let whole: Vec<Value> = whole
                .content
                .iter()
                .map(|block| serde_json::to_value(block).expect("block"))
                .collect();
            for pieces in cuts(&output) {
                let mut stream = MessagesStream::new(&spec, KIMI);
                let mut events = Vec::new();
                for piece in &pieces {
                    events.extend(stream.delta(piece));
                }
                events.extend(stream.end(Some(finish)));
                assert_eq!(fold_blocks(&events), whole, "{pieces:?}");
            }
        }
    }

    #[test]
    fn a_messages_stream_counts_its_calls_and_closes_its_last_block() {
        let mut stream = MessagesStream::new(&spec(policy()), KIMI);
        let mut events = stream.delta(&format!("Sure.\n{CALL}"));
        assert_eq!(stream.calls(), 1);
        events.extend(stream.end(None));
        assert!(matches!(
            events.last(),
            Some(MessageStreamEvent::ContentBlockStop { index: 1 })
        ));
        assert!(stream.end(Some("stop")).is_empty());
    }
}
