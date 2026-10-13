//! A format is data: the terminals a model writes between its text, the states that text falls
//! into, the transitions the terminals make between them, and the syntax of a call. The [`Engine`]
//! runs one; a family's definition is a function under [`formats`] today and a TOML file later.
//!
//! The table is the design's section 4. What it says:
//!
//! - A **terminal** is a spelling the model writes, `<think>` or `</tool_call>`, named so that a
//!   transition can speak of it. Token ids come in a later step; today a terminal is its text.
//! - A **state** says what the text inside it is: content, reasoning, a call's arguments, or the
//!   template's wrapping between calls. Text in a content state is `Content`; in a reasoning
//!   state `Reasoning`; in an arguments state it is fed to the call syntax's assembler and becomes
//!   the call's events; in a wrapper state whitespace is dropped and anything else is malformed.
//! - A **transition** `from + terminal = to` moves the engine between states when the terminal
//!   arrives in `from`. A terminal with no transition from the current state is text, where the
//!   model put it: `</think>` in content is content. Entering a reasoning state pushes
//!   `ReasoningStart` and leaving one `ReasoningEnd`; entering an arguments state opens a call and
//!   leaving one ends it, so `calls + call_open = calls` ends the call that was open and starts
//!   the next. The terminal itself is `Dropped { Wrapper }` on a transition.
//! - The **call syntax** says how the model writes a call inside an arguments state, and the
//!   assembler for it: a family's, or a call given as data ([`CallSyntax::Shaped`]), whose
//!   terminal opens a call only when the header after it matches. A format with no arguments
//!   state has none.
//! - An **ignored terminal** is a spelling the engine drops wherever it stands, a call's arguments
//!   included, without moving: the separator token MiniMax M3's template writes before every tag.
//!   No transition can name it.
//! - A **stop row** names an arguments state whose call the stream's stop closes. A family whose
//!   last call ends with a stop token writes no other marker after it, and the stop decoder hides
//!   that token, so without the row the call would read as cut short. With it, a stream that
//!   stops in that state ends the call as its closing marker would, with no bytes; a stream cut
//!   for length, or aborted, is still cut, and so is one that ended with any finish but `Stop`.
//!   `EngineFinish::Stop` also covers a stop sequence of the request's, which the row would take
//!   for the stop token and close the call on, with whatever arguments had arrived; a driver that
//!   knows a stop sequence ended the stream hands the end over as another finish
//!   (`EngineFinish::Other`), and the call stays cut, since a stop sequence never completes a
//!   call. A table read from a response template may name it for any call, since the template's
//!   reader closes a region at the end of its text whatever the region's closing pattern: a call
//!   given as data closes there as [`CallSyntax::Shaped`] says.
//! - A **framing row** names a content or wrapper state whose text the template discards: the
//!   space around a response template's fields, which Hugging Face's parser drops (its null sink),
//!   or a field the API has no place for. Every byte of text there, a terminal with no row from
//!   the state included, is `Dropped { Wrapper }`, and so is what the scanner still held there
//!   when the stream ended, at every finish: neither the client nor `Malformed` sees it. The
//!   state's rows move the engine as before.
//! - A **trim row** names a content or reasoning state whose text is trimmed stay by stay, as
//!   Hugging Face's `text` parser strips a field's value (`strip`, on unless a template turns it
//!   off). A stay is the text between the row that enters the state and the row that leaves it,
//!   a row back into the same state ending one stay and starting the next. Whitespace before the
//!   stay's first other character is `Dropped { Whitespace }`; a run of it after the last one is
//!   held until more text in the stay shows it was inside the text, or the stay ends and drops it.
//!   Whitespace is what Python's `str.strip` removes: Unicode's white space and the separators
//!   U+001C to U+001F.
//! - A **join row** names a content or reasoning state whose stays the template joins with a text
//!   (Hugging Face's `join`): a later stay shows the joiner first, so the stays read as one value
//!   with the joiner between them. A stay of the turn's own state counts once its first byte
//!   arrives, as an implicit field opens; a stay of any other state counts from the row that
//!   enters it, as a field with an opener does, even when nothing follows. The joiner is the one
//!   text an event shows that the model did not write: `Events::text` holds it where the stay it
//!   joins starts, and a table without the row conserves the output byte for byte. With no row,
//!   or an empty joiner, stays follow one another as they are.
//! - The **stop tokens** are the terminals that are tokens the engine stops on (Harmony's
//!   `<|call|>`): the stop decoder hides them, so the engine never reads one, and a constraint
//!   derived from the table can end a region one of them closes with no text of its own, since
//!   the model's stop ends the region there ([`Format::is_stop_token`]).
//!
//! The first state in the table is where an output starts, unless the prompt moved the engine
//! before the output began: the engine replays the prompt's terminals over the same table, from
//! the last **turn opener** the table names (`<|im_start|>assistant` for Qwen's ChatML), so that
//! a marker quoted in an earlier turn moves nothing. A table that names no opener is replayed
//! from the prompt's start.
//!
//! A table read from a response template names its **turn anchors** instead, as Hugging Face's
//! `start_anchor` does: one literal or several (Gemma 4's template starts a turn after
//! `<|turn>model\n` or after `<tool_response|>`). The anchors are found as a regular expression's
//! `finditer` finds an alternation of them: from the prompt's start, the earliest match, of the
//! anchors that match there the first in the table's order, then on from that match's end, so
//! that matches never overlap; the last match is the turn's. The replay starts after its end, in
//! the state the anchor names: the first state, or one the anchor already entered (a template
//! whose anchor ends with the thought's opener). A prompt that holds no anchor replays nothing,
//! and the output starts in the first state. A table names a turn opener or turn anchors, not
//! both.
//!
//! A **start row** names the state a turn starts in when it does not start with a terminal: a
//! response template's open with a `^` alternative (`(?:^|<think>\s*)`), which matches with no
//! text at the turn's first byte. Hugging Face's parser takes the longest delimiter at a place,
//! and a regular expression the first alternative that matches: so at the turn's first byte a
//! terminal whose row leads into another state wins, being longer than the empty match, while a
//! terminal whose row leads into the start state is text there, the `^` having matched first
//! (`<think>` at the first byte is the thought's text). Any other first piece starts the turn in
//! the start state and is read there. The turn's first byte is where the replay starts, after the
//! anchor; when the prompt leaves nothing to replay, it is the output's. A turn an anchor started
//! in a state of its own has no start to choose.
//!
//! The table holds what is static about a format. What belongs to one request, the tools that
//! type a tagged call's values, reaches the [`Engine`] with the request, not the table.
//!
//! [`Engine`]: crate::Engine
//! [`formats`]: crate::formats

use crate::{
    pythonic,
    shape::{CallShape, ValueRead, Values},
    tagged::{self, keyed},
};

/// How the model writes a call between the call markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallSyntax {
    /// One JSON object, `{"name": …, "arguments": {…}}`: Qwen3.
    Json,
    /// `<function=NAME>` and then `<parameter=KEY>` around each value's text, typed by the
    /// request's tools, which reach the engine with the request, with the family's spelling of a
    /// value that is not a string and where its template puts a value between the tags: Qwen 3.5
    /// and later and Qwen3-Coder (on a line of its own), Seed-OSS and MiMo (between the tags
    /// directly).
    Tagged(tagged::Spelling, tagged::Placement),
    /// DeepSeek's DSML: the arguments state is one `<｜DSML｜ invoke name="…">` block, whose
    /// parameter tags carry a `string` attribute that types each value. The terminal that enters
    /// the state is the invoke tag's opening, and the one that leaves it is the invoke's closing
    /// tag, which the call's end carries.
    Dsml,
    /// The call's name as text, then `<arg_key>` and `<arg_value>` pairs, in the family's spelling
    /// of the four tags, typed by the request's tools: GLM, Hy4, Ling, IQuest. The terminals that
    /// enter and leave the state are the call's own markers, and the call's end carries the
    /// closing one.
    Keyed(keyed::Tags),
    /// Python calls, `name(key=value, ...)`, one or several, in the family's dialect: framed as
    /// its template frames them (one per line: Olmo 3; a Python list: LFM2.5, Llama 3.2's
    /// pythonic template), the values as it writes them (Python's words and quotes: LFM2.5; JSON,
    /// through `tojson`: Olmo 3). The region's markers are the terminals; each call ends at its
    /// own `)`, and the region's closing marker is wrapping. The assembler reads either framing
    /// and both spellings; the grammar derived from the table writes the dialect named.
    Pythonic(pythonic::Dialect),
    /// A JSON list of calls with no markers around it, or content: xLAM. The table's one state is
    /// an arguments state entered at the output's start, and the assembler decides at the first
    /// byte that is not whitespace.
    JsonList,
    /// An XML tree of arguments: `<invoke name="…">`, then one element per argument, `<key>` to
    /// `</key>`, nested for an object and with `<item>` elements for a list, typed by the request's
    /// tools at every depth: MiniMax M3. The terminal that enters the state is the invoke tag's
    /// opening, and the one that leaves it is `</invoke>`, which the call's end carries.
    Xml,
    /// Parameters named by an attribute: the arguments state is one `<invoke name="…">` block
    /// whose `<parameter name="…">` tags carry the key in the attribute and are typed by the
    /// request's tools, in the family's spelling of the parameter tags: MiniMax M2. The terminal
    /// that enters the state is the invoke tag's opening, and the one that leaves it is
    /// `</invoke>`, which the call's end carries.
    Attributed(tagged::attributed::Tags),
    /// Kimi K3's XTML: the arguments state is one `<|open|>call tool="…" index="N"<|sep|>` block,
    /// whose argument tags carry a `type` attribute that types each value (`string`, `number`,
    /// `boolean`, `null`, `array`, `object`). The terminal that enters the state is the call tag's
    /// opening, and the one that leaves it is `<|close|>call<|sep|>`, which the call's end carries.
    Xtml,
    /// The tool's name as text, then this marker, then one JSON object `{"name": …, "args": …}`:
    /// Inkling, whose every message starts the arguments state and says what it is only after
    /// its opener (`<|content_thinking|>`, `<|content_text|>`, or the name and the marker). The
    /// terminals that leave the state are the message's own; the object's close is the call's end,
    /// and the message's closing terminal is the region's.
    NamedJson(&'static str),
    /// Gemma 4's `call:NAME{key:value,…}`: keys bare, a string between `<|"|>` tokens, numbers and
    /// words bare, lists and objects nested the same way, the keys sorted by the template. The
    /// terminals that enter and leave the state are the call's own markers, and the call's end
    /// carries the closing one.
    Gemma,
    /// A call given as data, as a response template writes one: what follows the region's
    /// terminal (the header, which may capture the tool's name), how the arguments are written,
    /// and what comes after them before the region's closing terminal (see
    /// [`CallShape`]). The terminal that enters the state opens a call only when the header after
    /// it matches; otherwise it is the text of the state it came in. A closing terminal closes the
    /// region only where the trailer ends right before it, and is the call's end; otherwise it is
    /// the call's text. The stream's stop, where [`Format::closes_at_stop`] names the state, closes
    /// one JSON value or object where the trailer ends with the stream, and tagged parameters
    /// wherever the trailer stands. The shape is [`intern`](crate::shape::intern)ed, so the syntax
    /// stays `Copy`.
    Shaped(&'static CallShape),
}

/// What the text inside a state is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emits {
    Content,
    Reasoning,
    /// A call's arguments, assembled by the format's call syntax.
    Arguments,
    /// The template's wrapping between calls: whitespace is `Dropped { Wrapper }`, anything else
    /// `Malformed`.
    Wrapper,
}

/// One format's table. Built with [`Format::new`] and the methods that add a row each; the
/// first state added is the initial one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Format {
    name: String,
    terminals: Vec<Terminal>,
    states: Vec<State>,
    transitions: Vec<Transition>,
    calls: Option<CallSyntax>,
    turn_opener: Option<String>,
    /// The turn anchors, in the order a match at one place prefers them.
    anchors: Vec<Anchor>,
    /// The state a turn starts in when it does not start with a terminal.
    start: Option<usize>,
    structural_tokens: Vec<String>,
    /// The arguments states whose call the stream's stop closes.
    stop_closes: Vec<usize>,
    /// The texts of the tokens the engine stops on that are terminals.
    stop_tokens: Vec<String>,
    /// The states whose text the template discards.
    framing: Vec<usize>,
    /// The states whose stays are trimmed.
    trimmed: Vec<usize>,
    /// The states whose stays are joined, each with its joiner.
    joiners: Vec<(usize, String)>,
    /// How a shaped call's values read.
    values: Values,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Terminal {
    name: String,
    text: String,
    /// Dropped wherever it stands, never a move.
    ignored: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    name: String,
    emits: Emits,
}

/// A literal that starts the model's turn in the prompt, and the state the output starts in after
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Anchor {
    text: String,
    state: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transition {
    from: usize,
    on: usize,
    to: usize,
}

/// A row that names a state or terminal the table does not have. Definitions are written in the
/// crate, so this is a programming error, and the builder panics with the name.
#[expect(
    clippy::panic,
    reason = "a definition that names a row it never added is a programming error in the crate"
)]
fn unknown(kind: &str, name: &str, format: &str) -> ! {
    panic!("format {format}: no {kind} named {name:?}")
}

impl Format {
    /// An empty table with a name; add terminals, states and transitions to it.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            terminals: Vec::new(),
            states: Vec::new(),
            transitions: Vec::new(),
            calls: None,
            turn_opener: None,
            anchors: Vec::new(),
            start: None,
            structural_tokens: Vec::new(),
            stop_closes: Vec::new(),
            stop_tokens: Vec::new(),
            framing: Vec::new(),
            trimmed: Vec::new(),
            joiners: Vec::new(),
            values: Values::inferred(),
        }
    }

    /// A terminal and its text spelling.
    #[must_use]
    pub fn terminal(mut self, name: &str, text: &str) -> Self {
        self.terminals.push(Terminal {
            name: name.to_string(),
            text: text.to_string(),
            ignored: false,
        });
        self
    }

    /// A terminal the engine drops wherever it stands, inside a call's arguments too, without
    /// moving: a template's separator token. No transition can name it.
    #[must_use]
    pub fn ignores(mut self, name: &str, text: &str) -> Self {
        self.terminals.push(Terminal {
            name: name.to_string(),
            text: text.to_string(),
            ignored: true,
        });
        self
    }

    /// A token of the model's vocabulary that the format's tags begin with (`<|open|>`,
    /// `<|close|>`). It reaches the engine whole or not at all, so a tail at the end of the output
    /// that begins with one and is the beginning of a terminal the state has a row for is a tag
    /// the end cut short, never the model's text: the engine drops it as it drops the whole tag,
    /// instead of releasing it as content. A tail shorter than the token stays text, as every
    /// parser keeps a partial marker at the end of the output; so does the beginning of a
    /// terminal the state keeps as text when whole.
    #[must_use]
    pub fn structural_token(mut self, text: &str) -> Self {
        self.structural_tokens.push(text.to_string());
        self
    }

    /// A state and what its text is; the first one added is where an output starts.
    #[must_use]
    pub fn state(mut self, name: &str, emits: Emits) -> Self {
        self.states.push(State {
            name: name.to_string(),
            emits,
        });
        self
    }

    /// `from + on = to`: the terminal `on`, arriving in state `from`, moves the engine to `to`.
    /// Both states and the terminal must already be in the table.
    #[must_use]
    pub fn transition(mut self, from: &str, on: &str, to: &str) -> Self {
        let transition = Transition {
            from: self.state_named(from),
            on: self.terminal_named(on),
            to: self.state_named(to),
        };
        self.transitions.push(transition);
        self
    }

    /// The syntax of a call inside an arguments state.
    #[must_use]
    pub fn calls(mut self, syntax: CallSyntax) -> Self {
        self.calls = Some(syntax);
        self
    }

    /// The arguments state `state` closes its call when the stream stops in it: the family ends
    /// its last call with a stop token, which the stop decoder hides, or a response template's
    /// reader closes the call's region at the end of its text. A stream cut for length,
    /// aborted, or ended with any finish but `Stop` is still cut; a driver hands a stop sequence of
    /// the request's over as such a finish (see the stop row in [`crate::format`]). The state must
    /// already be in the table.
    #[must_use]
    pub fn closes_at_stop(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        self.stop_closes.push(state);
        self
    }

    /// The template discards the text in `state`, a content or wrapper state: every byte of it is
    /// `Dropped { Wrapper }`, held bytes included, at every finish (see the framing row in
    /// [`crate::format`]). The state must already be in the table.
    #[must_use]
    pub fn framing(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        if !self.framing.contains(&state) {
            self.framing.push(state);
        }
        self
    }

    /// The text of `state`, a content or reasoning state, is trimmed stay by stay: whitespace at
    /// a stay's edges is `Dropped { Whitespace }` (see the trim row in [`crate::format`]). The
    /// state must already be in the table.
    #[must_use]
    pub fn trims(mut self, state: &str) -> Self {
        let state = self.state_named(state);
        if !self.trimmed.contains(&state) {
            self.trimmed.push(state);
        }
        self
    }

    /// A later stay of `state`, a content or reasoning state, shows `text` first: the template
    /// joins the stays with it (see the join row in [`crate::format`]). Naming the state again
    /// replaces its joiner. The state must already be in the table.
    #[must_use]
    pub fn joins(mut self, state: &str, text: &str) -> Self {
        let state = self.state_named(state);
        self.joiners.retain(|&(joined, _)| joined != state);
        self.joiners.push((state, text.to_string()));
        self
    }

    /// The texts among `stops` (the model's end-of-sequence and stop tokens) that are terminals of
    /// the table, each kept once in the order given; other texts are not kept. The terminals must
    /// already be in the table. Parsing does not change, since the stop decoder hides these
    /// tokens; [`Self::is_stop_token`] tells a constraint derived from the table which closing
    /// terminals the model's stop writes.
    #[must_use]
    pub fn stop_tokens<'t>(mut self, stops: impl IntoIterator<Item = &'t str>) -> Self {
        for stop in stops {
            let terminal = self.terminals.iter().any(|terminal| terminal.text == stop);
            if terminal && !self.stop_tokens.iter().any(|known| known == stop) {
                self.stop_tokens.push(stop.to_string());
            }
        }
        self
    }

    /// Whether `text` is a terminal of the table that is a token the engine stops on
    /// ([`Self::stop_tokens`]): a region it closes ends at the model's stop, so a constraint can
    /// end that region with no text of its own.
    pub fn is_stop_token(&self, text: &str) -> bool {
        self.stop_tokens.iter().any(|stop| stop == text)
    }

    /// The table's shaped calls read their values with `reader`: a tagged value's text by the
    /// shape's parser, typed by the parameter's schema; a member of JSON arguments held and typed
    /// where the reader says; a key written twice as it says. Without the row they read as the
    /// tagged tables read theirs ([`Inferred`](crate::shape::Inferred)). No other call syntax
    /// reads it.
    #[must_use]
    pub fn reads_values_with(mut self, reader: &'static dyn ValueRead) -> Self {
        self.values = Values(reader);
        self
    }

    /// The reader a shaped call's values read with.
    pub(crate) fn value_reader(&self) -> &'static dyn ValueRead {
        self.values.0
    }

    /// The text that opens the turn the model writes, `<|im_start|>assistant` for ChatML: the
    /// prompt's replay starts at its last occurrence.
    #[must_use]
    pub fn opens_turn(mut self, text: &str) -> Self {
        self.turn_opener = Some(text.to_string());
        self
    }

    /// A literal that starts the model's turn in the prompt, as a response template's
    /// `start_anchor`: the prompt's replay starts after the end of the last anchor found (see the
    /// turn anchors in [`crate::format`]), in the first state. Anchors added earlier are preferred
    /// where several match at one place.
    #[must_use]
    pub fn turn_anchor(mut self, text: &str) -> Self {
        self.anchors.push(Anchor {
            text: text.to_string(),
            state: 0,
        });
        self
    }

    /// A turn anchor after which the output starts in `state`: an anchor that already opened the
    /// thought, or closed it. The state must already be in the table.
    #[must_use]
    pub fn turn_anchor_starting_in(mut self, text: &str, state: &str) -> Self {
        let state = self.state_named(state);
        self.anchors.push(Anchor {
            text: text.to_string(),
            state,
        });
        self
    }

    /// The state a turn starts in when its first piece is not a terminal whose row leads into
    /// another state, as a response template's `^` alternative of the state's open (see the start
    /// row in [`crate::format`]). The state must already be in the table.
    #[must_use]
    pub fn starts_in(mut self, state: &str) -> Self {
        self.start = Some(self.state_named(state));
        self
    }

    /// The state the start row names, if the table has one.
    pub(crate) fn start(&self) -> Option<usize> {
        self.start
    }

    /// The turn opener, if the table names one.
    pub(crate) fn turn_opener(&self) -> Option<&str> {
        self.turn_opener.as_deref()
    }

    /// The part of `prompt` the engine replays to find the state the generation starts in: from
    /// the last occurrence of the turn opener, or the whole prompt when the table names none or
    /// the prompt does not hold it; for a table with turn anchors, from the start of the last
    /// anchor found, or nothing when the prompt holds none. A caller that keeps the prompt for a
    /// later `Input::Prompt` need keep only this: replayed again, it gives itself.
    pub fn replay<'p>(&self, prompt: &'p str) -> &'p str {
        if !self.anchors.is_empty() {
            let from = self.last_anchor(prompt).map_or(prompt.len(), |(at, _)| at);
            return &prompt[from..];
        }
        match self.turn_opener().and_then(|opener| prompt.rfind(opener)) {
            Some(at) => &prompt[at..],
            None => prompt,
        }
    }

    /// The text of `prompt` the engine replays, and the state the replay starts in: after the
    /// last turn anchor, in the state it names; otherwise [`Self::replay`]'s part, from the first
    /// state.
    pub(crate) fn turn<'p>(&self, prompt: &'p str) -> (&'p str, usize) {
        if self.anchors.is_empty() {
            return (self.replay(prompt), 0);
        }
        match self.last_anchor(prompt) {
            Some((at, anchor)) => (&prompt[at + anchor.text.len()..], anchor.state),
            None => ("", 0),
        }
    }

    /// The last turn anchor in `prompt` and where it starts, found as `finditer` finds the
    /// alternation of the anchors: the earliest match, the first anchor in the table's order
    /// where several match there, then on from that match's end. Each anchor's next occurrence is
    /// searched for again only once the matches have passed it, so a prompt costs one pass per
    /// anchor. An anchor with no text never matches.
    fn last_anchor(&self, prompt: &str) -> Option<(usize, &Anchor)> {
        let find = |anchor: &Anchor, from: usize| {
            if anchor.text.is_empty() {
                return None;
            }
            prompt[from..]
                .find(anchor.text.as_str())
                .map(|at| from + at)
        };
        let mut next: Vec<Option<usize>> =
            self.anchors.iter().map(|anchor| find(anchor, 0)).collect();
        let mut from = 0;
        let mut last = None;
        loop {
            for (anchor, at) in self.anchors.iter().zip(next.iter_mut()) {
                if at.is_some_and(|at| at < from) {
                    *at = find(anchor, from);
                }
            }
            let Some((at, index)) = next
                .iter()
                .enumerate()
                .filter_map(|(index, at)| at.map(|at| (at, index)))
                .min()
            else {
                return last;
            };
            let anchor = &self.anchors[index];
            last = Some((at, anchor));
            from = at + anchor.text.len();
        }
    }

    /// The format's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn state_named(&self, name: &str) -> usize {
        self.states
            .iter()
            .position(|state| state.name == name)
            .unwrap_or_else(|| unknown("state", name, &self.name))
    }

    fn terminal_named(&self, name: &str) -> usize {
        self.terminals
            .iter()
            .position(|terminal| terminal.name == name && !terminal.ignored)
            .unwrap_or_else(|| unknown("terminal", name, &self.name))
    }

    /// The terminals' texts, in the order the engine's scanner numbers them.
    pub(crate) fn terminal_texts(&self) -> impl Iterator<Item = &str> {
        self.terminals.iter().map(|terminal| terminal.text.as_str())
    }

    /// Whether `tail`, at the end of the output in `state`, is a tag the end cut short: it begins
    /// with a whole structural token and is the beginning of a terminal that, whole, would move
    /// the engine out of `state` or be dropped where it stands. The beginning of a terminal the
    /// state has no row for is text, as the whole terminal would be.
    pub(crate) fn is_cut_tag(&self, state: usize, tail: &str) -> bool {
        self.structural_tokens
            .iter()
            .any(|token| tail.starts_with(token.as_str()))
            && self.terminals.iter().enumerate().any(|(index, terminal)| {
                terminal.text.len() > tail.len()
                    && terminal.text.starts_with(tail)
                    && (terminal.ignored || self.next(state, index).is_some())
            })
    }

    /// The text of terminal `index`.
    pub(crate) fn terminal_text(&self, index: usize) -> &str {
        &self.terminals[index].text
    }

    /// Whether terminal `index` is one the engine drops wherever it stands.
    pub(crate) fn ignored(&self, index: usize) -> bool {
        self.terminals[index].ignored
    }

    /// What the text in state `index` is.
    pub(crate) fn emits(&self, index: usize) -> Emits {
        self.states[index].emits
    }

    /// How many states the table has; the first is the one the turn opener leaves the engine in.
    pub(crate) fn states(&self) -> usize {
        self.states.len()
    }

    /// Where terminal `on` takes the engine from state `from`, if the table says.
    pub(crate) fn next(&self, from: usize, on: usize) -> Option<usize> {
        self.transitions
            .iter()
            .find(|transition| transition.from == from && transition.on == on)
            .map(|transition| transition.to)
    }

    /// Every row of the table as `(from, on, to)`, in the order the rows were added.
    pub(crate) fn transitions(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.transitions
            .iter()
            .map(|transition| (transition.from, transition.on, transition.to))
    }

    /// The call syntax, for a format with an arguments state.
    pub(crate) fn call_syntax(&self) -> Option<&CallSyntax> {
        self.calls.as_ref()
    }

    /// Whether the stream's stop closes the call in state `index`.
    pub(crate) fn stop_closes(&self, index: usize) -> bool {
        self.stop_closes.contains(&index)
    }

    /// Whether the template discards the text in state `index`.
    pub(crate) fn is_framing(&self, index: usize) -> bool {
        self.framing.contains(&index)
    }

    /// Whether the text of state `index` is trimmed stay by stay.
    pub(crate) fn is_trimmed(&self, index: usize) -> bool {
        self.trimmed.contains(&index)
    }

    /// The text that joins the stays of state `index`, if the table names one.
    pub(crate) fn joiner(&self, index: usize) -> Option<&str> {
        self.joiners
            .iter()
            .find(|&&(state, _)| state == index)
            .map(|(_, text)| text.as_str())
    }

    /// What every table must hold before an engine runs it: a state to start in, which does not
    /// emit reasoning (a thought the output starts inside is entered by the prompt's replay or by
    /// the start row, either of which pushes `ReasoningStart`; an arguments state at the start has
    /// its call opened by the engine), a call syntax when a state emits arguments, a text for
    /// every terminal, a turn opener and turn anchors that are not empty and not both named, a
    /// start row only where a turn's first byte is read from the first state and no opener replays
    /// it, a stop row only for an arguments state, a framing row only for a state of content or
    /// wrapping, a trim or join row only for a state of content or reasoning that is not framing,
    /// and, for a shaped call, a shape a stream can run, opened only through its terminal (the
    /// turn does not start in its state), no terminal the table ignores, and, when its header is
    /// not empty, no row into its state from a state that holds a call. The builder keeps a row
    /// from naming what the table lacks; this checks the rest.
    pub fn validate(&self) -> Result<(), String> {
        let Some(first) = self.states.first() else {
            return Err(format!("format {}: a table with no state", self.name));
        };
        if first.emits == Emits::Reasoning {
            return Err(format!(
                "format {}: the first state {:?} emits reasoning; a thought the output starts \
                 inside is entered by the prompt",
                self.name, first.name
            ));
        }
        if self
            .states
            .iter()
            .any(|state| state.emits == Emits::Arguments)
            && self.calls.is_none()
        {
            return Err(format!(
                "format {}: an arguments state with no call syntax",
                self.name
            ));
        }
        if let Some(terminal) = self
            .terminals
            .iter()
            .find(|terminal| terminal.text.is_empty())
        {
            return Err(format!(
                "format {}: terminal {:?} has no text",
                self.name, terminal.name
            ));
        }
        if self.turn_opener.as_deref() == Some("") {
            return Err(format!("format {}: an empty turn opener", self.name));
        }
        if self.anchors.iter().any(|anchor| anchor.text.is_empty()) {
            return Err(format!("format {}: an empty turn anchor", self.name));
        }
        if self.turn_opener.is_some() && !self.anchors.is_empty() {
            return Err(format!(
                "format {}: a turn opener and turn anchors; a table names one",
                self.name
            ));
        }
        if self.start.is_some() {
            if first.emits == Emits::Arguments {
                return Err(format!(
                    "format {}: a start row, but the first state {:?} holds a call",
                    self.name, first.name
                ));
            }
            if self.turn_opener.is_some() {
                return Err(format!(
                    "format {}: a start row needs the turn's start, which a turn opener replays",
                    self.name
                ));
            }
        }
        if let Some(state) = self
            .stop_closes
            .iter()
            .map(|&index| &self.states[index])
            .find(|state| state.emits != Emits::Arguments)
        {
            return Err(format!(
                "format {}: a stop closes state {:?}, which holds no call",
                self.name, state.name
            ));
        }
        if let Some(state) = self
            .framing
            .iter()
            .map(|&index| &self.states[index])
            .find(|state| matches!(state.emits, Emits::Reasoning | Emits::Arguments))
        {
            return Err(format!(
                "format {}: state {:?} is framing, but it emits {:?}",
                self.name, state.name, state.emits
            ));
        }
        let trimmed = self.trimmed.iter().map(|&index| (index, "trimmed"));
        let joined = self.joiners.iter().map(|&(index, _)| (index, "joined"));
        for (index, row) in trimmed.chain(joined) {
            let state = &self.states[index];
            if matches!(state.emits, Emits::Arguments | Emits::Wrapper) {
                return Err(format!(
                    "format {}: state {:?} is {row}, but it emits {:?}",
                    self.name, state.name, state.emits
                ));
            }
            if self.is_framing(index) {
                return Err(format!(
                    "format {}: state {:?} is {row}, but its text is framing",
                    self.name, state.name
                ));
            }
        }
        if let Some(CallSyntax::Shaped(shape)) = self.calls {
            self.validate_shaped(shape)?;
        }
        Ok(())
    }

    /// What a table with a shaped call must hold besides: a shape a stream can run, every call
    /// opened by a terminal (a shaped call opens through its terminal and its header, so the
    /// output does not start in one, nor does a start row lead into one), no terminal the table
    /// ignores (a shaped call has no place to keep one among the bytes it holds), and, when the
    /// header is not empty, no row from a state that holds a call into one: a terminal whose
    /// header does not match is the text of the state it came in, which a closed call cannot take.
    fn validate_shaped(&self, shape: &CallShape) -> Result<(), String> {
        if let Err(why) = shape.check() {
            return Err(format!("format {}: the call's shape: {why}", self.name));
        }
        let calls = |index: usize| self.states[index].emits == Emits::Arguments;
        let anchored = self.anchors.iter().any(|anchor| calls(anchor.state));
        if calls(0) || self.start.is_some_and(calls) || anchored {
            return Err(format!(
                "format {}: a shaped call opens through its terminal, but the turn may start in \
                 one",
                self.name
            ));
        }
        if let Some(terminal) = self.terminals.iter().find(|terminal| terminal.ignored) {
            return Err(format!(
                "format {}: terminal {:?} is ignored, which a shaped call cannot keep in place",
                self.name, terminal.name
            ));
        }
        if shape.header.0.is_empty() {
            return Ok(());
        }
        match self
            .transitions
            .iter()
            .find(|row| calls(row.from) && calls(row.to))
        {
            Some(row) => Err(format!(
                "format {}: a row from {:?} into {:?}, both holding calls, but the call's header \
                 may not match, and then its terminal is the text of a state that holds none",
                self.name, self.states[row.from].name, self.states[row.to].name
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_states() -> Format {
        Format::new("t")
            .terminal("open", "<a>")
            .terminal("close", "</a>")
            .state("content", Emits::Content)
            .state("inside", Emits::Reasoning)
            .transition("content", "open", "inside")
            .transition("inside", "close", "content")
    }

    #[test]
    fn rows_are_looked_up_by_the_names_they_were_added_under() {
        let format = two_states();
        assert_eq!(format.name(), "t");
        assert_eq!(format.terminal_texts().collect::<Vec<_>>(), ["<a>", "</a>"]);
        assert_eq!(format.terminal_text(1), "</a>");
        assert_eq!(format.emits(0), Emits::Content);
        assert_eq!(format.emits(1), Emits::Reasoning);
        assert_eq!(format.next(0, 0), Some(1));
        assert_eq!(format.next(1, 1), Some(0));
        assert_eq!(format.next(0, 1), None, "a closer in content has no row");
        assert_eq!(format.next(1, 0), None, "an opener inside has no row");
        assert!(format.call_syntax().is_none());
        assert!(format.validate().is_ok());
        assert_eq!(
            Format::new("empty").validate(),
            Err("format empty: a table with no state".to_string())
        );
        assert_eq!(
            Format::new("calls")
                .state("calls", Emits::Arguments)
                .validate(),
            Err("format calls: an arguments state with no call syntax".to_string())
        );
        assert_eq!(
            Format::new("blank")
                .terminal("t", "")
                .state("c", Emits::Content)
                .validate(),
            Err("format blank: terminal \"t\" has no text".to_string())
        );
        assert_eq!(
            Format::new("opener")
                .state("c", Emits::Content)
                .opens_turn("")
                .validate(),
            Err("format opener: an empty turn opener".to_string())
        );
        assert_eq!(
            Format::new("thought")
                .state("r", Emits::Reasoning)
                .validate(),
            Err(
                "format thought: the first state \"r\" emits reasoning; a thought the output \
                 starts inside is entered by the prompt"
                    .to_string()
            )
        );
    }

    #[test]
    fn a_stop_row_names_an_arguments_state() {
        let format = Format::new("stop")
            .terminal("open", "<call>")
            .state("content", Emits::Content)
            .state("call", Emits::Arguments)
            .transition("content", "open", "call")
            .calls(CallSyntax::Json)
            .closes_at_stop("call");
        assert!(format.stop_closes(1));
        assert!(!format.stop_closes(0));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.closes_at_stop("content").validate(),
            Err("format stop: a stop closes state \"content\", which holds no call".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"call\"")]
    fn a_stop_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().closes_at_stop("call");
    }

    /// Gemma 4's two anchors, spelled from the response template of `google/gemma-4-31B-it` (its
    /// `tokenizer_config.json`), around a thought and the content.
    fn gemma_anchors() -> Format {
        Format::new("anchors")
            .terminal("think_open", "<|channel>thought\n")
            .terminal("think_close", "<channel|>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .turn_anchor("<|turn>model\n")
            .turn_anchor("<tool_response|>")
    }

    #[test]
    fn the_last_anchor_of_either_kind_starts_the_turn_after_its_end() {
        let format = gemma_anchors();
        let turn = "<|turn>user\nWeather?<turn|>\n<|turn>model\n";
        assert_eq!(format.replay(turn), "<|turn>model\n");
        assert_eq!(format.turn(turn), ("", 0));
        let after_a_tool = "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                            response:f{}<tool_response|><|channel>thought\n";
        assert_eq!(
            format.replay(after_a_tool),
            "<tool_response|><|channel>thought\n"
        );
        assert_eq!(format.turn(after_a_tool), ("<|channel>thought\n", 0));
        let later_turn = "<|turn>model\n<tool_response|>Done.<turn|>\n<|turn>user\nThanks<turn|>\n\
                          <|turn>model\n";
        assert_eq!(format.turn(later_turn), ("", 0));
    }

    #[test]
    fn anchors_are_found_as_finditer_finds_them() {
        let doubled = Format::new("doubled")
            .state("content", Emits::Content)
            .turn_anchor("aa");
        // `finditer` takes `aa` at 0 and goes on from 2: the turn starts after the first two.
        assert_eq!(doubled.turn("aaa"), ("a", 0));
        assert_eq!(doubled.turn("aaaa"), ("", 0));
        let longest_first = Format::new("longest first")
            .state("content", Emits::Content)
            .turn_anchor("ab")
            .turn_anchor("a");
        assert_eq!(longest_first.turn("xab"), ("", 0));
        let shortest_first = Format::new("shortest first")
            .state("content", Emits::Content)
            .turn_anchor("a")
            .turn_anchor("ab");
        assert_eq!(shortest_first.turn("xab"), ("b", 0));
        // A later anchor that starts inside the last match is not one.
        let overlapping = Format::new("overlapping")
            .state("content", Emits::Content)
            .turn_anchor("<a><b>")
            .turn_anchor("<b>c");
        assert_eq!(overlapping.turn("<a><b>c"), ("c", 0));
        assert_eq!(overlapping.turn("<b>c<a><b>c"), ("c", 0));
    }

    #[test]
    fn a_prompt_without_an_anchor_replays_nothing_and_a_replay_gives_itself() {
        let format = gemma_anchors();
        assert_eq!(format.replay("hello "), "");
        assert_eq!(format.turn("hello "), ("", 0));
        assert_eq!(format.turn(""), ("", 0));
        for prompt in [
            "<|turn>model\n",
            "<|turn>model\n<|channel>thought\nPlan",
            "a<tool_response|>b<|turn>model\nc<tool_response|>d",
            "no anchor at all",
        ] {
            let kept = format.replay(prompt);
            assert_eq!(format.replay(kept), kept, "{prompt:?}");
            assert_eq!(format.turn(kept), format.turn(prompt), "{prompt:?}");
        }
    }

    #[test]
    fn an_anchor_names_the_state_the_output_starts_in() {
        // TRL's Nemotron 3 template: the anchor pattern `<\|im_start\|>assistant\n<think>(?:\n|
        // </think>)?` is three literals, the longer ones first as the greedy `?` tries them.
        let format = Format::new("nemotron 3")
            .terminal("think_close", "</think>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .transition("reasoning", "think_close", "content")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think>\n", "reasoning")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think></think>", "content")
            .turn_anchor_starting_in("<|im_start|>assistant\n<think>", "reasoning");
        assert!(format.validate().is_ok());
        assert_eq!(
            format.turn("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"),
            ("", 1)
        );
        assert_eq!(
            format.turn("<|im_start|>assistant\n<think></think>"),
            ("", 0)
        );
        assert_eq!(format.turn("<|im_start|>assistant\n<think>"), ("", 1));
        assert_eq!(format.turn("<|im_start|>assistant\n<think>x"), ("x", 1));
    }

    #[test]
    fn an_anchor_is_not_empty_and_a_table_names_an_opener_or_anchors() {
        assert_eq!(
            Format::new("empty")
                .state("c", Emits::Content)
                .turn_anchor("")
                .validate(),
            Err("format empty: an empty turn anchor".to_string())
        );
        assert_eq!(
            Format::new("both")
                .state("c", Emits::Content)
                .opens_turn("<|assistant|>")
                .turn_anchor("<|assistant|>")
                .validate(),
            Err("format both: a turn opener and turn anchors; a table names one".to_string())
        );
        // An empty anchor never matches, even in a table nobody validated.
        let unchecked = Format::new("unchecked")
            .state("c", Emits::Content)
            .turn_anchor("")
            .turn_anchor("x");
        assert_eq!(unchecked.turn("axb"), ("b", 0));
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"thought\"")]
    fn an_anchor_starting_in_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().turn_anchor_starting_in("<turn>", "thought");
    }

    #[test]
    fn a_start_row_needs_a_first_state_without_a_call_and_no_opener() {
        let format = two_states().starts_in("inside");
        assert_eq!(format.start(), Some(1));
        assert!(format.validate().is_ok());
        assert_eq!(two_states().start(), None);
        assert_eq!(
            Format::new("list")
                .state("calls", Emits::Arguments)
                .state("content", Emits::Content)
                .calls(CallSyntax::JsonList)
                .starts_in("content")
                .validate(),
            Err("format list: a start row, but the first state \"calls\" holds a call".to_string())
        );
        assert_eq!(
            format.opens_turn("<turn>").validate(),
            Err(
                "format t: a start row needs the turn's start, which a turn opener replays"
                    .to_string()
            )
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"thought\"")]
    fn a_start_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().starts_in("thought");
    }

    #[test]
    fn a_framing_row_names_a_state_of_content_or_wrapping() {
        let format = Format::new("framing")
            .terminal("open", "<call>")
            .state("outside", Emits::Content)
            .state("between", Emits::Wrapper)
            .state("thought", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("outside", "open", "call")
            .calls(CallSyntax::Json)
            .framing("outside")
            .framing("between")
            .framing("outside");
        assert_eq!(format.framing, [0, 1], "a state named twice is kept once");
        assert!(format.is_framing(0));
        assert!(format.is_framing(1));
        assert!(!format.is_framing(2));
        assert!(!format.is_framing(3));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.clone().framing("thought").validate(),
            Err("format framing: state \"thought\" is framing, but it emits Reasoning".to_string())
        );
        assert_eq!(
            format.framing("call").validate(),
            Err("format framing: state \"call\" is framing, but it emits Arguments".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"outside\"")]
    fn a_framing_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().framing("outside");
    }

    #[test]
    fn a_trim_row_names_a_state_of_content_or_reasoning_that_is_not_framing() {
        let format = Format::new("trimmed")
            .terminal("open", "<call>")
            .state("content", Emits::Content)
            .state("thought", Emits::Reasoning)
            .state("between", Emits::Wrapper)
            .state("call", Emits::Arguments)
            .state("outside", Emits::Content)
            .transition("content", "open", "call")
            .calls(CallSyntax::Json)
            .framing("outside")
            .trims("content")
            .trims("thought")
            .trims("content");
        assert_eq!(format.trimmed, [0, 1], "a state named twice is kept once");
        assert!(format.is_trimmed(0) && format.is_trimmed(1) && !format.is_trimmed(2));
        assert!(format.validate().is_ok());
        assert_eq!(
            format.clone().trims("between").validate(),
            Err("format trimmed: state \"between\" is trimmed, but it emits Wrapper".to_string())
        );
        assert_eq!(
            format.clone().trims("call").validate(),
            Err("format trimmed: state \"call\" is trimmed, but it emits Arguments".to_string())
        );
        assert_eq!(
            format.trims("outside").validate(),
            Err(
                "format trimmed: state \"outside\" is trimmed, but its text is framing".to_string()
            )
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"answer\"")]
    fn a_trim_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().trims("answer");
    }

    #[test]
    fn a_join_row_names_a_state_of_content_or_reasoning_that_is_not_framing() {
        let format = Format::new("joined")
            .state("content", Emits::Content)
            .state("thought", Emits::Reasoning)
            .state("between", Emits::Wrapper)
            .state("outside", Emits::Content)
            .framing("outside")
            .joins("content", "|")
            .joins("thought", "\n")
            .joins("content", "");
        assert_eq!(
            format.joiner(0),
            Some(""),
            "a state named again takes the later joiner"
        );
        assert_eq!(format.joiner(1), Some("\n"));
        assert_eq!(format.joiner(2), None);
        assert!(format.validate().is_ok());
        assert_eq!(
            format.clone().joins("between", "|").validate(),
            Err("format joined: state \"between\" is joined, but it emits Wrapper".to_string())
        );
        assert_eq!(
            format.joins("outside", "|").validate(),
            Err("format joined: state \"outside\" is joined, but its text is framing".to_string())
        );
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"answer\"")]
    fn a_join_row_for_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().joins("answer", "|");
    }

    #[test]
    fn the_stop_tokens_kept_are_the_terminals_among_the_stops_each_once() {
        let format = two_states().terminal("call", "<|call|>").stop_tokens([
            "<|return|>",
            "<|call|>",
            "</a>",
            "<|call|>",
        ]);
        assert!(format.is_stop_token("<|call|>"));
        assert!(format.is_stop_token("</a>"));
        assert!(
            !format.is_stop_token("<|return|>"),
            "a stop the table has no terminal for is not kept"
        );
        assert!(!format.is_stop_token("<a>"));
        assert_eq!(format.stop_tokens, ["<|call|>", "</a>"]);
        assert!(
            !two_states()
                .stop_tokens(["<|call|>"])
                .is_stop_token("<|call|>"),
            "a stop named before its terminal is not one yet"
        );
        assert!(format.validate().is_ok());
    }

    /// A shape on abstract tags, its header `<n=NAME>` when `header`, empty otherwise.
    fn shape(header: bool) -> CallShape {
        use crate::shape::{Arguments, Atom, Class, Dialect, NameSource, Pattern, Slot};
        let word = Class::new([('a', 'z')], false);
        let (header, name) = if header {
            let atoms = [
                Atom::Lit("<n=".into()),
                Atom::Run {
                    class: word.clone(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Lit(">".into()),
            ];
            (Pattern(atoms.into()), NameSource::Header)
        } else {
            (Pattern::default(), NameSource::Literal("f".into()))
        };
        CallShape {
            header,
            name,
            arguments: Arguments::Json(Dialect::default()),
            trailer: Pattern::default(),
            word,
            hold_cap: 64,
        }
    }

    fn shaped(header: bool) -> CallSyntax {
        CallSyntax::Shaped(crate::shape::intern(shape(header)))
    }

    #[test]
    fn a_shaped_call_opens_from_a_state_that_holds_none() {
        let table = |syntax: CallSyntax| {
            Format::new("shaped")
                .terminal("open", "<c>")
                .terminal("close", "</c>")
                .state("content", Emits::Content)
                .state("call", Emits::Arguments)
                .transition("content", "open", "call")
                .transition("call", "close", "content")
                .calls(syntax)
        };
        assert!(table(shaped(true)).validate().is_ok());
        assert_eq!(
            table(shaped(true))
                .transition("call", "open", "call")
                .validate(),
            Err(
                "format shaped: a row from \"call\" into \"call\", both holding calls, but the \
                 call's header may not match, and then its terminal is the text of a state that \
                 holds none"
                    .to_string()
            )
        );
        assert!(
            table(shaped(false))
                .transition("call", "open", "call")
                .validate()
                .is_ok(),
            "a call with no header always opens at its terminal"
        );
        assert_eq!(
            table(shaped(true)).ignores("sep", "|").validate(),
            Err(
                "format shaped: terminal \"sep\" is ignored, which a shaped call cannot keep in \
                 place"
                    .to_string()
            )
        );
        for starts_in_a_call in [
            table(shaped(true)).starts_in("call"),
            table(shaped(true)).turn_anchor_starting_in("<turn>", "call"),
        ] {
            assert_eq!(
                starts_in_a_call.validate(),
                Err(
                    "format shaped: a shaped call opens through its terminal, but the turn may \
                     start in one"
                        .to_string()
                )
            );
        }
        let unnamed = CallSyntax::Shaped(crate::shape::intern(CallShape {
            name: crate::shape::NameSource::Body,
            ..shape(true)
        }));
        assert_eq!(
            table(unnamed).validate(),
            Err(
                "format shaped: the call's shape: the name comes from the arguments, which are \
                 no envelope"
                    .to_string()
            )
        );
    }

    #[test]
    fn an_ignored_terminal_is_scanned_for_and_cannot_be_named_in_a_row() {
        let format = two_states().ignores("sep", "|");
        assert_eq!(
            format.terminal_texts().collect::<Vec<_>>(),
            ["<a>", "</a>", "|"]
        );
        assert!(format.ignored(2));
        assert!(!format.ignored(0));
    }

    #[test]
    #[should_panic(expected = "format t: no terminal named \"sep\"")]
    fn a_transition_on_an_ignored_terminal_is_a_programming_error() {
        let _ = two_states()
            .ignores("sep", "|")
            .transition("content", "sep", "inside");
    }

    #[test]
    #[should_panic(expected = "format t: no terminal named \"missing\"")]
    fn a_transition_on_a_terminal_the_table_lacks_is_a_programming_error() {
        let _ = two_states().transition("content", "missing", "inside");
    }

    #[test]
    #[should_panic(expected = "format t: no state named \"nowhere\"")]
    fn a_transition_to_a_state_the_table_lacks_is_a_programming_error() {
        let _ = two_states().transition("content", "open", "nowhere");
    }
}
