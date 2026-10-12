//! Events from one call given as data, a [`CallShape`], as its bytes arrive.
//!
//! A table whose call syntax is [`CallSyntax::Shaped`] hands this assembler what follows the
//! region's terminal. A Hugging Face response template writes a call's region as patterns: the
//! open (`\s*<tool_call>\s*<function=(?P<name>[^>\n]+)>` in the `transformers serve` template
//! for Qwen 3.5), the arguments, and the close (`</function>\s*</tool_call>`). The table's
//! terminals are the open's first literal and the close's last one; the shape holds the rest: the
//! header after the open's terminal, which may capture the tool's name, the way the arguments are
//! written, and the trailer before the close's terminal.
//!
//! - **The header** is read first, with a [`Matcher`], before the call opens: the engine holds the
//!   terminal and everything after it, and moves nothing, until the header is decided. When it
//!   matches ([`Header::Matched`]), [`Assembler::start`] starts the call (named by the header, by
//!   the template, or later by the arguments' own object) and gives back the bytes that followed
//!   the header, for the engine to read again in the call. When it cannot match
//!   ([`Header::Failed`]), the terminal opened no call: the engine takes back what the header
//!   read ([`Assembler::into_header_bytes`]) and reads the terminal and those bytes again from the
//!   terminal's second character, as a regular expression's search goes on from the place after
//!   one where its pattern failed. A header holds at most the shape's cap of bytes undecided; past
//!   it, it is no match.
//! - **The arguments** follow: one JSON value ([`json::Arguments`], the model's bytes streamed as
//!   fragments) or one object that names the call ([`json::Assembler`]). What comes after them is
//!   the trailer, or text that belongs to no part of the call.
//! - **The trailer** is checked when a closing terminal comes ([`Assembler::closes`]): the region
//!   closes there only when the trailer ends right before the terminal, starting no earlier than
//!   the end of what the arguments took, and the earliest such start is the trailer's
//!   ([`Suffix`]). A terminal the trailer does not end at closes nothing: the engine hands it to
//!   the call as its text, as the template's closing pattern reads past it. An empty trailer ends
//!   wherever the terminal comes. Until the region closes, the bytes after the arguments are
//!   held, since they may turn out to be the trailer; with an empty trailer they are classified as
//!   they arrive.
//!
//! Every byte lands in one event. The header's bytes are the start's source (dropped as wrapping
//! when the arguments' object names the call, since its start carries that object's head); the
//! arguments' bytes are their fragments' sources; the trailer and the closing terminal are the
//! end's source. Whitespace after the arguments that is not the trailer is dropped as wrapping, and
//! anything else there is `Malformed`, run by run, so that where the chunks were cut changes
//! nothing. A region the stream cut is ended with what arrived: the call is not closed for the
//! client, and what was held comes back as `Malformed` with `UnterminatedRegion`, its whitespace
//! dropped.
//!
//! [`CallSyntax::Shaped`]: crate::CallSyntax::Shaped

use crate::{
    engine::TEXT_AFTER_THE_OBJECT,
    event::{DropReason, Event, Events, MalformedReason, Text},
    json,
    shape::{Arguments, CallShape, Match, Matcher, NameSource, Slot, Suffix, Verdict},
};

const TEXT_AFTER_THE_ARGUMENTS: &str = "text after a call's arguments";

/// What a call's header says, once the bytes fed after the terminal decide it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// More bytes can still change the answer.
    Pending,
    /// The header matched: [`Assembler::start`] starts the call.
    Matched,
    /// The header cannot match: the terminal opened no call, and
    /// [`Assembler::into_header_bytes`] gives back what the header read.
    Failed,
}

/// The events of one shaped call, from the bytes after its region's terminal.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    shape: &'static CallShape,
    /// The terminal's last character, which a `\b` at the header's start reads.
    before: Option<char>,
    stage: Stage,
}

#[derive(Clone, Debug)]
enum Stage {
    /// The header is being read; the matcher holds every byte after the terminal.
    Header(Box<Matcher<'static>>),
    /// The header matched: what it read, where it ended, and the name it gave; the call starts
    /// when the engine has moved into it.
    Matched { held: String, found: Match },
    /// The call has started: its arguments, and what follows them.
    Body(Box<Body>),
    /// The call has ended.
    Done,
}

/// A started call.
#[derive(Clone, Debug)]
struct Body {
    reader: Reader,
    /// The last character the call took, which a `\b` at the start of the trailer reads.
    before: Option<char>,
    /// Bytes after the arguments, held until the region closes, since they may be the trailer.
    after: String,
    /// The trailer's search over `after`, from the arguments' end on.
    tail: Option<Suffix<'static>>,
    /// Where the trailer starts in `after`, once a closing terminal was found to end it.
    trailer_at: Option<usize>,
}

/// How the arguments are read.
#[derive(Clone, Debug)]
enum Reader {
    /// One JSON value, the call already started.
    Json(json::Arguments),
    /// One object that names the call and holds its arguments.
    Envelope(json::Assembler),
}

impl Reader {
    /// Whether the arguments are over: the value or the object is whole.
    fn over(&self) -> bool {
        match self {
            Self::Json(arguments) => arguments.whole(),
            Self::Envelope(assembler) => assembler.done(),
        }
    }

    /// Why text after the arguments that is not whitespace is malformed: the reason the JSON
    /// syntax gives for text after a call's object, for an object that names the call.
    fn surplus(&self) -> MalformedReason {
        MalformedReason::Other(
            match self {
                Self::Json(_) => TEXT_AFTER_THE_ARGUMENTS,
                Self::Envelope(_) => TEXT_AFTER_THE_OBJECT,
            }
            .to_string(),
        )
    }
}

impl Assembler {
    /// An assembler for the call at `index`, with the id the format minted for it, of `shape`,
    /// right after its region's terminal, whose last character is `before`.
    pub fn new(
        index: u32,
        id: impl Into<String>,
        shape: &'static CallShape,
        before: Option<char>,
    ) -> Self {
        let matcher =
            Matcher::new(&shape.header, &[], &shape.word, before).hold_cap(Some(shape.cap()));
        Self {
            index,
            id: id.into(),
            shape,
            before,
            stage: Stage::Header(Box::new(matcher)),
        }
    }

    /// Read the next bytes after the terminal into the header, and say what they decide. Once
    /// decided, the answer stands, and more bytes are not read.
    pub fn header(&mut self, bytes: &str) -> Header {
        match &mut self.stage {
            Stage::Header(matcher) => {
                let verdict = matcher.feed(bytes);
                self.decided(verdict)
            }
            Stage::Matched { .. } | Stage::Body(_) | Stage::Done => Header::Matched,
        }
    }

    /// No more bytes come: what the bytes read decide.
    pub fn header_end(&mut self) -> Header {
        match &mut self.stage {
            Stage::Header(matcher) => {
                let verdict = matcher.finish();
                self.decided(verdict)
            }
            Stage::Matched { .. } | Stage::Body(_) | Stage::Done => Header::Matched,
        }
    }

    fn decided(&mut self, verdict: Verdict) -> Header {
        match verdict {
            Verdict::Pending => Header::Pending,
            Verdict::NoMatch => Header::Failed,
            Verdict::Matched(found) => {
                let Stage::Header(matcher) = std::mem::replace(&mut self.stage, Stage::Done) else {
                    return Header::Matched;
                };
                self.stage = Stage::Matched {
                    held: matcher.into_held(),
                    found,
                };
                Header::Matched
            }
        }
    }

    /// What the header read, given back when it did not match.
    pub fn into_header_bytes(self) -> String {
        match self.stage {
            Stage::Header(matcher) => matcher.into_held(),
            Stage::Matched { held, .. } => held,
            Stage::Body(_) | Stage::Done => String::new(),
        }
    }

    /// The header matched and the engine has moved into the call: push its start, named by the
    /// header or the template, with the header's bytes as its source (or the header's bytes as
    /// wrapping, when the arguments' own object names the call and its start comes with that
    /// object's head). Returns the bytes that followed the header, which the engine reads again,
    /// now inside the call.
    pub fn start(&mut self, out: &mut Events) -> String {
        let Stage::Matched { held, found } = std::mem::replace(&mut self.stage, Stage::Done) else {
            return String::new();
        };
        let end = found.end();
        let header = &held[..end];
        let before = header.chars().next_back().or(self.before);
        let reader = match &self.shape.arguments {
            Arguments::Envelope { .. } => {
                if !header.is_empty() {
                    out.push(Event::Dropped {
                        text: Text::uncounted(header),
                        why: DropReason::Wrapper,
                    });
                }
                Reader::Envelope(json::Assembler::new(self.index, self.id.clone()))
            }
            Arguments::Json(_) => {
                let name = match &self.shape.name {
                    NameSource::Literal(name) => name.to_string(),
                    NameSource::Header | NameSource::Body => found
                        .span(Slot::Name)
                        .map(|span| held[span].to_string())
                        .unwrap_or_default(),
                };
                out.push(Event::ToolCallStart {
                    index: self.index,
                    id: self.id.clone(),
                    name,
                    source: Text::uncounted(header),
                });
                Reader::Json(json::Arguments::new(self.index))
            }
        };
        self.stage = Stage::Body(Box::new(Body {
            reader,
            before,
            after: String::new(),
            tail: None,
            trailer_at: None,
        }));
        held[end..].to_string()
    }

    /// Whether the call has started: whether its `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        match &self.stage {
            Stage::Body(body) => match &body.reader {
                Reader::Json(_) => true,
                Reader::Envelope(assembler) => assembler.started(),
            },
            Stage::Header(_) | Stage::Matched { .. } | Stage::Done => false,
        }
    }

    /// Append the next bytes of the call, after its start, and push the events they complete.
    /// Every byte is the call's until the engine closes the region.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) {
        let shape = self.shape;
        let Stage::Body(body) = &mut self.stage else {
            return;
        };
        let mut rest = bytes;
        if !body.reader.over() {
            let taken = match &mut body.reader {
                Reader::Json(arguments) => arguments.feed(rest, out),
                Reader::Envelope(assembler) => assembler.feed(rest, out),
            };
            if let Some(last) = rest[..taken].chars().next_back() {
                body.before = Some(last);
            }
            rest = &rest[taken..];
            if !body.reader.over() {
                return;
            }
        }
        if shape.trailer.0.is_empty() {
            wrapping(rest, &body.reader.surplus(), out);
            return;
        }
        let tail = body.tail.get_or_insert_with(|| {
            Suffix::new(&shape.trailer, &shape.word, body.before).hold_cap(Some(shape.cap()))
        });
        tail.feed(rest);
        body.after.push_str(rest);
    }

    /// Whether the region closes at `terminal`, a closing terminal that just came (empty for the
    /// stream's stop): whether the trailer ends right before it, starting no earlier than where
    /// the arguments' bytes end. When it does not, the terminal closes nothing, and the engine
    /// feeds it to the call as its text.
    pub fn closes(&mut self, terminal: &str) -> bool {
        let shape = self.shape;
        let Stage::Body(body) = &mut self.stage else {
            return true;
        };
        let at = if shape.trailer.0.is_empty() {
            Some(body.after.len())
        } else {
            let next = terminal.chars().next();
            match &mut body.tail {
                Some(tail) => tail.start(next),
                // Nothing has come after the arguments: only a trailer that matches no text can
                // end here.
                None => Suffix::new(&shape.trailer, &shape.word, body.before)
                    .hold_cap(Some(shape.cap()))
                    .start(next),
            }
        };
        body.trailer_at = at;
        at.is_some()
    }

    /// The region closed at `terminal` (empty for the stream's stop), which [`closes`] said it
    /// does: the call ends. Bytes after the arguments and before the trailer are dropped where
    /// they are whitespace and malformed otherwise; the trailer and the terminal are the end's
    /// source. Returns whether the terminal was taken so; when the arguments' own object named the
    /// call, its close was the call's end, and the trailer and the terminal are the region's.
    ///
    /// [`closes`]: Self::closes
    pub fn close(mut self, terminal: &str, out: &mut Events) -> bool {
        let Stage::Body(body) = std::mem::replace(&mut self.stage, Stage::Done) else {
            return false;
        };
        let Body {
            reader,
            after,
            trailer_at,
            ..
        } = *body;
        let at = trailer_at.unwrap_or(after.len());
        let (between, trailer) = after.split_at(at);
        let surplus = reader.surplus();
        let enveloped = match reader {
            Reader::Json(mut arguments) => {
                arguments.finish(out);
                false
            }
            Reader::Envelope(assembler) => {
                assembler.finish(out);
                true
            }
        };
        wrapping(between, &surplus, out);
        if enveloped {
            if !trailer.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(trailer),
                    why: DropReason::Wrapper,
                });
            }
            return false;
        }
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(format!("{trailer}{terminal}")),
        });
        true
    }

    /// The stream was cut inside the region: the call ends with what arrived, not closed for the
    /// client, and what was held after its arguments comes back as `Malformed` with
    /// `UnterminatedRegion`, its whitespace dropped.
    pub fn finish(mut self, out: &mut Events) {
        let Stage::Body(body) = std::mem::replace(&mut self.stage, Stage::Done) else {
            return;
        };
        match body.reader {
            Reader::Json(mut arguments) => {
                arguments.finish(out);
                wrapping(&body.after, &MalformedReason::UnterminatedRegion, out);
                out.push(Event::ToolCallEnd {
                    index: self.index,
                    source: Text::default(),
                });
            }
            Reader::Envelope(assembler) => {
                assembler.finish(out);
                wrapping(&body.after, &MalformedReason::UnterminatedRegion, out);
            }
        }
    }
}

/// Bytes after a call's arguments that are no part of the call: each run of whitespace dropped as
/// wrapping, each run of anything else `Malformed` for `why`. Classifying run by run keeps the
/// result the same wherever the chunks were cut.
fn wrapping(text: &str, why: &MalformedReason, out: &mut Events) {
    let mut rest = text;
    while let Some(first) = rest.chars().next() {
        let space = first.is_whitespace();
        let length = rest
            .char_indices()
            .find(|(_, c)| c.is_whitespace() != space)
            .map_or(rest.len(), |(at, _)| at);
        let run = Text::uncounted(&rest[..length]);
        out.push(if space {
            Event::Dropped {
                text: run,
                why: DropReason::Wrapper,
            }
        } else {
            Event::Malformed {
                text: run,
                why: why.clone(),
            }
        });
        rest = &rest[length..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        format::{CallSyntax, Emits, Format},
        input::{EngineFinish, Input, TokenSpan},
        parser::Parser,
        shape::{intern, Atom, Class, Dialect, Path, Pattern},
        tagged::Declared,
    };

    fn word() -> Class {
        Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false)
    }

    fn space() -> Class {
        Class::new([('\t', '\r'), (' ', ' ')], false)
    }

    fn lit(text: &str) -> Atom {
        Atom::Lit(text.into())
    }

    /// `<n=NAME>` on abstract tags: the header of the shapes here.
    fn named_header() -> Pattern {
        Pattern(
            [
                lit("<n="),
                Atom::Run {
                    class: word(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                lit(">"),
            ]
            .into(),
        )
    }

    /// A call named by `<n=NAME>` after its terminal, its arguments one JSON value, with
    /// `trailer` before its closing terminal.
    fn named(trailer: Pattern) -> &'static CallShape {
        intern(CallShape {
            header: named_header(),
            name: NameSource::Header,
            arguments: Arguments::Json(Dialect::default()),
            trailer,
            word: word(),
            hold_cap: 64,
        })
    }

    /// `</f>` and the whitespace after it: a trailer on abstract tags.
    fn closing_f() -> Pattern {
        Pattern(
            [
                lit("</f>"),
                Atom::Run {
                    class: space(),
                    min: 0,
                    max: None,
                    greedy: true,
                    capture: None,
                },
            ]
            .into(),
        )
    }

    /// A call region between `<c>` and `</c>` beside a thought between `<t>` and `</t>`, the
    /// calls given by `shape`.
    fn table(shape: &'static CallShape) -> Format {
        Format::new("shaped")
            .terminal("think_open", "<t>")
            .terminal("think_close", "</t>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("content", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("content", "think_open", "reasoning")
            .transition("reasoning", "think_close", "content")
            .transition("content", "call_open", "call")
            .transition("reasoning", "call_open", "call")
            .transition("call", "call_close", "content")
            .calls(CallSyntax::Shaped(shape))
    }

    /// Events with adjacent text of one kind joined, so that chunkings compare by what they say.
    fn joined(events: Vec<Event>) -> Vec<Event> {
        let mut out: Vec<Event> = Vec::new();
        for event in events {
            match (out.last_mut(), event) {
                (
                    Some(Event::Dropped { text, why }),
                    Event::Dropped {
                        text: more,
                        why: again,
                    },
                ) if *why == again => text.text.push_str(&more.text),
                (
                    Some(Event::Malformed { text, why }),
                    Event::Malformed {
                        text: more,
                        why: again,
                    },
                ) if *why == again => text.text.push_str(&more.text),
                (
                    Some(Event::ToolCallArguments {
                        index,
                        json,
                        source,
                    }),
                    Event::ToolCallArguments {
                        index: other,
                        json: more,
                        source: bytes,
                    },
                ) if *index == other => {
                    json.push_str(&more);
                    source.text.push_str(&bytes.text);
                }
                (Some(Event::Content(text)), Event::Content(more))
                | (Some(Event::Reasoning(text)), Event::Reasoning(more)) => {
                    text.text.push_str(&more.text);
                }
                (_, event) => out.push(event),
            }
        }
        out
    }

    /// The events of `output` after `prompt`, the same whole, cut once at every place and a
    /// character at a time, every byte accounted for.
    fn run_after(format: &Format, prompt: &str, output: &str, finish: EngineFinish) -> Vec<Event> {
        let starts: Vec<usize> = output.char_indices().map(|(at, _)| at).skip(1).collect();
        let mut plans: Vec<Vec<usize>> = vec![vec![], starts.clone()];
        plans.extend(starts.iter().map(|&cut| vec![cut]));
        let mut runs = plans.iter().map(|cuts| {
            let mut engine = Engine::new(format.clone(), Declared::default());
            let mut out = Events::new();
            engine
                .feed(
                    Input::Prompt {
                        token_ids: &[],
                        text: prompt,
                    },
                    &mut out,
                )
                .expect("the prompt");
            let mut from = 0;
            for &cut in cuts.iter().chain([&output.len()]) {
                engine
                    .feed(
                        Input::Delta {
                            token_ids: &[],
                            text: &output[from..cut],
                            spans: &[],
                        },
                        &mut out,
                    )
                    .expect("a delta");
                from = cut;
            }
            engine
                .feed(
                    Input::End {
                        finish: finish.clone(),
                    },
                    &mut out,
                )
                .expect("the end");
            assert_eq!(out.text(), output, "every byte, cut at {cuts:?}");
            joined(out.drain())
        });
        let whole = runs.next().expect("the whole output is one plan");
        for (events, cuts) in runs.zip(plans.iter().skip(1)) {
            assert_eq!(events, whole, "cut at {cuts:?}");
        }
        whole
    }

    fn run(format: &Format, output: &str) -> Vec<Event> {
        run_after(format, "", output, EngineFinish::Stop)
    }

    fn content(text: &str) -> Event {
        Event::Content(Text::uncounted(text))
    }

    fn wrapper(text: &str) -> Event {
        Event::Dropped {
            text: Text::uncounted(text),
            why: DropReason::Wrapper,
        }
    }

    fn start(name: &str, source: &str) -> Event {
        Event::ToolCallStart {
            index: 0,
            id: "call_0".to_string(),
            name: name.to_string(),
            source: Text::uncounted(source),
        }
    }

    fn arguments(json: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(json),
        }
    }

    fn end(source: &str) -> Event {
        Event::ToolCallEnd {
            index: 0,
            source: Text::uncounted(source),
        }
    }

    fn finish(calls: u32) -> Event {
        Event::Finish {
            reason: crate::event::FinishReason::Stop,
            tool_calls: calls,
            reasoning_tokens: 0,
        }
    }

    #[test]
    fn a_header_names_the_call_and_the_closing_terminal_ends_it() {
        let events = run(
            &table(named(Pattern::default())),
            "Sure.<c><n=get_weather>{\"city\": \"Paris\"}\n</c>Done.",
        );
        assert_eq!(
            events,
            [
                content("Sure."),
                wrapper("<c>"),
                start("get_weather", "<n=get_weather>"),
                arguments("{\"city\": \"Paris\"}"),
                wrapper("\n"),
                end("</c>"),
                content("Done."),
                finish(1),
            ]
        );
    }

    #[test]
    fn a_terminal_whose_header_does_not_match_is_the_text_where_it_came() {
        let format = table(named(Pattern::default()));
        assert_eq!(
            run(&format, "Hi <c><x>bye"),
            [content("Hi <c><x>bye"), finish(0)],
            "the search goes on after the terminal's first character"
        );
        assert_eq!(
            run(&format, "<t>plan <c>oops</t>ok"),
            [
                wrapper("<t>"),
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("plan <c>oops")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                content("ok"),
                finish(0),
            ],
            "in a thought, the thought's text, and its close is still its close"
        );
        assert_eq!(
            run(&format, "<c><c><n=f>{}</c>"),
            [
                content("<c>"),
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "the next terminal after the first character opens the call"
        );
        assert_eq!(
            run(&format, "<c><n=f<c><n=g>{}</c>"),
            [
                content("<c><n=f"),
                wrapper("<c>"),
                start("g", "<n=g>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "a terminal inside a header is the header's text until the header fails"
        );
    }

    #[test]
    fn the_end_of_the_stream_decides_a_header_still_being_read() {
        let format = table(named(Pattern::default()));
        assert_eq!(run(&format, "<c><n=f"), [content("<c><n=f"), finish(0)]);
        let events = run_after(&format, "", "<c><n=f>", EngineFinish::Length);
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                end(""),
                Event::Finish {
                    reason: crate::event::FinishReason::Length,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ],
            "a header that matched at the end starts a call the end cut"
        );
    }

    #[test]
    fn a_given_back_terminal_is_read_by_the_rows_of_the_state_it_came_in() {
        let trimmed = table(named(Pattern::default())).trims("content");
        assert_eq!(
            run(&trimmed, " Hi <c>x "),
            [
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Whitespace,
                },
                content("Hi <c>x"),
                Event::Dropped {
                    text: Text::uncounted(" "),
                    why: DropReason::Whitespace,
                },
                finish(0),
            ],
            "the space held before the terminal was inside the text"
        );
        let framed = table(named(Pattern::default())).framing("content");
        assert_eq!(
            run(&framed, "x<c>y<c><n=f>{}</c>"),
            [
                wrapper("x<c>y<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ]
        );
        let starts_in_thought = table(named(Pattern::default()))
            .framing("content")
            .starts_in("reasoning");
        assert_eq!(
            run(&starts_in_thought, "<c>plan</t>"),
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("<c>plan")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                finish(0),
            ],
            "a terminal that opens no call is the turn's first text, in the start row's state"
        );
        assert_eq!(
            run(&starts_in_thought, "<c><n=f>{}</c>"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "a call at the turn's first byte opens from the turn's own state"
        );
    }

    #[test]
    fn a_turns_first_terminal_opens_its_call_from_the_state_the_turn_starts_in() {
        // The call's row leaves the start state only, not the first one.
        let format = Format::new("shaped")
            .terminal("think_close", "</t>")
            .terminal("call_open", "<c>")
            .terminal("call_close", "</c>")
            .state("outside", Emits::Content)
            .state("reasoning", Emits::Reasoning)
            .state("call", Emits::Arguments)
            .transition("reasoning", "think_close", "outside")
            .transition("reasoning", "call_open", "call")
            .transition("call", "call_close", "outside")
            .calls(CallSyntax::Shaped(named(Pattern::default())))
            .framing("outside")
            .starts_in("reasoning");
        assert_eq!(
            run(&format, "<c><n=f>{}</c>"),
            [
                Event::ReasoningStart,
                wrapper("<c>"),
                Event::ReasoningEnd,
                start("f", "<n=f>"),
                arguments("{}"),
                end("</c>"),
                finish(1),
            ],
            "the turn starts where its first piece says, and the call opens from there"
        );
        assert_eq!(
            run(&format, "<c>x</t>"),
            [
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("<c>x")),
                wrapper("</t>"),
                Event::ReasoningEnd,
                finish(0),
            ],
            "a header that does not match leaves the terminal to the start state's text"
        );
    }

    #[test]
    fn a_shaped_call_has_no_structural_tag_yet() {
        let tools = [openai_protocol::common::Tool {
            tool_type: "function".to_string(),
            function: openai_protocol::common::Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        }];
        let format = table(named(Pattern::default()));
        for (at_least_one, reasoning_open) in [(true, false), (false, false), (true, true)] {
            assert_eq!(format.grammar(&tools, at_least_one, reasoning_open), None);
        }
    }

    #[test]
    fn a_closing_terminal_closes_only_where_the_trailer_ends_before_it() {
        let format = table(named(closing_f()));
        assert_eq!(
            run(&format, "<c><n=f>{\"a\": 1} </f>\n</c>Done."),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{\"a\": 1}"),
                wrapper(" "),
                end("</f>\n</c>"),
                content("Done."),
                finish(1),
            ]
        );
        let events = run_after(
            &format,
            "",
            "<c><n=f>{\"a\": 1}</c>x</f></c>",
            EngineFinish::Stop,
        );
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{\"a\": 1}"),
                Event::Malformed {
                    text: Text::uncounted("</c>x"),
                    why: MalformedReason::Other(TEXT_AFTER_THE_ARGUMENTS.to_string()),
                },
                end("</f></c>"),
                finish(1),
            ],
            "a close without the trailer is the call's text"
        );
        let events = run_after(&format, "", "<c><n=f>{}</c>", EngineFinish::Length);
        assert_eq!(
            events,
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                Event::Malformed {
                    text: Text::uncounted("</c>"),
                    why: MalformedReason::UnterminatedRegion,
                },
                end(""),
                Event::Finish {
                    reason: crate::event::FinishReason::Length,
                    tool_calls: 1,
                    reasoning_tokens: 0,
                },
            ],
            "a region the trailer never closed is cut by the end"
        );
    }

    #[test]
    fn a_stop_closes_the_call_where_the_trailer_ends_with_the_stream() {
        let format = table(named(closing_f())).closes_at_stop("call");
        assert_eq!(
            run(&format, "<c><n=f>{}</f>"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end("</f>"),
                finish(1),
            ]
        );
        assert_eq!(
            run(&format, "<c><n=f>{}"),
            [
                wrapper("<c>"),
                start("f", "<n=f>"),
                arguments("{}"),
                end(""),
                finish(1),
            ],
            "with no trailer the stop does not close the call, which is cut"
        );
    }

    #[test]
    fn an_envelope_names_its_call_and_a_literal_name_needs_no_header() {
        let envelope = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(["name"]),
                arguments: Path::new(["arguments"]),
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        assert_eq!(
            run(
                &table(envelope),
                "<c>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</c>"
            ),
            [
                wrapper("<c>"),
                start("f", "\n{\"name\": \"f\", \"arguments\": "),
                arguments("{\"a\": 1}"),
                end("}"),
                wrapper("\n</c>"),
                finish(1),
            ]
        );
        let literal = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Literal("lookup".into()),
            arguments: Arguments::Json(Dialect::default()),
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        assert_eq!(
            run(&table(literal), "<c>[1, 2]</c>"),
            [
                wrapper("<c>"),
                start("lookup", ""),
                arguments("[1, 2]"),
                end("</c>"),
                finish(1),
            ]
        );
    }

    #[test]
    fn an_envelope_after_the_marker_reads_as_the_json_syntax_does() {
        let envelope = intern(CallShape {
            header: Pattern::default(),
            name: NameSource::Body,
            arguments: Arguments::Envelope {
                dialect: Dialect::default(),
                name: Path::new(["name"]),
                arguments: Path::new(["arguments"]),
            },
            trailer: Pattern::default(),
            word: word(),
            hold_cap: 64,
        });
        let json = crate::formats::qwen3(CallSyntax::Json);
        let shaped = crate::formats::qwen3(CallSyntax::Shaped(envelope));
        for output in [
            "<think>\nplan\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"get_weather\", \
             \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
            "<tool_call>junk</tool_call><tool_call>{\"name\": \"f\", \"arguments\": {}}\
             </tool_call>",
            "<tool_call>{\"arguments\": {\"a\": 1}, \"name\": \"f\"}</tool_call>",
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": [1, 2",
            "<tool_call>{\"name\": \"f\", \"arguments\": {}}x y</tool_call>",
            "<tool_call>{\"name\": \"f\"}<tool_call>{\"name\": \"g\", \"arguments\": 12abc}",
            "<tool_call>{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        ] {
            for finish in [EngineFinish::Stop, EngineFinish::Length] {
                assert_eq!(
                    run_after(&shaped, "", output, finish.clone()),
                    run_after(&json, "", output, finish),
                    "{output:?}"
                );
            }
        }
    }

    #[test]
    fn tokens_given_back_are_counted_in_the_outputs_order() {
        let format = table(named(Pattern::default()));
        let text = "a<c>b<c><n=f>{}</c>";
        // One token per character but the markers, each one token.
        let mut spans = Vec::new();
        let mut at = 0;
        for (token_id, piece) in (0..).zip([
            "a", "<c>", "b", "<c>", "<", "n", "=", "f", ">", "{", "}", "</c>",
        ]) {
            spans.push(TokenSpan {
                token_id,
                start: at,
                end: at + piece.len(),
                continued: false,
            });
            at += piece.len();
        }
        let mut engine = Engine::new(format, Declared::default());
        let mut out = Events::new();
        for input in [
            Input::Delta {
                token_ids: &[],
                text,
                spans: &spans,
            },
            Input::End {
                finish: EngineFinish::Stop,
            },
        ] {
            engine.feed(input, &mut out).expect("in order");
        }
        let counted: Vec<(String, Option<u32>)> = out
            .drain()
            .into_iter()
            .filter_map(|event| match event {
                Event::Content(text) | Event::Dropped { text, .. } => {
                    Some((text.text, text.tokens))
                }
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => Some((source.text, source.tokens)),
                _ => None,
            })
            .collect();
        let as_said: Vec<(&str, Option<u32>)> = counted
            .iter()
            .map(|(text, tokens)| (text.as_str(), *tokens))
            .collect();
        assert_eq!(
            as_said,
            [
                ("a<c>b", Some(3)),
                ("<c>", Some(1)),
                ("<n=f>", Some(5)),
                ("{}", Some(2)),
                ("</c>", Some(1)),
            ],
            "the given-back terminal's token counts where its first byte is shown"
        );
    }
}
