//! Addressed calls: the events of one call written as a message addressed to the tool, its name
//! in the message's header and its arguments as tagged parameters, from its bytes.
//!
//! The syntax puts together two shapes: Harmony's addressing, in which a call is a message whose
//! recipient is the tool and whose header carries the tool's name, and tagged parameters in place
//! of Harmony's JSON arguments. A call is the recipient marker, the tool's name, a fixed spelling
//! that closes the header, and then one tag pair per argument, the value as plain text between
//! them, with the template's newlines around it. In the synthetic spelling the tests use:
//!
//! ```text
//! <|start|>assistant to=get_weather<|channel|>commentary<|message|><parameter name="city">
//! Paris
//! </parameter>
//! <parameter name="days">
//! 3
//! </parameter><|call|>
//! ```
//!
//! The engine's table owns the markers around the message: it enters this assembler after the
//! recipient marker (` to=` here) and leaves it at the marker that ends the message, which it hands
//! over as the call's end. Between the two the assembler reads the name, the header's close and the
//! parameters, each in the family's spelling ([`Spelling`]).
//!
//! What the syntax says, and what the assembler does with it:
//!
//! - **The name is the text between the recipient marker and the header's close**, exactly as a
//!   reader of this syntax takes it: `to=(?P<name>[^\s<]+)` with the header's close right after
//!   it. `ToolCallStart` waits for the header's close, which follows the name at once; names are
//!   short, and the name and the header are its source. A name holds no whitespace and no `<`, and
//!   any other character is the name's, a quote, a brace or `>` among them, since the header's
//!   close that follows says where the name ends. Whitespace before or after the name makes the
//!   header no call's, as it does for that reader. Any other text there is reported, and no call
//!   starts: the message is reported, every byte of it, and the adapters show `Malformed` text as
//!   content, where a reader that matches the header with that pattern leaves the message out.
//!   A header that does not close before a parameter tag or the message's end starts no
//!   call either, so a header the template never writes is never read as a call; one the stream
//!   was cut inside is the template's, cut short, and is dropped, since the model was still
//!   writing the markers the template writes there and nothing in it is the model's text.
//! - **A value's type comes from the request's tools**, as in the Qwen assembler: a declared
//!   string is its text, streamed as it arrives; every other value is written whole at its closing
//!   tag through [`json`], which reads the JSON a template writes for a number, a boolean, a list,
//!   an object or null, and infers the rest.
//! - **The template writes one newline after the key's tag and one before the value's closing
//!   tag**; neither is the value's, and the assembler takes exactly those two away, as the Qwen
//!   assembler does. A newline at the end of a streamed piece is held back until the next piece
//!   says whether the closing tag follows it.
//! - **Inside a value only the value's closing tag is a tag**; the header's close and the key's
//!   tags there are the value's text, so a value that ends in `>` or holds `">` keeps them. A
//!   spelling can say that the next key closes a value too ([`Spelling::next_key_closes`]): a
//!   value the model left unclosed then ends at a newline followed by the key's opening tag, the
//!   newline being the template's, as it is before the closing tag; a reader whose pattern ends a
//!   value at a newline before either tag reads it so. The message's own markers are the
//!   engine's terminals and end the call wherever they stand.
//! - **A key the call already has is not taken again.** The call keeps the value that came first,
//!   and the later parameter, its tags with it, is reported whole once its value ends, or the
//!   call ends inside it: an arguments object holds each key once, so the arguments every API
//!   returns stay one JSON object with no key twice. A reader that collects the parameters into a
//!   dictionary keeps the last value instead.
//! - **Two endings.** [`Assembler::close`] is the marker that ends the message, or the stream's
//!   stop where the table says the stop closes the call: after the last value it closes the
//!   object and pushes `ToolCallEnd` with the marker's bytes; inside a value it closes the value
//!   as its closing tag would, a string with its quote and any other value written whole, then
//!   the object. [`Assembler::finish`] is a stream that was cut: nothing is closed, and a value
//!   the cut stopped keeps what arrived, a string's fragment open and any other value written as
//!   its text reads so far, with the object left open. Either way the arguments keep every value
//!   that arrived, whatever its type, and no tag reaches the content: a tag the end cut short (a
//!   key with no value yet, the start of a closing tag) is the template's and goes with the bytes
//!   around it, into the object's close, or, at a cut, `Dropped { Wrapper }`. Bytes the end left
//!   held inside a value that cannot be its closing tag are the value's text.
//! - **Every byte of the call lands in exactly one event**, with the Qwen assembler's accounting:
//!   the name's and the header's bytes in `ToolCallStart`, each tag's and value's bytes in the
//!   fragments they produce. Text where the syntax has tags, and a tag out of its place, are
//!   `Malformed`, and so is the whitespace inside such a run: `stray words` is reported as the
//!   model wrote it, its space included. Whitespace beside a tag the call takes (the template's
//!   newline after a value's closing tag or before a key's opening one) is the template's: it
//!   goes into the tag's event, or is `Dropped { Wrapper }` before a report that follows a tag.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    markers::{Piece, Scanner},
    tagged::{
        assembler::{push_escaped, push_quoted},
        value::{json, Declared, Kind, NULL_WORDS},
    },
};

const HEADER_CLOSE: usize = 0;
const KEY_OPEN: usize = 1;
const KEY_CLOSE: usize = 2;
const VALUE_CLOSE: usize = 3;
const TEXT_BETWEEN_TAGS: &str = "text between a call's tags";
const TAG_OUT_OF_PLACE: &str = "a tag where the call's syntax has none";
const TAG_CUT_SHORT: &str = "a tag that another tag cut short";
const EMPTY_KEY: &str = "a key with no text";
const EMPTY_NAME: &str = "a call header without a name";
const NOT_A_NAME: &str = "text where the call's name should be";
const HEADER_OPEN: &str = "a call header that did not close";
const WITHOUT_A_CALL: &str = "a call message that named no call";
const REPEATED_KEY: &str = "a parameter whose key the call already has";

/// How a family spells an addressed call: what closes the header after the name, and the three
/// tags around a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spelling {
    /// What follows the name and closes the message's header: `<|channel|>commentary<|message|>`
    /// in the example above.
    pub header_close: &'static str,
    /// What opens a parameter's key: `<parameter name="`.
    pub key_open: &'static str,
    /// What closes the key and opens the value: `">`.
    pub key_close: &'static str,
    /// What closes the value: `</parameter>`.
    pub value_close: &'static str,
    /// Whether a newline and then [`Self::key_open`] close a value as a newline and
    /// [`Self::value_close`] do: a parameter the model left unclosed ends where the next one
    /// opens. Otherwise the key's opening tag inside a value is the value's text.
    pub next_key_closes: bool,
}

impl Spelling {
    fn text(&self, tag: usize) -> &'static str {
        [
            self.header_close,
            self.key_open,
            self.key_close,
            self.value_close,
        ][tag]
    }
}

/// The events of one addressed call, from the bytes after the recipient marker.
#[derive(Clone, Debug)]
pub struct Assembler {
    index: u32,
    id: String,
    spelling: Spelling,
    scanner: Scanner,
    /// Bytes no event has accounted for yet, in order; the next event takes them as its source.
    carried: String,
    stage: Stage,
    /// The function's name once `ToolCallStart` was pushed.
    function: Option<String>,
    /// Members written to the arguments object so far.
    written: u32,
    /// The keys whose values the call has taken, in order.
    keys: Vec<String>,
    /// The last bytes read were reported as `Malformed`, and no tag the call takes came since:
    /// whitespace that follows is held, and joins the next report when more text or a tag out of
    /// its place follows it, since it then stands inside what the model wrote there.
    malformed: bool,
}

#[derive(Clone, Debug)]
enum Stage {
    /// Before the header's close: the name is the carried text, less the whitespace around it.
    Name,
    /// After the header or a value, before the next key; also after a header that named no call.
    Between,
    /// Inside a parameter's key: the key is `carried[start..]` until the key's close.
    Key { start: usize },
    /// Inside a value.
    Value(ValueState),
}

#[derive(Clone, Debug)]
struct ValueState {
    key: String,
    kind: Option<Kind>,
    /// Whether the newline the template writes after the key's tag may still come: no value text
    /// yet.
    leading: bool,
    mode: Mode,
    /// The call already has the key: the parameter, from its key's opening tag, is carried whole
    /// (`Mode::Whole`) and reported at the value's end instead of written.
    repeated: bool,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Pushed piece by piece; `held_newline` says `carried` ends with a newline that may be the
    /// template's.
    Streaming { held_newline: bool },
    /// Written at the close with [`json`]; the text is `carried[start..]`.
    Whole { start: usize },
    /// A string that may be null: written whole if its text is a null word, streamed from the
    /// first byte that rules the words out; the text is `carried[start..]` until then.
    Undecided { start: usize },
}

impl Assembler {
    /// An assembler for the call at `index` with the id the format minted for it, positioned just
    /// after the recipient marker, in the family's `spelling`.
    pub fn new(index: u32, id: impl Into<String>, spelling: Spelling) -> Self {
        Self {
            index,
            id: id.into(),
            scanner: Scanner::new([
                spelling.header_close,
                spelling.key_open,
                spelling.key_close,
                spelling.value_close,
            ]),
            spelling,
            carried: String::new(),
            stage: Stage::Name,
            function: None,
            written: 0,
            keys: Vec::new(),
            malformed: false,
        }
    }

    /// Whether a call has started, that is, whether `ToolCallStart` has been pushed.
    pub fn started(&self) -> bool {
        self.function.is_some()
    }

    /// Append the next bytes of the call and push the events they complete. Every byte is the
    /// call's until the engine says otherwise, so this takes them all.
    pub fn feed(&mut self, bytes: &str, declared: &Declared, out: &mut Events) {
        for piece in self.scanner.feed(bytes) {
            match piece {
                Piece::Text(text) => self.text(&text, out),
                Piece::Marker(tag) => self.tag(tag, declared, out),
            }
        }
    }

    /// The call's end: `terminal` is the marker that ended the message, whose bytes go into
    /// `ToolCallEnd`, or nothing when the stream's stop ended it. After the last value the object
    /// is closed; inside a value the value is closed as its closing tag would close it, then the
    /// object; a key or a tag the end cut short is the template's and goes into the object's
    /// close; a message whose header never closed, or that named no call, has its bytes reported,
    /// the marker's with them, and no call.
    pub fn close(mut self, terminal: &str, out: &mut Events) {
        let held = self.scanner.held().to_string();
        if !self.started() {
            self.carried.push_str(&held);
            self.carried.push_str(terminal);
            let why = if matches!(self.stage, Stage::Name) {
                HEADER_OPEN
            } else {
                WITHOUT_A_CALL
            };
            self.report(why, out);
            return;
        }
        if matches!(self.stage, Stage::Value(_)) {
            let tag = self.end_value(&held, out);
            self.close_value(&tag, out);
            self.close_object(Text::default(), out);
        } else {
            // The whitespace before the end, a key with no value yet and a tag the end cut short
            // are the template's: they go into the object's close.
            self.carried.push_str(&held);
            let source = Text::uncounted(std::mem::take(&mut self.carried));
            self.close_object(source, out);
        }
        self.stage = Stage::Between;
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::uncounted(terminal),
        });
    }

    /// The stream was cut: nothing is closed, so arguments cut short never look complete to a
    /// client. A value the cut stopped keeps what arrived: a streamed string's fragment stays
    /// open, and any other value is written as its text reads so far, with the object left open.
    /// A key or a tag the cut stopped is the template's and is `Dropped { Wrapper }`, and so is a
    /// header the cut stopped before its close (the name so far and a part of the close): the
    /// model was writing the markers that open a call, and none of it reaches the content. Bytes
    /// of a message whose header closed and named no call come back as
    /// `Malformed { UnterminatedRegion }`. A call that started still ends, with no bytes of its
    /// own, as every assembler ends one.
    pub fn finish(mut self, out: &mut Events) {
        let held = self.scanner.held().to_string();
        if !self.started() {
            self.carried.push_str(&held);
            if self.carried.is_empty() {
                return;
            }
            let text = Text::uncounted(std::mem::take(&mut self.carried));
            out.push(if matches!(self.stage, Stage::Name) {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                }
            } else {
                Event::Malformed {
                    text,
                    why: MalformedReason::UnterminatedRegion,
                }
            });
            return;
        }
        let mut tag = held;
        if matches!(self.stage, Stage::Value(_)) {
            tag = self.end_value(&tag, out);
            if let Stage::Value(ValueState { repeated: true, .. }) = self.stage {
                // A parameter whose key the call already has, which the cut stopped: reported
                // whole, the held start of a tag with it.
                self.carried.push_str(&tag);
                self.report(REPEATED_KEY, out);
                tag.clear();
            } else if let Stage::Value(ValueState {
                mode: Mode::Whole { start } | Mode::Undecided { start },
                ..
            }) = self.stage
            {
                let member = self.member(start);
                let mut source = std::mem::take(&mut self.carried);
                source.push_str(&tag);
                self.push_fragment(member, Text::uncounted(source), out);
                tag.clear();
            }
        }
        self.carried.push_str(&tag);
        self.drop_carried(out);
        out.push(Event::ToolCallEnd {
            index: self.index,
            source: Text::default(),
        });
    }

    /// Bytes the end of the call left held inside a value: the start of the value's closing tag,
    /// returned to go with the value's close, or, when they cannot be that, the value's text.
    fn end_value(&mut self, held: &str, out: &mut Events) -> String {
        if self.spelling.value_close.starts_with(held) {
            return held.to_string();
        }
        self.value_text(held, out);
        String::new()
    }

    fn tag(&mut self, tag: usize, declared: &Declared, out: &mut Events) {
        let bytes = self.spelling.text(tag);
        match &self.stage {
            Stage::Name if tag == HEADER_CLOSE => self.name_call(bytes, out),
            Stage::Name => {
                // A parameter tag before the header closed: the header is reported, no call
                // starts, and the tag is read as between keys.
                self.report(HEADER_OPEN, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
            Stage::Between => self.tag_between(tag, out),
            Stage::Key { start } if tag == KEY_CLOSE => {
                let start = *start;
                let key = self.carried[start..].to_string();
                self.carried.push_str(bytes);
                if key.is_empty() {
                    self.report(EMPTY_KEY, out);
                    self.stage = Stage::Between;
                } else if self.keys.contains(&key) {
                    self.repeat(start, key, out);
                } else {
                    self.keys.push(key.clone());
                    self.malformed = false;
                    self.open_value(key, declared, out);
                }
            }
            // A tag cuts the key short: its bytes so far are reported, and the tag is read as
            // between keys.
            Stage::Key { .. } => {
                self.report(TAG_CUT_SHORT, out);
                self.stage = Stage::Between;
                self.tag_between(tag, out);
            }
            Stage::Value(_) if tag == VALUE_CLOSE => self.close_value(bytes, out),
            // The next key closes the value where the spelling says so, after a newline that may
            // be the template's; the key's tag then opens the next parameter.
            Stage::Value(_) if tag == KEY_OPEN && self.closed_by_next_key() => {
                self.close_value_at_next_key(out);
                self.tag_between(tag, out);
            }
            // Inside a value, the other tags are the value's text.
            Stage::Value(_) => self.value_text(bytes, out),
        }
    }

    /// Whether a key's opening tag that arrives now closes the value: the spelling says the next
    /// key closes one, and the value's text so far ends with a newline that may be the
    /// template's, a newline after the one the template writes after the key's tag.
    fn closed_by_next_key(&self) -> bool {
        let Stage::Value(value) = &self.stage else {
            return false;
        };
        self.spelling.next_key_closes
            && match value.mode {
                Mode::Streaming { held_newline } => held_newline,
                Mode::Whole { start } | Mode::Undecided { start } => {
                    !value.leading && self.carried[start..].ends_with('\n')
                }
            }
    }

    /// The next key closed the value, after a newline that is the template's. A value the call
    /// takes is closed as its closing tag would close it, with the newline as the close's bytes. A
    /// parameter whose key the call already has is reported without the newline, which goes with
    /// the next key's tag, as whitespace before a tag the call takes does.
    fn close_value_at_next_key(&mut self, out: &mut Events) {
        if let Stage::Value(ValueState { repeated: true, .. }) = self.stage {
            self.carried.pop();
            self.close_value("", out);
            self.carried.push('\n');
        } else {
            self.close_value("", out);
        }
    }

    /// The key's tag closed on a key the call already has: the parameter, from its key's opening
    /// tag, is carried whole until its value ends, to be reported then. The whitespace before that
    /// tag goes as whitespace before any report goes ([`Self::report_text`]).
    fn repeat(&mut self, key_start: usize, key: String, out: &mut Events) {
        let parameter = self
            .carried
            .split_off(key_start - self.spelling.key_open.len());
        if !self.malformed {
            self.drop_carried(out);
        }
        self.carried.push_str(&parameter);
        self.stage = Stage::Value(ValueState {
            key,
            kind: None,
            leading: true,
            mode: Mode::Whole {
                start: self.carried.len(),
            },
            repeated: true,
        });
    }

    /// A tag between keys: a key's opening tag opens one, once the call has started, with the
    /// whitespace before it as its own; the other three, and any tag of a message that named no
    /// call, have no place here and are reported.
    fn tag_between(&mut self, tag: usize, out: &mut Events) {
        let bytes = self.spelling.text(tag);
        if tag == KEY_OPEN && self.started() {
            self.carried.push_str(bytes);
            self.malformed = false;
            self.stage = Stage::Key {
                start: self.carried.len(),
            };
        } else {
            self.report_text(bytes, TAG_OUT_OF_PLACE, out);
        }
    }

    fn text(&mut self, text: &str, out: &mut Events) {
        match &self.stage {
            Stage::Name | Stage::Key { .. } => self.carried.push_str(text),
            Stage::Between => self.text_between(text, out),
            Stage::Value(_) => self.value_text(text, out),
        }
    }

    /// Text where the syntax has tags: anything but whitespace is reported, and whitespace waits
    /// with the carried bytes for what follows it to say whose it is ([`Self::report_text`]).
    fn text_between(&mut self, text: &str, out: &mut Events) {
        let mut rest = text;
        while let Some(first) = rest.chars().next() {
            let space = first.is_whitespace();
            let length = rest
                .char_indices()
                .find(|(_, c)| c.is_whitespace() != space)
                .map_or(rest.len(), |(at, _)| at);
            if space {
                self.carried.push_str(&rest[..length]);
            } else {
                self.report_text(&rest[..length], TEXT_BETWEEN_TAGS, out);
            }
            rest = &rest[length..];
        }
    }

    /// Reports `text`, text or a tag where the syntax has no place for it, as `Malformed` with
    /// `why`. The whitespace carried before it is inside what the model wrote there when a report
    /// came just before it, and is reported with `text`; otherwise it follows a tag the call took,
    /// and is the template's, `Dropped { Wrapper }`.
    fn report_text(&mut self, text: &str, why: &str, out: &mut Events) {
        let mut run = String::new();
        if self.malformed {
            run = std::mem::take(&mut self.carried);
        } else {
            self.drop_carried(out);
        }
        run.push_str(text);
        out.push(Event::Malformed {
            text: Text::uncounted(run),
            why: MalformedReason::Other(why.to_string()),
        });
        self.malformed = true;
    }

    /// The header closed: the text before it is the call's name, and the call starts, with the
    /// name and the header as its source. A name with no text, or text that is not a name,
    /// whitespace around it included, is reported with the header, and no call starts.
    fn name_call(&mut self, header: &str, out: &mut Events) {
        let name = self.carried.clone();
        self.carried.push_str(header);
        self.stage = Stage::Between;
        if name.is_empty() {
            self.report(EMPTY_NAME, out);
            return;
        }
        if !is_name(&name) {
            self.report(NOT_A_NAME, out);
            return;
        }
        out.push(Event::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: name.clone(),
            source: Text::uncounted(std::mem::take(&mut self.carried)),
        });
        self.function = Some(name);
        self.malformed = false;
    }

    /// The key's tag is whole: the value begins, typed by what the tool declares for it. A
    /// declared string opens its fragment now.
    fn open_value(&mut self, key: String, declared: &Declared, out: &mut Events) {
        let function = self.function.as_deref().unwrap_or_default();
        let kind = declared.kind(function, &key);
        let mode = match kind {
            Some(Kind::String) => {
                self.open_string_fragment(&key, out);
                Mode::Streaming {
                    held_newline: false,
                }
            }
            Some(Kind::NullableString) => Mode::Undecided {
                start: self.carried.len(),
            },
            Some(Kind::Integer | Kind::Array | Kind::Object) | None => Mode::Whole {
                start: self.carried.len(),
            },
        };
        self.stage = Stage::Value(ValueState {
            key,
            kind,
            leading: true,
            mode,
            repeated: false,
        });
    }

    /// `{"key": "` or `, "key": "`, with the carried bytes as its source.
    fn open_string_fragment(&mut self, key: &str, out: &mut Events) {
        let mut opening = String::with_capacity(key.len() + 8);
        opening.push_str(self.separator());
        push_quoted(&mut opening, key);
        opening.push_str(": \"");
        let source = Text::uncounted(std::mem::take(&mut self.carried));
        self.push_fragment(opening, source, out);
        self.written += 1;
    }

    fn value_text(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let mut text = text;
        if value.leading && !text.is_empty() {
            value.leading = false;
            if let Some(rest) = text.strip_prefix('\n') {
                self.carried.push('\n');
                if let Mode::Whole { start } | Mode::Undecided { start } = &mut value.mode {
                    *start = self.carried.len();
                }
                text = rest;
            }
        }
        if text.is_empty() {
            return;
        }
        match value.mode {
            Mode::Streaming { .. } => self.stream(text, out),
            Mode::Whole { .. } => self.carried.push_str(text),
            Mode::Undecided { start } => {
                self.carried.push_str(text);
                let so_far = self.whole_text(start);
                if !NULL_WORDS.iter().any(|word| word.starts_with(so_far)) {
                    self.open_string_late(start, out);
                }
            }
        }
    }

    /// A string that may have been null is not: open it now, then stream what has arrived.
    fn open_string_late(&mut self, start: usize, out: &mut Events) {
        let arrived = self.carried.split_off(start);
        let Stage::Value(value) = &mut self.stage else {
            return;
        };
        let key = value.key.clone();
        value.mode = Mode::Streaming {
            held_newline: false,
        };
        self.open_string_fragment(&key, out);
        self.stream(&arrived, out);
    }

    /// A piece of a streamed string. A newline the piece ends with is held, since it may be the
    /// template's; a newline held before it turns out to be the value's.
    fn stream(&mut self, text: &str, out: &mut Events) {
        let Stage::Value(ValueState {
            mode: Mode::Streaming { held_newline },
            ..
        }) = &mut self.stage
        else {
            return;
        };
        let had_newline = *held_newline;
        let (body, ends_with_newline) = match text.strip_suffix('\n') {
            Some(body) => (body, true),
            None => (text, false),
        };
        *held_newline = ends_with_newline;
        if body.is_empty() && !had_newline {
            // The piece is one newline; it waits with the carried bytes for the piece that follows.
            self.carried.push('\n');
            return;
        }
        let mut json = String::with_capacity(body.len() + 2);
        if had_newline {
            json.push_str("\\n");
        }
        push_escaped(&mut json, body);
        // The source: the carried bytes (the template's newline after the tag, or the newline that
        // was held and is now the value's) and the piece, less the newline held in its turn.
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(body);
        if ends_with_newline {
            self.carried.push('\n');
        }
        self.push_fragment(json, Text::uncounted(source), out);
    }

    /// The value's closing tag: a streamed string gets its closing quote, any other value is
    /// written whole, and a parameter whose key the call already has is reported whole, the tag
    /// with it.
    fn close_value(&mut self, tag: &str, out: &mut Events) {
        let Stage::Value(value) = &self.stage else {
            return;
        };
        if value.repeated {
            self.carried.push_str(tag);
            self.report(REPEATED_KEY, out);
            self.stage = Stage::Between;
            return;
        }
        let fragment = match value.mode {
            Mode::Streaming { .. } => "\"".to_string(),
            Mode::Whole { start } | Mode::Undecided { start } => self.member(start),
        };
        let mut source = std::mem::take(&mut self.carried);
        source.push_str(tag);
        self.push_fragment(fragment, Text::uncounted(source), out);
        self.malformed = false;
        self.stage = Stage::Between;
    }

    /// The member of a value written whole, its text from `start`: `{"key": value` or
    /// `, "key": value`.
    fn member(&mut self, start: usize) -> String {
        let Stage::Value(value) = &self.stage else {
            return String::new();
        };
        let written = json(self.whole_text(start), value.kind);
        let mut member = String::with_capacity(value.key.len() + written.len() + 8);
        member.push_str(self.separator());
        push_quoted(&mut member, &value.key);
        member.push_str(": ");
        member.push_str(&written);
        self.written += 1;
        member
    }

    /// The text of a value written whole: what came after the tag and its newline, less the
    /// newline the template writes before the closing tag.
    fn whole_text(&self, start: usize) -> &str {
        let text = &self.carried[start..];
        text.strip_suffix('\n').unwrap_or(text)
    }

    /// Reports the carried bytes as `Malformed` with `why`, whitespace included.
    fn report(&mut self, why: &str, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Malformed {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: MalformedReason::Other(why.to_string()),
        });
        self.malformed = true;
    }

    /// Whitespace carried after a tag the call took, before text that is reported: the
    /// template's, dropped so the report holds only what the model wrote out of place.
    fn drop_carried(&mut self, out: &mut Events) {
        if self.carried.is_empty() {
            return;
        }
        out.push(Event::Dropped {
            text: Text::uncounted(std::mem::take(&mut self.carried)),
            why: DropReason::Wrapper,
        });
    }

    /// `, ` before every member but the first, which opens the object.
    fn separator(&self) -> &'static str {
        if self.written == 0 {
            "{"
        } else {
            ", "
        }
    }

    /// `}` or `{}`, with `source` as its bytes.
    fn close_object(&mut self, source: Text, out: &mut Events) {
        let json = if self.written == 0 { "{}" } else { "}" }.to_string();
        self.push_fragment(json, source, out);
    }

    fn push_fragment(&self, json: String, source: Text, out: &mut Events) {
        out.push(Event::ToolCallArguments {
            index: self.index,
            json,
            source,
        });
    }
}

/// Whether `text`, all of the text between the recipient marker and the header's close, is an
/// addressed call's name: what a reader of the syntax takes there, `to=(?P<name>[^\s<]+)` with the
/// header's close right after it, so no whitespace, before, inside or after it, and no `<`. The
/// header's close follows the name, so a quote, a brace or `>` in it says nothing about where the
/// name ends.
fn is_name(text: &str) -> bool {
    !text.chars().any(|c| c.is_whitespace() || c == '<')
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::{Function, Tool};
    use serde_json::json as value;

    use super::*;

    /// A synthetic spelling: Harmony's commentary channel closes the header, and each parameter
    /// is a named tag.
    const SPELLING: Spelling = Spelling {
        header_close: "<|channel|>commentary<|message|>",
        key_open: "<parameter name=\"",
        key_close: "\">",
        value_close: "</parameter>",
        next_key_closes: false,
    };
    /// The same spelling where the next key closes a value the model left unclosed.
    const NEXT_KEY_CLOSES: Spelling = Spelling {
        next_key_closes: true,
        ..SPELLING
    };
    const END: &str = "<|call|>";

    /// A tool `get_weather` with a string `city`, an integer `days` and a nullable string `note`.
    fn declared() -> Declared {
        Declared::of(&[Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: value!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                }}),
                strict: None,
                extra: Default::default(),
            },
        }])
    }

    const CALL: &str = concat!(
        "get_weather<|channel|>commentary<|message|><parameter name=\"city\">\nParis\n",
        "</parameter>\n<parameter name=\"days\">\n3\n</parameter>"
    );

    /// The events of one call fed in `pieces`, ended by the message's marker, or cut.
    fn run(pieces: &[&str], cut: bool) -> Vec<Event> {
        run_spelled(SPELLING, pieces, cut)
    }

    /// [`run`] in `spelling`.
    fn run_spelled(spelling: Spelling, pieces: &[&str], cut: bool) -> Vec<Event> {
        let declared = declared();
        let mut assembler = Assembler::new(0, "call_0", spelling);
        let mut out = Events::new();
        for piece in pieces {
            assembler.feed(piece, &declared, &mut out);
        }
        if cut {
            assembler.finish(&mut out);
        } else {
            assembler.close(END, &mut out);
        }
        out.drain()
    }

    fn bytes(events: &[Event]) -> String {
        let mut list = Events::new();
        for event in events {
            list.push(event.clone());
        }
        list.text()
    }

    fn fragments(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    fn arguments(events: &[Event]) -> String {
        fragments(events).concat()
    }

    fn name_of(events: &[Event]) -> Option<&str> {
        events.iter().find_map(|event| match event {
            Event::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
    }

    fn malformed(events: &[Event]) -> Vec<(&str, &str)> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed {
                    text,
                    why: MalformedReason::Other(why),
                } => Some((text.text.as_str(), why.as_str())),
                Event::Malformed { text, .. } => Some((text.text.as_str(), "unterminated")),
                _ => None,
            })
            .collect()
    }

    fn ends(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallEnd { source, .. } => Some(source.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn fragment(json: &str, source: &str) -> Event {
        Event::ToolCallArguments {
            index: 0,
            json: json.to_string(),
            source: Text::uncounted(source),
        }
    }

    /// No fragment is empty and no `Malformed` or `Dropped` text is empty.
    fn nothing_empty(events: &[Event]) {
        for event in events {
            match event {
                Event::ToolCallArguments { json, .. } => assert!(!json.is_empty(), "{event:?}"),
                Event::Malformed { text, .. } | Event::Dropped { text, .. } => {
                    assert!(!text.text.is_empty(), "{event:?}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn the_name_starts_the_call_at_the_header_and_a_declared_string_streams() {
        let events = run(&[CALL], false);
        assert_eq!(
            events,
            vec![
                Event::ToolCallStart {
                    index: 0,
                    id: "call_0".to_string(),
                    name: "get_weather".to_string(),
                    source: Text::uncounted("get_weather<|channel|>commentary<|message|>"),
                },
                fragment("{\"city\": \"", "<parameter name=\"city\">"),
                fragment("Paris", "\nParis"),
                fragment("\"", "\n</parameter>"),
                fragment(
                    ", \"days\": 3",
                    "\n<parameter name=\"days\">\n3\n</parameter>"
                ),
                fragment("}", ""),
                Event::ToolCallEnd {
                    index: 0,
                    source: Text::uncounted(END),
                },
            ]
        );
        assert_eq!(arguments(&events), r#"{"city": "Paris", "days": 3}"#);
        assert_eq!(bytes(&events), format!("{CALL}{END}"));
    }

    #[test]
    fn every_chunking_gives_the_same_call_and_accounts_for_every_byte() {
        let whole = run(&[CALL], false);
        for cut in 1..CALL.len() {
            if !CALL.is_char_boundary(cut) {
                continue;
            }
            let events = run(&[&CALL[..cut], &CALL[cut..]], false);
            assert_eq!(arguments(&events), arguments(&whole), "cut at {cut}");
            assert_eq!(name_of(&events), Some("get_weather"), "cut at {cut}");
            assert_eq!(bytes(&events), format!("{CALL}{END}"), "cut at {cut}");
            assert!(malformed(&events).is_empty(), "cut at {cut}");
            nothing_empty(&events);
        }
        let pieces: Vec<&str> = CALL
            .char_indices()
            .map(|(at, c)| &CALL[at..at + c.len_utf8()])
            .collect();
        let events = run(&pieces, false);
        assert_eq!(arguments(&events), arguments(&whole));
        assert_eq!(bytes(&events), format!("{CALL}{END}"));
        nothing_empty(&events);
    }

    #[test]
    fn a_value_keeps_its_own_newlines_and_loses_the_templates_two() {
        let call = concat!(
            "get_weather<|channel|>commentary<|message|>",
            "<parameter name=\"city\">\n\nline one\nline two\n\n</parameter>"
        );
        let events = run(&[call], false);
        assert_eq!(arguments(&events), r#"{"city": "\nline one\nline two\n"}"#);
        assert_eq!(bytes(&events), format!("{call}{END}"));
        // The same with the value split after each newline, where the hold-back decides.
        let events = run(
            &[
                "get_weather<|channel|>commentary<|message|><parameter name=\"city\">",
                "\n",
                "\n",
                "line one\n",
                "line two\n",
                "\n",
                "</parameter>",
            ],
            false,
        );
        assert_eq!(arguments(&events), r#"{"city": "\nline one\nline two\n"}"#);
        assert_eq!(bytes(&events), format!("{call}{END}"));
        nothing_empty(&events);
    }

    #[test]
    fn inside_a_value_only_its_closing_tag_is_a_tag() {
        // A value that ends in `>` and holds the key's closing spelling, the header's close and a
        // key's opening: all of it is the value's text.
        let value_text = concat!(
            "<p class=\"x\">a</p><|channel|>commentary<|message|>",
            "<parameter name=\"y\">"
        );
        let call = format!(
            "get_weather<|channel|>commentary<|message|><parameter name=\"city\">\n{value_text}\n\
             </parameter>"
        );
        let whole = run(&[&call], false);
        let read: serde_json::Value =
            serde_json::from_str(&arguments(&whole)).expect("the arguments are JSON");
        assert_eq!(read["city"], value_text);
        assert!(malformed(&whole).is_empty());
        for cut in 1..call.len() {
            let events = run(&[&call[..cut], &call[cut..]], false);
            assert_eq!(arguments(&events), arguments(&whole), "cut at {cut}");
            assert_eq!(bytes(&events), format!("{call}{END}"), "cut at {cut}");
        }
    }

    #[test]
    fn values_are_typed_by_the_tools_and_inferred_for_a_function_they_do_not_declare() {
        for (text, expected) in [
            ("null", "null"),
            ("None", "null"),
            ("no rain", r#""no rain""#),
            ("", r#""""#),
        ] {
            let call = format!(
                "get_weather<|channel|>commentary<|message|>\
                 <parameter name=\"note\">\n{text}\n</parameter>"
            );
            assert_eq!(
                arguments(&run(&[&call], false)),
                format!(r#"{{"note": {expected}}}"#),
                "{text:?}"
            );
        }
        // A declared string keeps text that reads as another type.
        let call = "get_weather<|channel|>commentary<|message|>\
                    <parameter name=\"city\">\n007\n</parameter>";
        assert_eq!(arguments(&run(&[call], false)), r#"{"city": "007"}"#);
        let call = concat!(
            "lookup<|channel|>commentary<|message|><parameter name=\"q\">\nParis\n</parameter>\n",
            "<parameter name=\"n\">\n5\n</parameter>\n<parameter name=\"o\">\n",
            "{\"a\": [1, true, null]}\n</parameter>"
        );
        let events = run(&[call], false);
        assert_eq!(name_of(&events), Some("lookup"));
        assert_eq!(
            arguments(&events),
            r#"{"q": "Paris", "n": 5, "o": {"a": [1, true, null]}}"#
        );
    }

    #[test]
    fn a_call_without_parameters_is_an_empty_object() {
        let events = run(&["ping<|channel|>commentary<|message|>"], false);
        assert_eq!(name_of(&events), Some("ping"));
        assert_eq!(fragments(&events), ["{}"]);
        assert_eq!(ends(&events), [END]);
    }

    #[test]
    fn a_header_that_does_not_close_starts_no_call_and_reports_every_byte() {
        // Another channel than the spelling's, then a parameter; and a message that ends before
        // its header closes.
        for (pieces, why) in [
            (
                "get_weather<|channel|>analysis<|message|><parameter name=\"city\">\nParis\n\
                 </parameter>",
                HEADER_OPEN,
            ),
            ("get_weather<|channel|>comm", HEADER_OPEN),
        ] {
            let events = run(&[pieces], false);
            assert_eq!(name_of(&events), None, "{pieces:?}");
            assert!(fragments(&events).is_empty(), "{pieces:?}");
            assert!(ends(&events).is_empty(), "{pieces:?}");
            assert!(
                malformed(&events).iter().any(|(_, reason)| *reason == why),
                "{pieces:?}"
            );
            assert_eq!(bytes(&events), format!("{pieces}{END}"));
            nothing_empty(&events);
        }
    }

    #[test]
    fn text_that_is_not_a_name_or_no_name_starts_no_call() {
        for (header, why) in [
            ("get weather<|channel|>commentary<|message|>", NOT_A_NAME),
            (
                "{\"name\": \"f\"}<|channel|>commentary<|message|>",
                NOT_A_NAME,
            ),
            ("<|channel|>commentary<|message|>", EMPTY_NAME),
        ] {
            let call = format!("{header}<parameter name=\"city\">\nParis\n</parameter>");
            let events = run(&[&call], false);
            assert_eq!(name_of(&events), None, "{header:?}");
            assert!(fragments(&events).is_empty(), "{header:?}");
            assert!(ends(&events).is_empty(), "{header:?}");
            assert!(
                malformed(&events).iter().any(|(_, reason)| *reason == why),
                "{header:?}"
            );
            assert_eq!(bytes(&events), format!("{call}{END}"));
            nothing_empty(&events);
        }
    }

    #[test]
    fn text_between_tags_and_tags_out_of_place_are_reported() {
        let call = concat!(
            "get_weather<|channel|>commentary<|message|>\n junk \n",
            "<parameter name=\"days\">\n1\n</parameter>\">",
            "<|channel|>commentary<|message|><parameter name=\"\">\nx\n</parameter>"
        );
        let events = run(&[call], false);
        assert_eq!(arguments(&events), r#"{"days": 1}"#);
        let reported = malformed(&events);
        assert!(
            reported.contains(&("junk", TEXT_BETWEEN_TAGS)),
            "{reported:?}"
        );
        assert!(
            reported.contains(&("\">", TAG_OUT_OF_PLACE)),
            "{reported:?}"
        );
        assert!(
            reported.contains(&("<|channel|>commentary<|message|>", TAG_OUT_OF_PLACE)),
            "{reported:?}"
        );
        assert!(
            reported.contains(&("<parameter name=\"\">", EMPTY_KEY)),
            "{reported:?}"
        );
        assert_eq!(bytes(&events), format!("{call}{END}"));
        nothing_empty(&events);
    }

    #[test]
    fn a_tag_inside_a_key_cuts_it_short_and_is_read_between_keys() {
        let call = concat!(
            "get_weather<|channel|>commentary<|message|><parameter name=\"da",
            "<parameter name=\"days\">\n2\n</parameter>"
        );
        let events = run(&[call], false);
        assert_eq!(arguments(&events), r#"{"days": 2}"#);
        assert!(malformed(&events).contains(&("<parameter name=\"da", TAG_CUT_SHORT)));
    }

    /// The `Malformed` and `Dropped` events: what reaches the content, and what is hidden.
    fn shown_and_hidden(events: &[Event]) -> (Vec<&str>, Vec<&str>) {
        let mut shown = Vec::new();
        let mut hidden = Vec::new();
        for event in events {
            match event {
                Event::Malformed { text, .. } => shown.push(text.text.as_str()),
                Event::Dropped { text, .. } => hidden.push(text.text.as_str()),
                _ => {}
            }
        }
        (shown, hidden)
    }

    /// `call` whole and split in two at every character boundary.
    fn every_split(call: &str, cut: bool) -> Vec<Vec<Event>> {
        every_split_spelled(SPELLING, call, cut)
    }

    /// [`every_split`] in `spelling`, character by character too.
    fn every_split_spelled(spelling: Spelling, call: &str, cut: bool) -> Vec<Vec<Event>> {
        let characters: Vec<&str> = call
            .char_indices()
            .map(|(at, c)| &call[at..at + c.len_utf8()])
            .collect();
        std::iter::once(run_spelled(spelling, &[call], cut))
            .chain(
                (1..call.len())
                    .filter(|at| call.is_char_boundary(*at))
                    .map(|at| run_spelled(spelling, &[&call[..at], &call[at..]], cut)),
            )
            .chain(std::iter::once(run_spelled(spelling, &characters, cut)))
            .collect()
    }

    #[test]
    fn a_key_the_call_already_has_is_reported_and_the_first_value_kept() {
        const HEADER: &str = "get_weather<|channel|>commentary<|message|>";
        const FIRST: &str = "<parameter name=\"city\">\nParis\n</parameter>";
        // The repeated parameter last, between two others, and a value written whole repeated:
        // the call keeps the first value, so its arguments hold each key once, and the later
        // parameter is reported whole, tags and newlines included; the newline the template
        // writes before its key's tag is dropped as the template's.
        for (body, repeated, expected) in [
            (
                format!("{FIRST}\n<parameter name=\"city\">\nRome\n</parameter>"),
                "<parameter name=\"city\">\nRome\n</parameter>",
                r#"{"city": "Paris"}"#,
            ),
            (
                format!(
                    "{FIRST}\n<parameter name=\"city\">\nRome\n</parameter>\n\
                     <parameter name=\"days\">\n2\n</parameter>"
                ),
                "<parameter name=\"city\">\nRome\n</parameter>",
                r#"{"city": "Paris", "days": 2}"#,
            ),
            (
                "<parameter name=\"days\">\n2\n</parameter>\n<parameter name=\"days\">\n3\n\
                 </parameter>"
                    .to_string(),
                "<parameter name=\"days\">\n3\n</parameter>",
                r#"{"days": 2}"#,
            ),
        ] {
            let call = format!("{HEADER}{body}");
            for events in every_split(&call, false) {
                assert_eq!(arguments(&events), expected, "{body:?}");
                assert_eq!(malformed(&events), [(repeated, REPEATED_KEY)], "{body:?}");
                let (_, hidden) = shown_and_hidden(&events);
                assert_eq!(hidden, ["\n"], "{body:?}");
                assert_eq!(ends(&events), [END], "{body:?}");
                assert_eq!(bytes(&events), format!("{call}{END}"), "{body:?}");
                nothing_empty(&events);
            }
        }
        // Ended inside the repeated value, by the message's marker or by a cut: the parameter so
        // far is reported, the held start of a tag with it, and the first value stays the call's.
        for (body, cut, expected) in [
            (
                "\n<parameter name=\"city\">\nRo",
                false,
                r#"{"city": "Paris"}"#,
            ),
            (
                "\n<parameter name=\"city\">\nRome\n</param",
                false,
                r#"{"city": "Paris"}"#,
            ),
            (
                "\n<parameter name=\"city\">\nRo",
                true,
                r#"{"city": "Paris""#,
            ),
            (
                "\n<parameter name=\"city\">\nRome\n</param",
                true,
                r#"{"city": "Paris""#,
            ),
        ] {
            let call = format!("{HEADER}{FIRST}{body}");
            let reported = body.trim_start_matches('\n');
            for events in every_split(&call, cut) {
                assert_eq!(arguments(&events), expected, "{body:?} cut={cut}");
                assert_eq!(malformed(&events), [(reported, REPEATED_KEY)], "{body:?}");
                let end = if cut { "" } else { END };
                assert_eq!(ends(&events), [end], "{body:?}");
                assert_eq!(bytes(&events), format!("{call}{end}"), "{body:?}");
                nothing_empty(&events);
            }
        }
    }

    #[test]
    fn the_next_key_closes_a_value_left_unclosed_where_the_spelling_says_so() {
        const HEADER: &str = "get_weather<|channel|>commentary<|message|>";
        // A newline and the next key's tag close the value, the newline being the template's,
        // whatever the value's type; the value keeps a newline of its own before that one.
        for (body, expected) in [
            (
                "<parameter name=\"city\">\nParis\n<parameter name=\"days\">\n3\n</parameter>",
                r#"{"city": "Paris", "days": 3}"#,
            ),
            (
                "<parameter name=\"days\">\n3\n<parameter name=\"city\">\nParis\n</parameter>",
                r#"{"days": 3, "city": "Paris"}"#,
            ),
            (
                "<parameter name=\"note\">\nNone\n<parameter name=\"city\">\nRome\n</parameter>",
                r#"{"note": null, "city": "Rome"}"#,
            ),
            (
                "<parameter name=\"city\">\nline one\n\n<parameter name=\"days\">\n1\n\
                 </parameter>",
                r#"{"city": "line one\n", "days": 1}"#,
            ),
            (
                "<parameter name=\"city\">\n\n<parameter name=\"days\">\n1\n</parameter>",
                r#"{"city": "", "days": 1}"#,
            ),
            // Unclosed at the message's end: closed by the end, as before.
            (
                "<parameter name=\"city\">\nParis\n<parameter name=\"days\">\n3",
                r#"{"city": "Paris", "days": 3}"#,
            ),
        ] {
            let call = format!("{HEADER}{body}");
            for events in every_split_spelled(NEXT_KEY_CLOSES, &call, false) {
                assert_eq!(arguments(&events), expected, "{body:?}");
                assert!(malformed(&events).is_empty(), "{body:?}");
                assert_eq!(bytes(&events), format!("{call}{END}"), "{body:?}");
                nothing_empty(&events);
            }
        }
        // The key's tag without a newline of the value's before it is the value's text: right
        // after the newline the template writes after the key's tag, and after other text.
        for (body, expected) in [
            (
                "<parameter name=\"city\">\n<parameter name=\"days\">\n1\n</parameter>",
                r#"{"city": "<parameter name=\"days\">\n1"}"#,
            ),
            (
                "<parameter name=\"city\">\nParis<parameter name=\"days\">\n1\n</parameter>",
                r#"{"city": "Paris<parameter name=\"days\">\n1"}"#,
            ),
        ] {
            let call = format!("{HEADER}{body}");
            for events in every_split_spelled(NEXT_KEY_CLOSES, &call, false) {
                assert_eq!(arguments(&events), expected, "{body:?}");
                assert_eq!(bytes(&events), format!("{call}{END}"), "{body:?}");
            }
        }
        // A repeated key the next key closes is reported without the newline, which goes with the
        // next key's tag.
        let call = format!(
            "{HEADER}<parameter name=\"city\">\nParis\n</parameter>\n<parameter name=\"city\">\n\
             Rome\n<parameter name=\"days\">\n2\n</parameter>"
        );
        for events in every_split_spelled(NEXT_KEY_CLOSES, &call, false) {
            assert_eq!(arguments(&events), r#"{"city": "Paris", "days": 2}"#);
            assert_eq!(
                malformed(&events),
                [("<parameter name=\"city\">\nRome", REPEATED_KEY)]
            );
            assert_eq!(bytes(&events), format!("{call}{END}"));
            nothing_empty(&events);
        }
        // Without the spelling's say, the next key's tag is the value's text, as before.
        let call = format!(
            "{HEADER}<parameter name=\"city\">\nParis\n<parameter name=\"days\">\n3\n</parameter>"
        );
        for events in every_split(&call, false) {
            assert_eq!(
                arguments(&events),
                r#"{"city": "Paris\n<parameter name=\"days\">\n3"}"#
            );
        }
    }

    #[test]
    fn an_end_inside_a_value_keeps_every_value_that_arrived_and_shows_no_tag() {
        const HEADER: &str = "get_weather<|channel|>commentary<|message|>";
        for (body, expected) in [
            // A streamed string: what arrived is kept and closed; the held `<` may only start
            // the closing tag, which goes into the closing quote's bytes.
            ("<parameter name=\"city\">\nPar <", r#"{"city": "Par "}"#),
            // A held `"` cannot be the closing tag: it is the value's text.
            (
                "<parameter name=\"city\">\nsay \"hi\"",
                r#"{"city": "say \"hi\""}"#,
            ),
            // Values written whole are written as their closing tag would write them, before
            // and after a string, with and without the template's newline before the end.
            ("<parameter name=\"days\">\n12", r#"{"days": 12}"#),
            (
                "<parameter name=\"city\">\nParis\n</parameter>\n<parameter name=\"days\">\n2",
                r#"{"city": "Paris", "days": 2}"#,
            ),
            ("<parameter name=\"days\">\n3\n</parame", r#"{"days": 3}"#),
            ("<parameter name=\"note\">\nnull", r#"{"note": null}"#),
            (
                "<parameter name=\"o\">\n{\"a\": [1]}",
                r#"{"o": {"a": [1]}}"#,
            ),
            // A key with no value yet, and a tag the end cut short between values, are the
            // template's: the object closes over them.
            ("\n<parameter na", "{}"),
            (
                "<parameter name=\"city\">\nParis\n</parameter>\n<parameter name=\"da",
                r#"{"city": "Paris"}"#,
            ),
        ] {
            let call = format!("{HEADER}{body}");
            for events in every_split(&call, false) {
                assert_eq!(arguments(&events), expected, "{body:?}");
                let (shown, _) = shown_and_hidden(&events);
                assert!(shown.is_empty(), "{body:?}: {shown:?}");
                assert_eq!(ends(&events), [END], "{body:?}");
                assert_eq!(bytes(&events), format!("{call}{END}"), "{body:?}");
                nothing_empty(&events);
            }
        }
    }

    #[test]
    fn a_cut_stream_keeps_every_value_that_arrived_and_closes_nothing() {
        const HEADER: &str = "get_weather<|channel|>commentary<|message|>";
        for (body, expected) in [
            ("<parameter name=\"city\">\nPar", r#"{"city": "Par"#),
            ("<parameter name=\"days\">\n12", r#"{"days": 12"#),
            (
                "<parameter name=\"city\">\nParis\n</parameter>\n<parameter name=\"days\">\n1",
                r#"{"city": "Paris", "days": 1"#,
            ),
            ("<parameter name=\"days\">\n3\n</parame", r#"{"days": 3"#),
            ("<parameter name=\"note\">\nnu", r#"{"note": "nu""#),
            ("<parameter name=\"da", ""),
            ("\n<parameter na", ""),
        ] {
            let call = format!("{HEADER}{body}");
            for events in every_split(&call, true) {
                assert_eq!(arguments(&events), expected, "{body:?}");
                assert!(
                    serde_json::from_str::<serde_json::Value>(&arguments(&events)).is_err(),
                    "{body:?}: a cut call never looks complete"
                );
                let (shown, _) = shown_and_hidden(&events);
                assert!(shown.is_empty(), "{body:?}: {shown:?}");
                assert_eq!(ends(&events), [""], "{body:?}");
                assert_eq!(bytes(&events), call, "{body:?}");
                nothing_empty(&events);
            }
        }
    }

    #[test]
    fn a_cut_inside_the_header_drops_it_and_starts_no_call() {
        // The cut stopped the model inside the markers that open a call: the name so far and the
        // part of the header's close it wrote are the template's, cut short, and none of them is
        // content. A header the model stopped writing (the stop token) is reported instead.
        for header in [
            "get_wea",
            "get_weather",
            "get_weather<|chan",
            "get_weather<|channel|>commentary<|mess",
        ] {
            for events in every_split(header, true) {
                assert_eq!(name_of(&events), None, "{header:?}");
                assert!(ends(&events).is_empty(), "{header:?}");
                let (shown, hidden) = shown_and_hidden(&events);
                assert!(shown.is_empty(), "{header:?}: {shown:?}");
                assert_eq!(hidden.concat(), header, "{header:?}");
                assert_eq!(bytes(&events), header, "{header:?}");
                nothing_empty(&events);
            }
        }
        let events = run(&["get_weather<|chan"], false);
        assert_eq!(name_of(&events), None);
        assert!(malformed(&events)
            .iter()
            .any(|(text, reason)| *text == format!("get_weather<|chan{END}")
                && *reason == HEADER_OPEN));
    }

    #[test]
    fn whitespace_inside_text_out_of_place_stays_with_it() {
        const HEADER: &str = "get_weather<|channel|>commentary<|message|>";
        // Between two parameters: the newlines beside the tags are the template's, the spaces
        // between the words are the model's.
        let call = format!(
            "{HEADER}<parameter name=\"city\">\nParis\n</parameter>\nsome stray words\n\
             <parameter name=\"days\">\n2\n</parameter>"
        );
        for events in every_split(&call, false) {
            assert_eq!(arguments(&events), r#"{"city": "Paris", "days": 2}"#);
            let (shown, _) = shown_and_hidden(&events);
            assert_eq!(shown.concat(), "some stray words");
            assert_eq!(bytes(&events), format!("{call}{END}"));
            nothing_empty(&events);
        }
        // After the last value, the next message's markers and an answer, all text out of place
        // inside the call: every byte of it is shown as written.
        let tail = "<|end|>stray words<|start|>assistant<|channel|>final<|message|>Hi there.";
        let call = format!("{HEADER}<parameter name=\"city\">\nParis\n</parameter>{tail}");
        for events in every_split(&call, false) {
            assert_eq!(arguments(&events), r#"{"city": "Paris"}"#);
            let (shown, _) = shown_and_hidden(&events);
            assert_eq!(shown.concat(), tail);
            nothing_empty(&events);
        }
        // A header that names no call: the message is shown whole, the value's spaces and
        // newlines included, since no tag of it was taken.
        let call = concat!(
            "get weather<|channel|>commentary<|message|><parameter name=\"city\">\nNew York\n",
            "</parameter>"
        );
        for events in every_split(call, false) {
            assert_eq!(name_of(&events), None);
            let (shown, hidden) = shown_and_hidden(&events);
            assert_eq!(shown.concat(), format!("{call}{END}"));
            assert!(hidden.is_empty(), "{hidden:?}");
            nothing_empty(&events);
        }
    }

    #[test]
    fn a_name_is_what_the_recipient_pattern_takes() {
        for name in ["get\"x", "{f}", "a>b", "it's", "ns.tool-1", "計算"] {
            let call = format!(
                "{name}<|channel|>commentary<|message|><parameter name=\"city\">\nParis\n</parameter>"
            );
            let events = run(&[&call], false);
            assert_eq!(name_of(&events), Some(name), "{name:?}");
            assert_eq!(arguments(&events), r#"{"city": "Paris"}"#, "{name:?}");
            assert!(malformed(&events).is_empty(), "{name:?}");
        }
        for name in ["get weather", "get<x", "a\tb"] {
            let call = format!(
                "{name}<|channel|>commentary<|message|><parameter name=\"city\">\nParis\n</parameter>"
            );
            let events = run(&[&call], false);
            assert_eq!(name_of(&events), None, "{name:?}");
            assert!(
                malformed(&events)
                    .iter()
                    .any(|(_, reason)| *reason == NOT_A_NAME),
                "{name:?}"
            );
            assert_eq!(bytes(&events), format!("{call}{END}"));
        }
    }

    #[test]
    fn whitespace_around_the_name_makes_the_header_no_call() {
        // The reader's pattern takes the name right after the recipient marker and wants the
        // header's close right after the name: ` to= get_weather <|channel|>…` is no call there,
        // so it is none here either, and its bytes are shown as what the model wrote.
        for name in [
            " get_weather ",
            " get_weather",
            "get_weather ",
            "get_weather\n",
            "\tget_weather",
        ] {
            let call = format!(
                "{name}<|channel|>commentary<|message|><parameter name=\"city\">\nParis\n</parameter>"
            );
            for events in every_split(&call, false) {
                assert_eq!(name_of(&events), None, "{name:?}");
                assert!(fragments(&events).is_empty(), "{name:?}");
                assert!(
                    malformed(&events)
                        .iter()
                        .any(|(text, reason)| *reason == NOT_A_NAME
                            && text.starts_with(&format!("{name}<|channel|>"))),
                    "{name:?}"
                );
                let (shown, _) = shown_and_hidden(&events);
                assert_eq!(shown.concat(), format!("{call}{END}"), "{name:?}");
                nothing_empty(&events);
            }
        }
    }

    #[test]
    fn keys_and_values_are_escaped_as_json() {
        let call = concat!(
            "lookup<|channel|>commentary<|message|><parameter name=\"q\\\"k\">\n",
            "say \"hi\" \\ tab\tend\n</parameter>"
        );
        let events = run(&[call], false);
        let read: serde_json::Value =
            serde_json::from_str(&arguments(&events)).expect("the arguments are JSON");
        assert_eq!(read["q\\\"k"], "say \"hi\" \\ tab\tend");
    }
}
