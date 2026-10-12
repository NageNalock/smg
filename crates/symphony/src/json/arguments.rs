//! A call's arguments written as one JSON value, the call named elsewhere.
//!
//! Some formats write a call's name outside its arguments: in a header before them
//! (`<|channel|>commentary to=functions.get_weather <|message|>{"city": "Paris"}`), in a template's
//! text, or nowhere at all, as when a grammar forced one function and the whole output is its
//! arguments. What is left is one JSON value, and [`Arguments`] turns its bytes into the call's
//! argument events as they arrive: the whitespace before the value is dropped as such, the value's
//! bytes are fragments for as long as they keep it a JSON prefix (judged by [`Prefix`], so the
//! fragments keep the promise `ToolCallArguments` makes, exactly as strong as for the
//! [`Assembler`](super::Assembler)), and from the first byte that does not, the bytes come back as
//! `Malformed` with `InvalidArguments`. The fragments are the model's own bytes, spacing included.
//! A word being spelled (`nu`) is held until it is a literal or is refused, which the judge does
//! from its first letter, so that where the chunks were cut does not decide which of its bytes
//! were sent.
//!
//! The value ends where it is whole: at its closing bracket or quote, or before the byte after a
//! number or a literal; a number or a literal that the text ends on is whole once the caller says
//! no more bytes come. `feed` returns how many of a piece's bytes the value took, all of them
//! before its end and up to it in the piece that carries it, so the caller routes what follows (a
//! trailer, a closing marker, text that belongs to no value) itself. The call's start and end are
//! the caller's too, since it knows the name and where the call closes: a forced function starts
//! its call at the value's first byte, a format at its header.

use crate::{
    event::{DropReason, Event, Events, MalformedReason, Text},
    json::{is_complete, Prefix},
};

/// The whitespace JSON allows before a value.
fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// The argument events of one call, from the bytes of its arguments value.
#[derive(Clone, Debug)]
pub struct Arguments {
    index: u32,
    /// The value's bytes so far, from its first one.
    value: String,
    /// Whether the value's first byte has arrived.
    started: bool,
    /// Bytes of `value` accounted for, as fragments or as malformed text.
    emitted: usize,
    /// Whether the bytes are still a JSON prefix, judged as they arrive.
    prefix: Prefix,
    /// Where the value stopped being a JSON prefix, if it did.
    invalid_from: Option<usize>,
    /// Where the value ended, once it is whole.
    whole: Option<usize>,
}

impl Arguments {
    /// A reader for the arguments of the call at `index`.
    pub fn new(index: u32) -> Self {
        Self {
            index,
            value: String::new(),
            started: false,
            emitted: 0,
            prefix: Prefix::default(),
            invalid_from: None,
            whole: None,
        }
    }

    /// Whether the value's first byte has arrived.
    pub fn started(&self) -> bool {
        self.started
    }

    /// Whether the value is whole.
    pub fn whole(&self) -> bool {
        self.whole.is_some()
    }

    /// Append the next bytes, push the events they complete, and return how many of them the
    /// value took: all of them before its end, up to its end in the piece that carries it, and
    /// none once it is whole.
    pub fn feed(&mut self, bytes: &str, out: &mut Events) -> usize {
        if self.whole.is_some() {
            return 0;
        }
        let mut taken = 0;
        let mut rest = bytes;
        if !self.started {
            let value = rest.trim_start_matches(json_space);
            let space = &rest[..rest.len() - value.len()];
            if !space.is_empty() {
                out.push(Event::Dropped {
                    text: Text::uncounted(space),
                    why: DropReason::Whitespace,
                });
            }
            taken = space.len();
            if value.is_empty() {
                return taken;
            }
            self.started = true;
            rest = value;
        }
        let before = self.value.len();
        self.value.push_str(rest);
        let refused = self.prefix.feed(rest);
        if let Some(end) = self.prefix.ended() {
            if self.invalid_from.is_none() && is_complete(&self.value[..end]) {
                self.whole = Some(end);
            }
        }
        let end = self.whole.unwrap_or(self.value.len());
        if self.invalid_from.is_none() {
            // A refusal at or after the value's end is the bytes that follow it, not the value's.
            if let Some(from) = refused.filter(|&from| from < end) {
                self.invalid_from = Some(from.max(self.emitted));
            }
        }
        let held = match (self.whole, self.invalid_from) {
            (None, None) => self.prefix.word_start(),
            _ => None,
        };
        self.emit(
            held.map_or(end, |word| word.clamp(self.emitted, end)),
            end,
            out,
        );
        self.value.truncate(end);
        taken + end - before
    }

    /// No more bytes come: a number or a literal the bytes end on is whole now, and a word they
    /// end on is sent as the prefix it is. Returns whether the value is whole.
    pub fn finish(&mut self, out: &mut Events) -> bool {
        if self.whole.is_none()
            && self.started
            && self.invalid_from.is_none()
            && is_complete(&self.value)
        {
            self.whole = Some(self.value.len());
        }
        let end = self.value.len();
        self.emit(end, end, out);
        self.whole.is_some()
    }

    /// The value's bytes up to `upto` not yet accounted for, the rest up to `end` held: a fragment
    /// for the part that keeps the value a JSON prefix, malformed text for the part after it.
    fn emit(&mut self, upto: usize, end: usize, out: &mut Events) {
        let valid_end = self.invalid_from.unwrap_or(upto).min(upto);
        if self.emitted < valid_end {
            let fresh = &self.value[self.emitted..valid_end];
            out.push(Event::ToolCallArguments {
                index: self.index,
                json: fresh.to_string(),
                source: Text::uncounted(fresh),
            });
            self.emitted = valid_end;
        }
        if self.invalid_from.is_some() && self.emitted < end {
            out.push(Event::Malformed {
                text: Text::uncounted(&self.value[self.emitted..end]),
                why: MalformedReason::InvalidArguments,
            });
            self.emitted = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The events for `pieces`, what the value took of each, and whether it is whole after the
    /// last piece and after `finish`.
    fn run(pieces: &[&str]) -> (Vec<Event>, Vec<usize>, bool, bool) {
        let mut arguments = Arguments::new(0);
        let mut out = Events::new();
        let taken = pieces
            .iter()
            .map(|piece| arguments.feed(piece, &mut out))
            .collect();
        let fed = arguments.whole();
        let finished = arguments.finish(&mut out);
        (out.drain(), taken, fed, finished)
    }

    fn fragments(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::ToolCallArguments { json, .. } => Some(json.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The bytes the events account for.
    fn sources(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::ToolCallArguments { source, .. } => source.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                _ => "",
            })
            .collect()
    }

    #[test]
    fn an_object_streams_as_its_bytes_and_ends_at_its_brace() {
        let (events, taken, whole, _) = run(&[" \n{\"city\": ", "\"Paris\"}", "\n</x>"]);
        assert_eq!(
            events[0],
            Event::Dropped {
                text: Text::uncounted(" \n"),
                why: DropReason::Whitespace,
            }
        );
        assert_eq!(fragments(&events), "{\"city\": \"Paris\"}");
        assert_eq!(taken, [11, 8, 0], "nothing after the brace is the value's");
        assert!(whole);
        assert_eq!(sources(&events), " \n{\"city\": \"Paris\"}");
    }

    #[test]
    fn every_cut_takes_the_same_bytes() {
        let text = " {\"a\": [1, 2.5, \"x\\\"y\"], \"b\": null}  tail";
        let value_end = text.find("  tail").expect("the value's end");
        for cut in (1..text.len()).filter(|&cut| text.is_char_boundary(cut)) {
            let (events, taken, whole, _) = run(&[&text[..cut], &text[cut..]]);
            assert!(whole, "cut at {cut}");
            assert_eq!(taken.iter().sum::<usize>(), value_end, "cut at {cut}");
            assert_eq!(
                fragments(&events),
                text[..value_end].trim_start(),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_number_is_whole_at_the_byte_after_it_or_at_the_end() {
        let (events, taken, whole, finished) = run(&["12", "5 }"]);
        assert_eq!(fragments(&events), "125");
        assert_eq!(taken, [2, 1]);
        assert!(whole && finished);
        let (_, _, whole, finished) = run(&["12"]);
        assert!(!whole, "more digits may come");
        assert!(finished, "the end of the bytes ends the number");
        let (_, _, _, finished) = run(&["1."]);
        assert!(!finished, "a number cut short is not whole");
    }

    #[test]
    fn bytes_that_break_the_prefix_come_back_malformed() {
        let (events, taken, whole, finished) = run(&["{\"a\": trux, \"b\": 1}"]);
        assert_eq!(fragments(&events), "{\"a\": ");
        assert_eq!(
            events.last(),
            Some(&Event::Malformed {
                text: Text::uncounted("trux, \"b\": 1}"),
                why: MalformedReason::InvalidArguments,
            }),
            "the word is refused from its first letter"
        );
        assert_eq!(taken, [19], "a broken value takes every byte");
        assert!(!whole && !finished);
        let (events, _, _, _) = run(&["{\"a\": t", "rux, \"b\": 1}"]);
        assert_eq!(
            fragments(&events),
            "{\"a\": ",
            "a word is held until it is decided, so the cut changes nothing"
        );
        assert_eq!(sources(&events), "{\"a\": trux, \"b\": 1}");
        let (events, _, _, finished) = run(&["[nu", "ll, tr"]);
        assert_eq!(
            fragments(&events),
            "[null, tr",
            "a literal is sent once it ends, and a word the bytes end on as the prefix it is"
        );
        assert!(!finished);
        let (events, _, whole, _) = run(&["[1}", " more"]);
        assert_eq!(fragments(&events), "[1");
        assert!(!whole, "an array a brace cut short is not whole");
        assert_eq!(sources(&events), "[1} more");
    }

    #[test]
    fn whitespace_alone_is_no_value() {
        let (events, taken, whole, finished) = run(&[" \t", "\r\n"]);
        assert_eq!(taken, [2, 2]);
        assert!(!whole && !finished);
        assert!(fragments(&events).is_empty());
        assert_eq!(sources(&events), " \t\r\n");
    }
}
