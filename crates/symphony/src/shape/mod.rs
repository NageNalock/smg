//! Call shapes as data: the patterns a template writes around a call's parts, and a matcher that
//! runs them while the output streams.
//!
//! A table under [`formats`] spells a family's markers as terminals and leaves what lies between
//! them to the assembler its call syntax names. A model whose template publishes its own reader, a
//! Hugging Face response template, says more of the call, as regular expressions: what follows the
//! call's marker and names the tool (`<function=(?P<name>[^>\n]+)>`), a parameter's tag with its
//! key and its value (`<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>`), what comes
//! before the call's closing marker (`</function>\s*`). This module holds such expressions as data
//! and runs them over text as it arrives, so that a call syntax can be given as data rather than
//! written as one more assembler.
//!
//! - A [`Pattern`] is linear: [`Atom`]s matched one after another, each a literal, a run of one
//!   [`Class`] of characters (greedy or lazy, bounded or not, captured as the tool's name, a key or
//!   a value when its [`Slot`] says so), `\b`, or an optional literal. The patterns public
//!   response templates write inside a call lower to one; a compiler refuses the rest by name.
//! - An [`End`] is one way a value ends: text the match takes, then text it only looks ahead at.
//! - A [`Class`] is a set of characters as ranges. Whoever lowers a template resolves `\w`, `\s`
//!   and `\d` into ranges with the template's own dialect, so the crate holds no character
//!   tables.
//! - A [`Matcher`] runs a pattern, with its ends, over text as it arrives. Its answer is the one
//!   Python's `regex` module gives for the whole text (`match` under `DOTALL`, the dialect response
//!   templates are written in), and it gives it as soon as no continuation of the text can change
//!   it.
//! - [`Pattern::check`] refuses a pattern that a stream could decide only at the end of the
//!   output.
//! - An [`Interner`] keeps one copy of each distinct value for the life of the process, so a table
//!   built at run time can hold its shapes as `&'static` data and stay `Copy`.
//!
//! [`formats`]: crate::formats

mod class;
mod intern;
mod matcher;

pub use class::Class;
pub use intern::Interner;
pub use matcher::{Match, Matcher, Verdict, HOLD_CAP};

/// A linear pattern: its atoms, matched one after another from the start of the text it is run
/// over.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Pattern(pub Box<[Atom]>);

/// One step of a [`Pattern`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    /// These characters, in order.
    Lit(Box<str>),
    /// A run of characters of `class`: `min` at least, `max` at most (no bound when `None`), as
    /// many as the rest of the pattern allows when `greedy` and as few when not, captured as
    /// `capture` says: `\w+`, `[^"]+`, `\S+?`, `\s*`, `\n?`, `.*?`, `[^>]*?`.
    Run {
        class: Class,
        min: u16,
        max: Option<u16>,
        greedy: bool,
        capture: Option<Slot>,
    },
    /// `\b`: a word character on one side and none on the other, by the word class the pattern is
    /// run with.
    /// Before the text's start and after its end there is no word character.
    Boundary,
    /// `(?:LIT)?`: these characters, or nothing when the rest of the pattern does not match after
    /// them.
    Maybe(Box<str>),
}

/// What a captured run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// The tool's name: `(?P<name>…)`.
    Name,
    /// A parameter's key: `(?P<key>…)`.
    Key,
    /// A parameter's value: `(?P<value>…)`.
    Value,
}

/// One way a value ends, as a parameter's tag spells it: `consume` is part of the match
/// (`\s*</parameter>`), and `ahead` must follow it but is not (a value that also ends where the
/// next parameter's tag begins, `(?=<parameter=)`). Either may be empty.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct End {
    pub consume: Pattern,
    pub ahead: Pattern,
}

/// Why a stream cannot run a pattern. An atom is named by its place in its pattern.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PatternError {
    /// A literal with no characters, which the lowering should have left out.
    #[error("atom {atom}: a literal with no characters")]
    EmptyLiteral { atom: usize },
    /// A run whose least number of characters is above its most.
    #[error("atom {atom}: a run of at least {min} and at most {max} characters")]
    Bounds { atom: usize, min: u16, max: u16 },
    /// A greedy run of every character with no bound. No text after it ends the run, so the
    /// stream would hold everything that follows until the output ends.
    #[error("atom {atom}: a greedy run of any character with no bound")]
    Undecidable { atom: usize },
    /// A second run captured as a slot one run already captures.
    #[error("atom {atom}: a second run captured as {slot:?}")]
    CapturedTwice { atom: usize, slot: Slot },
    /// An end with nothing to match and nothing to look ahead at: every value would end where it
    /// starts.
    #[error("an end with nothing to match or to look ahead at")]
    EmptyEnd,
}

impl Pattern {
    /// Whether a stream can run the pattern: every literal has characters, every run's bounds are
    /// in order, no slot is captured twice, and every run decides before the output ends. A run
    /// that cannot is a greedy one of every character with no bound (`.*`, `[\s\S]*`): it takes
    /// all the text there is and gives it back only when nothing comes after it. A greedy run of a
    /// narrower class ends at the first character outside it (`\S+` at a space), and a lazy run
    /// ends at the first place the rest of the pattern matches, so neither needs the end of the
    /// output to decide.
    pub fn check(&self) -> Result<(), PatternError> {
        let mut captured: Vec<Slot> = Vec::new();
        for (atom, step) in self.0.iter().enumerate() {
            match step {
                Atom::Lit(text) | Atom::Maybe(text) if text.is_empty() => {
                    return Err(PatternError::EmptyLiteral { atom });
                }
                Atom::Run {
                    min,
                    max: Some(max),
                    ..
                } if min > max => {
                    return Err(PatternError::Bounds {
                        atom,
                        min: *min,
                        max: *max,
                    });
                }
                Atom::Run {
                    class,
                    max: None,
                    greedy: true,
                    ..
                } if class.is_any() => return Err(PatternError::Undecidable { atom }),
                _ => {}
            }
            if let Atom::Run {
                capture: Some(slot),
                ..
            } = step
            {
                if captured.contains(slot) {
                    return Err(PatternError::CapturedTwice { atom, slot: *slot });
                }
                captured.push(*slot);
            }
        }
        Ok(())
    }
}

impl End {
    /// Whether a stream can run this end: one of its parts holds something, and each part passes
    /// [`Pattern::check`].
    pub fn check(&self) -> Result<(), PatternError> {
        if self.consume.0.is_empty() && self.ahead.0.is_empty() {
            return Err(PatternError::EmptyEnd);
        }
        self.consume.check()?;
        self.ahead.check()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(class: Class, min: u16, max: Option<u16>, greedy: bool) -> Atom {
        Atom::Run {
            class,
            min,
            max,
            greedy,
            capture: None,
        }
    }

    fn captured(atom: Atom, slot: Slot) -> Atom {
        match atom {
            Atom::Run {
                class,
                min,
                max,
                greedy,
                ..
            } => Atom::Run {
                class,
                min,
                max,
                greedy,
                capture: Some(slot),
            },
            other => other,
        }
    }

    fn pattern(atoms: impl IntoIterator<Item = Atom>) -> Pattern {
        Pattern(atoms.into_iter().collect())
    }

    #[test]
    fn a_greedy_run_of_any_character_with_no_bound_is_refused() {
        let any = Class::any();
        let word = Class::new([('a', 'z')], false);
        assert_eq!(
            pattern([Atom::Lit("<a>".into()), run(any.clone(), 0, None, true)]).check(),
            Err(PatternError::Undecidable { atom: 1 })
        );
        let spelled = Class::new([('\0', '\u{D7FF}'), ('\u{E000}', char::MAX)], false);
        assert_eq!(
            pattern([run(spelled, 1, None, true)]).check(),
            Err(PatternError::Undecidable { atom: 0 }),
            "`[\\s\\S]+` is the same run"
        );
        for decidable in [
            run(any.clone(), 0, None, false),
            run(any.clone(), 0, Some(64), true),
            run(Class::new([('>', '>')], true), 0, None, true),
            run(word, 1, None, true),
        ] {
            assert_eq!(
                pattern([decidable.clone(), Atom::Lit(">".into())]).check(),
                Ok(()),
                "{decidable:?}"
            );
        }
    }

    #[test]
    fn literals_bounds_and_captures_are_checked_by_place() {
        let word = Class::new([('a', 'z')], false);
        assert_eq!(
            pattern([Atom::Boundary, Atom::Maybe("".into())]).check(),
            Err(PatternError::EmptyLiteral { atom: 1 })
        );
        assert_eq!(
            pattern([run(word.clone(), 3, Some(2), true)]).check(),
            Err(PatternError::Bounds {
                atom: 0,
                min: 3,
                max: 2
            })
        );
        assert_eq!(
            pattern([
                captured(run(word.clone(), 1, None, true), Slot::Name),
                Atom::Lit("=".into()),
                captured(run(word.clone(), 1, None, true), Slot::Name),
            ])
            .check(),
            Err(PatternError::CapturedTwice {
                atom: 2,
                slot: Slot::Name
            })
        );
        assert_eq!(
            pattern([
                captured(run(word.clone(), 1, None, true), Slot::Key),
                Atom::Lit(">".into()),
                captured(run(Class::any(), 0, None, false), Slot::Value),
            ])
            .check(),
            Ok(())
        );
    }

    #[test]
    fn an_end_needs_something_to_match_or_to_look_ahead_at() {
        assert_eq!(End::default().check(), Err(PatternError::EmptyEnd));
        let close = pattern([Atom::Lit("</a>".into())]);
        assert_eq!(
            End {
                consume: close.clone(),
                ahead: Pattern::default()
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            End {
                consume: Pattern::default(),
                ahead: close
            }
            .check(),
            Ok(())
        );
        assert_eq!(
            End {
                consume: pattern([run(Class::any(), 0, None, true)]),
                ahead: Pattern::default()
            }
            .check(),
            Err(PatternError::Undecidable { atom: 0 })
        );
    }
}
