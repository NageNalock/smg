//! Lowering a pattern, as the reader read it, into the linear pieces a symphony table holds, and
//! splitting a delimiter's pieces into its marker and what surrounds it.
//!
//! A pattern lowers node by node: literal characters into one literal, a class or a repeat of
//! one character into a run of a class resolved with the `regex` module's own `\w`, `\s` and
//! `\d` tables, a group the transform reads into a captured run, `(?:LIT)?` into an optional
//! literal, `\b` into a boundary. An alternation lowers into alternatives, each linear, in the
//! order the `regex` module tries them. Anything else is refused by name, and the walk goes on, so
//! that one pass names every construct a pattern uses that no table reads.

use std::sync::OnceLock;

use symphony::shape::{Atom, Class, Pattern, Slot};

use super::refusal::{Construct, Refused};
use crate::{
    py,
    pyre::{self, Escape, Look, Node, Other, Quantifier, Regex, Set},
};

/// The most alternatives one pattern lowers into.
pub(super) const MAX_ALTERNATIVES: usize = 32;

/// A set of characters: sorted, disjoint ranges, or every character outside them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Chars {
    ranges: Vec<(char, char)>,
    negated: bool,
}

impl Chars {
    pub(super) fn new(mut ranges: Vec<(char, char)>, negated: bool) -> Self {
        ranges.retain(|(lo, hi)| lo <= hi);
        ranges.sort_unstable();
        let mut merged: Vec<(char, char)> = Vec::with_capacity(ranges.len());
        for (lo, hi) in ranges {
            match merged.last_mut() {
                Some((_, last)) if next(*last).is_none_or(|after| lo <= after) => {
                    *last = (*last).max(hi);
                }
                _ => merged.push((lo, hi)),
            }
        }
        Self {
            ranges: merged,
            negated,
        }
    }

    /// Every character outside the sorted, disjoint `ranges`, as ranges.
    fn complement(ranges: &[(char, char)]) -> Vec<(char, char)> {
        let mut out = Vec::new();
        let mut from = Some('\0');
        for &(lo, hi) in ranges {
            if let Some(start) = from {
                if let Some(end) = previous(lo).filter(|&end| start <= end) {
                    out.push((start, end));
                }
            }
            from = next(hi);
        }
        if let Some(start) = from {
            out.push((start, char::MAX));
        }
        out
    }

    /// The characters in the set, as ranges.
    fn members(&self) -> Vec<(char, char)> {
        if self.negated {
            Self::complement(&self.ranges)
        } else {
            self.ranges.clone()
        }
    }

    /// Whether the set holds a character, and every character it holds passes `test`.
    pub(super) fn all(&self, test: impl Fn(char) -> bool) -> bool {
        let members = self.members();
        !members.is_empty() && members.iter().all(|&(lo, hi)| (lo..=hi).all(&test))
    }

    /// Whether `c` is in the set.
    pub(super) fn contains(&self, c: char) -> bool {
        let at = self.ranges.partition_point(|&(lo, _)| lo <= c);
        let inside = at > 0 && c <= self.ranges[at - 1].1;
        inside != self.negated
    }

    /// The set as a symphony class.
    pub(super) fn class(&self) -> Class {
        Class::new(self.ranges.iter().copied(), self.negated)
    }
}

/// The character after `c`, skipping the surrogates no text holds.
fn next(c: char) -> Option<char> {
    match c {
        '\u{D7FF}' => Some('\u{E000}'),
        c => char::from_u32(u32::from(c) + 1),
    }
}

/// The character before `c`, skipping the surrogates.
fn previous(c: char) -> Option<char> {
    match c {
        '\u{E000}' => Some('\u{D7FF}'),
        c => u32::from(c).checked_sub(1).and_then(char::from_u32),
    }
}

/// The ranges of the characters that pass `test`, over every character.
fn table(test: fn(char) -> bool) -> Vec<(char, char)> {
    let mut out: Vec<(char, char)> = Vec::new();
    for c in (0..=u32::from(char::MAX)).filter_map(char::from_u32) {
        if !test(c) {
            continue;
        }
        match out.last_mut() {
            Some((_, hi)) if next(*hi) == Some(c) => *hi = c,
            _ => out.push((c, c)),
        }
    }
    out
}

/// The `regex` module's `\w`, `\s` and `\d`, as ranges, computed once.
fn tables() -> &'static [Vec<(char, char)>; 3] {
    static TABLES: OnceLock<[Vec<(char, char)>; 3]> = OnceLock::new();
    TABLES.get_or_init(|| {
        [
            table(pyre::is_word),
            table(char::is_whitespace),
            table(pyre::is_digit),
        ]
    })
}

/// The ranges a class escape holds.
fn escape_ranges(escape: Escape) -> Vec<(char, char)> {
    let [word, space, digit] = tables();
    match escape {
        Escape::Word => word.clone(),
        Escape::NotWord => Chars::complement(word),
        Escape::Space => space.clone(),
        Escape::NotSpace => Chars::complement(space),
        Escape::Digit => digit.clone(),
        Escape::NotDigit => Chars::complement(digit),
    }
}

/// The characters `set` holds, or why the lowering does not take it.
pub(super) fn chars(set: &Set) -> Result<Chars, Refused> {
    match set {
        Set::Char(c) => Ok(Chars::new(vec![(*c, *c)], false)),
        Set::Any => Ok(Chars::new(Vec::new(), true)),
        Set::Escape(escape) => Ok(Chars::new(escape_ranges(*escape), false)),
        Set::Property(name) => Err(Refused::new(
            Construct::Property,
            format!("the property or POSIX class {name}"),
        )),
        Set::Class {
            negated,
            ranges,
            escapes,
            properties,
        } => {
            if let Some(name) = properties.first() {
                return Err(Refused::new(
                    Construct::Property,
                    format!("the property or POSIX class {name} in a class"),
                ));
            }
            let mut all = ranges.to_vec();
            for escape in escapes {
                all.extend(escape_ranges(*escape));
            }
            Ok(Chars::new(all, *negated))
        }
    }
}

/// One piece of a lowered pattern: a symphony [`Atom`], its class kept as [`Chars`] for the
/// checks that read its members.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Item {
    Lit(String),
    Run {
        chars: Chars,
        min: u16,
        max: Option<u16>,
        greedy: bool,
        capture: Option<Slot>,
    },
    Boundary,
    Maybe(String),
}

impl Item {
    /// The piece as a symphony atom.
    pub(super) fn atom(&self) -> Atom {
        match self {
            Self::Lit(text) => Atom::Lit(text.as_str().into()),
            Self::Run {
                chars,
                min,
                max,
                greedy,
                capture,
            } => Atom::Run {
                class: chars.class(),
                min: *min,
                max: *max,
                greedy: *greedy,
                capture: *capture,
            },
            Self::Boundary => Atom::Boundary,
            Self::Maybe(text) => Atom::Maybe(text.as_str().into()),
        }
    }

    /// Whether this is optional whitespace, as a delimiter writes around its marker: a greedy run
    /// of a class of whitespace that may be empty, or an optional literal of whitespace. The
    /// whitespace is what Python's `str.strip` removes, so trimming the state beside the marker
    /// drops the same text the pattern takes.
    pub(super) fn is_slack(&self) -> bool {
        match self {
            Self::Run {
                chars,
                min: 0,
                greedy: true,
                capture: None,
                ..
            } => chars.all(py::is_space),
            Self::Maybe(text) => text.chars().all(py::is_space),
            _ => false,
        }
    }

    /// Whether this piece of whitespace can take `c`.
    pub(super) fn takes(&self, c: char) -> bool {
        match self {
            Self::Run { chars, .. } => chars.contains(c),
            Self::Maybe(text) => text.starts_with(c),
            Self::Lit(_) | Self::Boundary => false,
        }
    }
}

/// Pieces as a symphony pattern.
pub(super) fn pattern(items: &[Item]) -> Pattern {
    Pattern(items.iter().map(Item::atom).collect())
}

/// A pattern's alternatives, each a sequence of pieces, in the order the `regex` module tries
/// them.
pub(super) type Branches = Vec<Vec<Item>>;

/// The groups a lowering captures: each group's number and the slot its run is captured as.
#[derive(Clone, Debug, Default)]
pub(super) struct Captures(pub(super) Vec<(usize, Slot)>);

impl Captures {
    fn slot(&self, group: usize) -> Option<Slot> {
        self.0
            .iter()
            .find(|(number, _)| *number == group)
            .map(|(_, slot)| *slot)
    }
}

/// Lower `regex`, capturing the groups `captures` names. Every construct the lowering does not
/// take is in the error, in the order met.
pub(super) fn lower(regex: &Regex, captures: &Captures) -> Result<Branches, Vec<Refused>> {
    let mut refused = Vec::new();
    if !regex.flags.is_empty() {
        refused.push(Refused::new(
            Construct::InlineFlags,
            format!("inline flags (?{})", regex.flags),
        ));
    }
    let mut lowering = Lowering {
        captures,
        refused: &mut refused,
    };
    let branches = lowering.node(&regex.node, vec![Vec::new()]);
    if refused.is_empty() {
        Ok(branches)
    } else {
        Err(refused)
    }
}

/// Literal delimiters, each its own alternative, in the order given.
pub(super) fn literals(texts: &[String]) -> Branches {
    texts
        .iter()
        .map(|text| vec![Item::Lit(text.clone())])
        .collect()
}

struct Lowering<'a> {
    captures: &'a Captures,
    refused: &'a mut Vec<Refused>,
}

impl Lowering<'_> {
    fn refuse(&mut self, construct: Construct, detail: impl Into<String>) {
        self.refused.push(Refused::new(construct, detail));
    }

    /// The alternatives after `node`: each of `branches` followed by each way through `node`.
    fn node(&mut self, node: &Node, branches: Branches) -> Branches {
        match node {
            Node::One(Set::Char(c)) => append(branches, &Item::Lit(c.to_string())),
            Node::One(set) => self.run(set, (1, Some(1)), true, branches),
            Node::Look(Look::WordBoundary) => append(branches, &Item::Boundary),
            Node::Look(Look::NotWordBoundary) => {
                self.refuse(Construct::NotWordBoundary, "\\B");
                branches
            }
            Node::Look(Look::Start) => {
                self.refuse(Construct::StartOfText, "^ or \\A");
                branches
            }
            Node::Look(Look::Dollar | Look::End) => {
                self.refuse(Construct::EndOfText, "$, \\Z or \\z");
                branches
            }
            Node::Around {
                behind, negated, ..
            } => {
                let (construct, what) = match (behind, negated) {
                    (_, true) => (Construct::NegativeLookAround, "a negative look-around"),
                    (true, false) => (Construct::LookBehind, "a lookbehind"),
                    (false, false) => (Construct::LookAhead, "a lookahead"),
                };
                self.refuse(construct, what);
                branches
            }
            Node::Group(Some(group), inner) => match self.captures.slot(*group) {
                Some(slot) => self.captured(inner, slot, branches),
                None => self.node(inner, branches),
            },
            Node::Group(None, inner) => self.node(inner, branches),
            Node::Cat(nodes) => nodes
                .iter()
                .fold(branches, |branches, node| self.node(node, branches)),
            Node::Alt(alternatives) => {
                let mut ways = Vec::new();
                for alternative in alternatives {
                    ways.extend(self.node(alternative, vec![Vec::new()]));
                }
                self.then(branches, &ways, false)
            }
            Node::Repeat {
                node: inner,
                min,
                max,
                kind,
            } => self.repeat(inner, (*min, *max), *kind, branches),
            Node::Backref(group) => {
                self.refuse(
                    Construct::Backreference,
                    format!("a backreference to group {group}"),
                );
                branches
            }
            Node::Other(other, _) => {
                self.refuse(Construct::OtherRegex, other_name(*other));
                branches
            }
        }
    }

    /// A run of the characters `set` holds.
    fn run(
        &mut self,
        set: &Set,
        (min, max): (u16, Option<u16>),
        greedy: bool,
        branches: Branches,
    ) -> Branches {
        match chars(set) {
            Ok(chars) => append(
                branches,
                &Item::Run {
                    chars,
                    min,
                    max,
                    greedy,
                    capture: None,
                },
            ),
            Err(refused) => {
                self.refused.push(refused);
                branches
            }
        }
    }

    /// A group the transform reads as `slot`: one run of a class, captured.
    fn captured(&mut self, inner: &Node, slot: Slot, branches: Branches) -> Branches {
        let before = self.refused.len();
        let lowered = self.node(inner, vec![Vec::new()]);
        if self.refused.len() > before {
            return branches;
        }
        let run = match lowered.as_slice() {
            [single] => match single.as_slice() {
                [Item::Run {
                    chars,
                    min,
                    max,
                    greedy,
                    capture: None,
                }] => Some(Item::Run {
                    chars: chars.clone(),
                    min: *min,
                    max: *max,
                    greedy: *greedy,
                    capture: Some(slot),
                }),
                _ => None,
            },
            _ => None,
        };
        match run {
            Some(run) => append(branches, &run),
            None => {
                self.refuse(
                    Construct::Capture,
                    format!("the {slot:?} group is not one run of a class"),
                );
                branches
            }
        }
    }

    /// `inner{min,max}`: a run of one character's class, an optional literal, or refused.
    fn repeat(
        &mut self,
        inner: &Node,
        (min, max): (u32, Option<u32>),
        kind: Quantifier,
        branches: Branches,
    ) -> Branches {
        if kind == Quantifier::Possessive {
            self.refuse(Construct::Possessive, "a possessive repeat");
            return branches;
        }
        let greedy = kind == Quantifier::Greedy;
        if let Some(set) = one(inner) {
            if let (Set::Char(c), 0, Some(1), true) = (set, min, max, greedy) {
                return append(branches, &Item::Maybe(c.to_string()));
            }
            let (Ok(min), Ok(max)) = (u16::try_from(min), max.map(u16::try_from).transpose())
            else {
                self.refuse(Construct::LongRepeat, "a repeat bound above 65,535");
                return branches;
            };
            return self.run(set, (min, max), greedy, branches);
        }
        if (min, max, greedy) != (0, Some(1), true) {
            self.refuse(
                Construct::NestedRepeat,
                "a repeat of a group, other than an optional one",
            );
            return branches;
        }
        if let Some(text) = literal(inner).filter(|text| !text.is_empty()) {
            return append(branches, &Item::Maybe(text));
        }
        // `(?:A|B)?`: the group's alternatives, then nothing, as a greedy `?` tries them.
        let before = self.refused.len();
        let ways = self.node(inner, vec![Vec::new()]);
        if self.refused.len() > before {
            return branches;
        }
        self.then(branches, &ways, true)
    }

    /// Each of `branches` followed by each of `ways`, and then by nothing when `optional`: the
    /// order the module tries them, every way after one branch before the next branch.
    fn then(&mut self, branches: Branches, ways: &Branches, optional: bool) -> Branches {
        let mut out = Vec::new();
        for branch in branches {
            for way in ways {
                out.extend(way.iter().fold(vec![branch.clone()], append));
            }
            if optional {
                out.push(branch);
            }
            if out.len() > MAX_ALTERNATIVES {
                self.refuse(
                    Construct::Alternation,
                    format!("more than {MAX_ALTERNATIVES} alternatives"),
                );
                out.truncate(MAX_ALTERNATIVES);
                break;
            }
        }
        out
    }
}

/// The set of a node that is one character, inside groups that capture nothing.
fn one(node: &Node) -> Option<&Set> {
    match node {
        Node::One(set) => Some(set),
        Node::Group(None, inner) => one(inner),
        Node::Cat(nodes) if nodes.len() == 1 => nodes.first().and_then(one),
        _ => None,
    }
}

/// The text of a node that is literal characters only, inside groups that capture nothing.
fn literal(node: &Node) -> Option<String> {
    match node {
        Node::One(Set::Char(c)) => Some(c.to_string()),
        Node::Group(None, inner) => literal(inner),
        Node::Cat(nodes) => nodes.iter().map(literal).collect(),
        _ => None,
    }
}

/// Each branch followed by `item`, a literal merged into a literal before it.
fn append(mut branches: Branches, item: &Item) -> Branches {
    for branch in &mut branches {
        match (branch.last_mut(), item) {
            (Some(Item::Lit(before)), Item::Lit(text)) => before.push_str(text),
            _ => branch.push(item.clone()),
        }
    }
    branches
}

fn other_name(other: Other) -> &'static str {
    match other {
        Other::Atomic => "an atomic group",
        Other::BranchReset => "a branch reset group",
        Other::Conditional => "a conditional",
        Other::Fuzzy => "fuzzy matching",
        Other::GroupCall => "a group call",
        Other::Keep => "the escape \\K",
        Other::SearchAnchor => "the escape \\G",
        Other::Grapheme => "the escape \\X",
        Other::LineBreak => "the escape \\R",
        Other::WordStart => "the escape \\m",
        Other::WordEnd => "the escape \\M",
        Other::Fail => "the verb (*FAIL)",
        Other::Prune => "the verb (*PRUNE)",
        Other::Skip => "the verb (*SKIP)",
    }
}

/// An open's alternative, split: the whitespace before its marker, the marker, and what follows
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Open {
    pub(super) slack: Vec<Item>,
    pub(super) marker: String,
    pub(super) header: Vec<Item>,
}

/// A close's alternative, split: what comes before its marker, the marker, and the whitespace
/// after it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Close {
    pub(super) trailer: Vec<Item>,
    pub(super) marker: String,
    pub(super) slack: Vec<Item>,
}

/// Split an open's alternative at its marker: its first literal after leading whitespace.
pub(super) fn split_open(branch: &[Item]) -> Result<Open, Refused> {
    let lead = branch.iter().take_while(|item| item.is_slack()).count();
    match branch.get(lead) {
        Some(Item::Lit(marker)) => Ok(Open {
            slack: branch[..lead].to_vec(),
            marker: marker.clone(),
            header: branch[lead + 1..].to_vec(),
        }),
        _ => Err(Refused::new(
            Construct::NoLiteral,
            "an open whose first piece after whitespace is not literal text",
        )),
    }
}

/// Split a close's alternative at its marker: its last literal before trailing whitespace.
pub(super) fn split_close(branch: &[Item]) -> Result<Close, Refused> {
    let tail = branch
        .iter()
        .rev()
        .take_while(|item| item.is_slack())
        .count();
    let at = branch.len() - tail;
    match at.checked_sub(1).map(|last| (last, branch.get(last))) {
        Some((last, Some(Item::Lit(marker)))) => Ok(Close {
            trailer: branch[..last].to_vec(),
            marker: marker.clone(),
            slack: branch[at..].to_vec(),
        }),
        _ => Err(Refused::new(
            Construct::NoLiteral,
            "a close whose last piece before whitespace is not literal text",
        )),
    }
}

/// The literals a start anchor's alternatives spell, in the order the `regex` module tries them;
/// an optional literal spells two, the one with it first, as a greedy `?` tries.
pub(super) fn anchor_literals(branches: &Branches) -> Result<Vec<String>, Refused> {
    let mut out: Vec<String> = Vec::new();
    for branch in branches {
        let mut spelled = vec![String::new()];
        for item in branch {
            match item {
                Item::Lit(text) => {
                    for prefix in &mut spelled {
                        prefix.push_str(text);
                    }
                }
                Item::Maybe(text) => {
                    spelled = spelled
                        .into_iter()
                        .flat_map(|prefix| [format!("{prefix}{text}"), prefix])
                        .collect();
                }
                Item::Run { .. } | Item::Boundary => {
                    return Err(Refused::new(
                        Construct::AnchorNotFinite,
                        "a start anchor pattern that is not literal text",
                    ))
                }
            }
            if out.len() + spelled.len() > MAX_ALTERNATIVES {
                return Err(Refused::new(
                    Construct::AnchorNotFinite,
                    format!("a start anchor pattern of more than {MAX_ALTERNATIVES} literals"),
                ));
            }
        }
        for text in spelled {
            if text.is_empty() {
                return Err(Refused::new(
                    Construct::AnchorNotFinite,
                    "a start anchor pattern that matches empty text",
                ));
            }
            if !out.contains(&text) {
                out.push(text);
            }
        }
    }
    Ok(out)
}
