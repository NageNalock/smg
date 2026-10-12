# smg-response-template

A Rust port of the response-template parser of Hugging Face transformers
(`transformers.utils.chat_parsing`, release 5.18.0). A checkpoint can describe
in the `response_template` of its `tokenizer_config.json` how its output splits
into message fields (`thinking`, `content`, `tool_calls`, ...). This crate loads
such a template and parses model output with it, streaming or at once, and
returns what transformers returns: the same keys in the same order, the same
strings, ints kept apart from floats, floats bit for bit, and the same region
events for the same chunks.

A template this crate would not parse exactly as transformers does is refused
when it loads (`LoadError::Unsupported`, naming the construct), never parsed
differently. Callers can then fall back to another parser.
`read_response_template` reads every template transformers loads, its patterns
as the `regex` module reads them, also one this crate does not run;
`ResponseTemplate::unsupported` then names what it does not run.

## API

| transformers 5.18.0 | this crate |
|---|---|
| `ResponseParser(template, prefix=..., tools=...)` | `ResponseParser::new(&template, prefix, &tools)` |
| `.initial_events`, `.feed(text)`, `.finalize()` | `initial_events()`, `feed(text)`, `finalize(self)` |
| `parse_response(text, template, prefix=..., tools=...)` | `parse_response(text, &template, prefix, &tools)` |
| `load_response_template(dict)` | `load_response_template(&serde_json::Value)`; `read_response_template` also returns a template this crate does not run (`ResponseTemplate::unsupported`) |
| `ResponseTemplate.truncate_past_last_anchor` | `ResponseTemplate::truncate_past_last_anchor` |
| event dicts / message dict | `Event` (serializes to the same dicts) / `Message` |
| `ValueError` while loading | `LoadError::Invalid` |
| exceptions while parsing | `ParseError` (`Content { kind, .. }` names the Python class; `Unsupported` for a template this crate does not run) |

`ResponseTemplate::last_anchor_end` gives where the last start anchor of a text
ends, or `None` without one, where `truncate_past_last_anchor` keeps the whole
text.

`prefix` is required as in transformers; pass `""` to opt out. `finalize`
consumes the parser, so the `RuntimeError`s of feeding or finalizing twice
cannot happen. Results match transformers' for a template `Value` equal to
what Python's `json.loads` returns; serde_json reads some float literals, and
integers beyond u64, differently. The files follow transformers' layout:
`response_templates.rs`, `response_parser.rs` and `content_parsers.rs` port
the files of the same names, keeping their function names and statement
order. `pyre.rs` stands in for the `regex` module and `py.rs` for the Python
built-ins the parser applies to model output (`str.strip`, `int`, `float`,
`json.loads`). As in transformers, every `feed` searches the text not yet
committed again, so a stream costs time quadratic in how much text stays
uncommitted.

The `adapter` feature adds the glue smg's reasoning and tool parsers use, which
is not part of transformers: `adapter::check` says whether they can use a
template, and an `adapter::ResponseParserState` holds the parser of one output
for both of them. They read the fields `transformers serve` reads (`thinking`,
here also `reasoning_content`, `content` and `tool_calls`): a stream from the
region chunks and closed tool-call regions after the prompt, a complete output
from the parsed message.

Where smg's use differs from `transformers serve` by design:

- smg does not use a template that holds a float, since it reads the tokenizer
  config with serde_json.
- The parsers see the output without the stop or end-of-sequence text the stop
  decoder removes; `transformers serve` also feeds that text, so whitespace
  before it, or the marker inside a region it ends, can differ.
- The prompt tail is cut from the prompt tokens decoded with special tokens
  kept, as in `transformers serve`, but the tokens are smg's own encode. They
  can differ from transformers' where transformers 5 rebuilds the tokenizer (as
  for a legacy Llama `tokenizer.json`), a processor adds a BOS token or media
  expands to a different number of tokens, and the model's input then differs
  already. smg does not apply `clean_up_tokenization_spaces`, which
  transformers applies where a tokenizer sets it and its model is not BPE.
  Without a tokenizer, or when decoding fails, the rendered prompt stands in. A
  prompt without a start anchor gives no tail: the parsers start from the
  output, where transformers parses the whole prompt, user text included, as
  the start of the message; the gateway logs it once per model and tokenizer.
  An anchor that only an earlier turn wrote still starts the tail there.
- Without tools (or with `tool_choice: none`) the tool parser does not run and
  tool calls are dropped; with `separate_reasoning: false` reasoning is
  returned as content; no prompt text is streamed, neither prompt regions nor
  text at the end of the prompt that the parser held back as the possible
  start of a delimiter (`transformers serve` streams that once the output
  shows it began none); a continued assistant message returns only the
  generated part.
- When the template reads content only after an opener, a forced tool
  choice (Chat `required`, a named function or `allowed_tools` in required
  mode, Messages `any` or `tool`) and a JSON `response_format` are refused
  with a 400: smg would constrain them to JSON from the first token, which
  opens no region, so the parsers would read none of it. A template whose
  content has no opener reads that JSON as content.
- Tool-call arguments are cast to the types the request's tools declare, as
  `parse_response(..., tools=...)` does; `transformers serve` passes no tools.
- A tool-call region whose value is a list gives one call per item
  (`transformers serve` fails there), and arguments are serialized as smg's
  other tool parsers do.
- In a stream, when the opener of a `tool_calls` region captures the name of
  its call (the transform is one dict whose `function.name` is a group of the
  open pattern, with a `function.arguments`), the tool parser sends the name
  once the region is open, and the arguments when it closes; `transformers
  serve` sends the whole call when it closes. An open pattern whose match
  reaches the end of the text so far could still grow, so the region opens, and
  the name goes out, with the next chunk. Output that follows the close in the
  same chunk waits for the next chunk, so that the arguments come first.
  Otherwise the content of a chunk is sent before the tool calls that close in
  it.
- A region that fails to parse (its content does not parse, it holds a value
  JSON cannot, or its calls cannot be read), also of a field the parsers do not
  read, is content from where it began, as generated, and the rest of the
  output passes through unparsed, where `transformers serve` fails a complete
  response and ends a stream. A complete output is still the parsed message
  wherever `transformers serve` can read that. `take_error` says why, for a
  log. A missing required field is not an error. A call whose name was sent and
  whose region then fails, for example when the output is cut short or its
  arguments do not parse, keeps its name without arguments, and the region is
  content after it. A Chat stream then finishes with `tool_calls`, or `length`
  when cut short. A Messages stream keeps a `tool_use` block with an empty
  input and stops with `tool_use`, or `max_tokens`. Responses keeps a
  `function_call` with empty arguments, `incomplete` when cut short, and does
  not run an MCP tool for it.
- A stream holds back at most 8 KiB of text that could still begin a delimiter,
  where `transformers serve` holds it all and searches it again for every
  chunk; past that, the text goes to the current region, and only a delimiter
  that starts in its last 4 KiB is waited for. A longer delimiter, such as a
  long whitespace run before an opener, can leave part of its text in the
  region.

## Regular expressions

transformers compiles patterns with the `regex` module (`regex.compile(pattern,
regex.DOTALL)`, version 0 behaviour), and streaming depends on its
`search(..., partial=True)`, whose results come from the module's matcher
rather than from the pattern's language.

`pyre/parse.rs` ports the module's parser (`_regex_core.py`) and reads a
pattern into a tree of every construct the module reads: literals and every
escape, classes with ranges, class escapes and POSIX classes, assertions,
every kind of group, look-around, backreferences, quantifiers of any item
(greedy, lazy and possessive, `{m,n}` included), inline flags (a verbose
pattern is read as verbose), fuzzy settings, group calls, conditionals and
verbs. It rejects what the module rejects. It does not read Unicode properties
in braces (`\p{...}`), named characters (`\N{...}`), POSIX classes other than
the 14 POSIX names, group names outside ASCII, a group inside a group of the
same name, version 1 behaviour (`(?V1)`), or groups nested more than 100 deep
(the module takes up to 198; a template must not be able to exhaust the
stack): a template with one of these in an open, close or start anchor pattern
is refused when it is read.

`pyre.rs` runs a tree on a backtracking matcher that reproduces the module's
results. It runs:

- literal characters and escapes of them (`\n`, `\t`, `\x41`, `\u00e9`,
  `\012`, ...), also `{`, `}` and `]` where the module reads them as
  literals;
- `.`, `\w`, `\W`, `\s`, `\S`, `\d`, `\D` (with the module's Unicode
  tables), and classes of characters, ranges of them and class escapes, read
  as the module reads them (`[\s-a]` is `\s`, `-` and `a`; `[\b]` is a
  backspace; `[[a]]` is `[` or `a`, then `]`; a class with a class escape
  and its complement is any character, also when negated: `[^\s\S]` matches
  every character);
- `^`, `\A`, `$`, `\Z`, `\z`, `\b`;
- groups, named or not, `(?:...)`, alternation, comments, and inline flags
  that change nothing a match returns (`s`, `u`, `x`, `f`, `V0`);
- `*`, `+`, `?` and `{m,n}` after one character, and `?` after a group;
- `*?`, `+?`, `??` and `{m,n}?` after one character. Open and close patterns
  are searched partially, and after a lazy repeat the module's partial
  matches follow its own rules: before a literal character it reports a
  partial match once at most one character is left, and before a literal
  string once no later copy of the string is left, whatever the repeat could
  take. The matcher reproduces them where they decide the result alone: for
  `.*?` before a literal, for `*?` after another item and before `\b` and a
  literal, and for a lazy repeat without a maximum (`*?`, `+?`, `{m,}?`) after
  another item and before a literal, outside alternatives, repeats and
  lookaheads (as in a call name read with `\S+?`);
- backreferences in start anchors and tag patterns;
- `(?=...)` without groups or backreferences in it, in start anchors and tag
  patterns (they are only used with `finditer`, never searched partially).

The matcher refuses everything else, naming it: lookahead in open and close
patterns, negative lookahead and lookbehind, inline flags that change a match
(`i`, `m`, `w`, `a`, `L`, `-s`, and the global `b`, `e`, `p`, `r`), possessive
quantifiers, repeats of groups and assertions, other lazy repeats in open and
close patterns (where the module's own first sets, or its search for a literal
that follows, decide where a partial match starts), `\B`, `\m`, `\M`, `\G`,
`\K`, `\X`, `\R`, `\h`, Unicode properties and POSIX classes, atomic groups,
branch reset groups, conditionals, group calls, fuzzy settings, verbs, and
backreferences in open and close patterns. So are alternatives two of
which can end in a negated class of one character, also before an empty group
(the module drops empty groups, merges such alternatives into one class and
matches `[^a]|[^b]` and `[^a](?:)|[^b]` as `[^ab]`), and a backreference inside
an optional group (the module does not retry the group where it failed before,
even when the group it refers to has changed since).

## Other templates refused when they load

`read_response_template` reads these too; `ResponseTemplate::unsupported`
names the reason.

- A start anchor or tag pattern that can match the empty string, or a field
  whose open and close patterns can both match it (transformers can loop
  forever there).
- `content_args` that transformers only reads while parsing and that fail
  there or depend on Python typing: not a dict, a `value_parser` that is not a
  dict naming a known parser, a `string_delims` that is not a list of pairs of
  non-empty strings, an empty or non-string `line_sep` or `kv_sep`, a
  non-string `tag_pattern`, a `key` or `value` group a match can leave unset.
- A value transformers stores in two places and then changes in place: a list
  in `defaults` for a repeated field without `join` (transformers appends to
  the template's own list, so it grows from one parse to the next), or a
  transform that uses the parsed content, or an item of it, at two paths where
  one contains the other (casting tool-call arguments changes both).

## Known differences

Both raise an explicit error where transformers returns a value:

- A value JSON cannot hold (NaN, an infinity, an integer outside i64 and u64, a
  string with a lone surrogate): `ParseError::Unrepresentable`, from the call
  whose events or message would carry it. A lone surrogate is parsed into a
  private-use character (from U+F0000 or U+100000 on), so a JSON text that
  also holds one of those characters and gives a key holding it twice raises
  it too, even where transformers then drops the value.
- JSON nested 512 deep: `ParseError::Content` with `PyErrorKind::Recursion`.
  Python raises `RecursionError` only near its C recursion limit, about 10,000
  levels.

## Parity tests

`scripts/generate_hf_fixtures.py` runs transformers itself and records what it
returns in `tests/fixtures/hf/`: every parser session of transformers' own tests
(`tests/utils/test_chat_parsing.py` at tag v5.18.0), the templates built into
`transformers serve`, the templates huggingface/trl writes for gpt-oss and
GLM-4.5 (`trl/chat_template_utils.py`), templates with the shapes of public
checkpoints written with neutral markers, prompts for these templates that end
inside a region, a delimiter or the start anchor (with and without tools), rules
of loading and parsing, seeded random templates and sessions, regex rows for
every pattern of that corpus, for Python-dialect details and for random
patterns, and the Python built-ins and character tables the port reproduces. The
tests replay each session whole, at every two-way split and in recorded
chunkings, and compare values and event traces. `tests/unsupported.tsv` lists
the recorded templates transformers accepts and this crate refuses, with the
reason; the test fails when that changes in either direction. Every template
transformers accepts is read, except those the reader does not read, and every
regex row's pattern is read exactly when the `regex` module compiles it.

The fixtures are regenerated by hand, and CI only replays them in the crate's
tests. They are pinned to transformers 5.18.0 (tag v5.18.0, commit
a906d3c4b65095f2308b6a6a193e934d03b8eb5d), `regex` 2026.9.10 and Python 3.12,
and are next regenerated when transformers 5.19 is released. To regenerate:

```bash
pip install transformers==5.18.0 regex==2026.9.10
git clone --depth 1 --branch v5.18.0 https://github.com/huggingface/transformers /tmp/transformers
python crates/response_template/scripts/generate_hf_fixtures.py --transformers-src /tmp/transformers
cargo test -p smg-response-template --features adapter
```

`--check` fails when the committed fixtures are stale.
`--extra-cases FILE --out DIR` records local cases and `--fuzz N --out DIR`
random ones; run the tests on them with `RESPONSE_TEMPLATE_HF_FIXTURES=DIR`.

To move to a newer transformers: update the pins in the generator (its
docstring and `PINNED_*`) and here, regenerate, review the fixture diff, adjust
the port, and update `TRANSFORMERS_VERSION` in `tests/hf_parity.rs`.

## License

Apache-2.0. Portions are ported from Hugging Face transformers (Apache-2.0);
the fixtures are recorded from it and include templates and inputs from its
tests.
