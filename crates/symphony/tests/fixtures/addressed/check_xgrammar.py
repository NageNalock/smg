"""Check the addressed-call tags in addressed.json against xgrammar.

The Rust tests (`tests/addressed_grammar.rs`) hold the other half of the round
trip: the tables build exactly the tags in addressed.json, and read each render
there back into its message. This script checks, on CPU, with a tokenizer that
holds the fixture's markers as tokens (any Hugging Face tokenizer, the markers
added to it as tokens that decode as text):

- each tag compiles with xgrammar, accepts the renders it lists, and rejects
  the others it lists and the near misses below; a single-call tag also
  rejects a second call, of the same tool or another;
- a JSON string, in a value or in the final JSON, takes no `<|end|>`,
  `<|call|>`, `<|return|>` or `</parameter>`, as one token or spelled with
  ordinary ones, and takes other text with `<` and `|`;
- xgrammar reads the markers as text, and its stop set is the end-of-sequence
  token alone, so the `<|call|>` token stays a reachable literal of the tags;
  spelled with ordinary tokens, `<|call|>` does not end a call.

Usage: python check_xgrammar.py --tokenizer <a tokenizer directory>
"""

import argparse
import json
import sys
from pathlib import Path

FIXTURE = Path(__file__).with_name("addressed.json")
MARK = "\x00"
PARAMETER = '<parameter name="{}">\n{}\n</parameter>'


def load(path, tokens):
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_tokens([AddedToken(t, special=False, normalized=False) for t in tokens])
    return tokenizer


def near_misses(renders):
    """Outputs each forced-call tag must reject, by what is wrong with them."""
    single = renders["single"]
    city = PARAMETER.format("city", "Paris")
    days = PARAMETER.format("days", "3")
    return {
        "a value of the wrong type": single.replace("\n3\n", "\nthree\n"),
        "a missing required parameter": single.replace("\n" + days, ""),
        "parameters out of schema order": single.replace(f"{city}\n{days}", f"{days}\n{city}"),
        "an enum value not listed": single.replace(
            "</parameter><|call|>",
            "</parameter>\n" + PARAMETER.format("unit", "k") + "<|call|>",
        ),
        "an undeclared parameter": single.replace(
            "</parameter><|call|>",
            "</parameter>\n" + PARAMETER.format("extra", "x") + "<|call|>",
        ),
        "no thought": single.replace("<|channel|>analysis<|message|>Need the weather.<|end|>", ""),
        "a name no tool has": single.replace("to=get_weather", "to=ns.get_weather"),
        "the call header without <|constrain|>xml": single.replace(" <|constrain|>xml", ""),
        "</parameter> inside a text value": single.replace("Paris", "Pa</parameter>ris"),
        "<|end|> inside a text value": single.replace("Paris", "Pa<|end|>ris"),
        "<|call|> inside the thought": single.replace("Need the", "Need<|call|>the"),
        "<|start|> inside the thought": single.replace("Need the", "Need<|start|>the"),
        "a last call closed by <|end|>": single.replace("<|call|>", "<|end|>"),
    }


def json_strings(renders, eos):
    """JSON strings in renders, by the render they are in and where: the texts
    around a delimiter in the string, which the render's tags must reject, and
    one with near misses, which they take."""
    bad = ["<|end|>", "<|call|>", eos, "</parameter>"]
    good = "</div> a<b |x <|en <|call| <|retur </parameter"
    cases = [
        ("optional_only", '{"lang": "en"}', "a value", lambda t: f'{{"lang": "e{t}n"}}'),
        (
            "final_json",
            '{"answer": "yes", "score": 2}',
            "the answer",
            lambda t: f'{{"answer": "y{t}es", "score": 2}}',
        ),
        ("final_object", '{"a": [1, 2]}', "any JSON", lambda t: f'{{"a": [1, "x{t}"]}}'),
        ("final_object", '{"a": [1, 2]}', "a key", lambda t: f'{{"a": [1, 2], "k{t}": {{}}}}'),
    ]
    out = []
    for name, original, where, value in cases:
        text = renders[name].removesuffix(eos)
        assert original in text, name
        around = text.replace(original, value(MARK)).split(MARK)
        out.append((name, where, bad, around, text.replace(original, value(good))))
    return out


class Checker:
    def __init__(self, tokenizer, tokens, eos):
        import xgrammar as xgr

        self.xgr = xgr
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.eos = tokenizer.convert_tokens_to_ids(eos)
        self.info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=len(tokenizer), stop_token_ids=[self.eos]
        )
        self.compiler = xgr.GrammarCompiler(tokenizer_info=self.info)
        self.failures = []

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def spellings(self, before, word, after):
        """`word` between two texts: as the tokenizer writes it, and, for a
        marker token, spelled with ordinary tokens."""
        out = {"": self.ids(before + word + after)}
        if word in self.tokens:
            ids = self.ids(before) + self.ids("<|") + self.ids(word[2:]) + self.ids(after)
            out[" spelled"] = ids
        return out

    def accepts(self, grammar, ids):
        """Whether the tag takes `ids` and then the EOS token."""
        matcher = self.xgr.GrammarMatcher(grammar)
        return all(matcher.accept_token(i) for i in ids) and matcher.accept_token(self.eos)

    def expect(self, ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            self.failures.append(what)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True, help="a tokenizer directory")
    args = parser.parse_args()

    fixture = json.loads(FIXTURE.read_text())
    tokens, eos = fixture["tokens"], fixture["eos"]
    tokenizer = load(args.tokenizer, tokens)
    check = Checker(tokenizer, tokens, eos)
    check.expect(
        list(check.info.stop_token_ids) == [check.eos],
        f"xgrammar stops on EOS alone: {list(check.info.stop_token_ids)}",
    )
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    special = set(check.info.special_token_ids)
    check.expect(
        [check.info.decoded_vocab[i] for i in ids] == [t.encode() for t in tokens]
        and not special.intersection(ids),
        "xgrammar reads the marker tokens as text",
    )
    for token, i in zip(tokens, ids):
        check.expect(check.ids(token) == [i], f"{token} is one token")
    renders = {case["name"]: case["text"] for case in fixture["renders"]}

    misses = near_misses(renders)
    strings = json_strings(renders, eos)
    for case in fixture["tags"]:
        grammar = check.compiler.compile_structural_tag(json.dumps(case["tag"]))
        for name in case["accepts"]:
            ids = check.ids(renders[name].removesuffix(eos))
            check.expect(check.accepts(grammar, ids), f"{case['name']} accepts {name}")
        for name in case["rejects"]:
            ids = check.ids(renders[name].removesuffix(eos))
            check.expect(not check.accepts(grammar, ids), f"{case['name']} rejects {name}")
        for name, where, bad, (before, after), good in strings:
            if name not in case["accepts"]:
                continue
            for delimiter in bad:
                for how, ids in check.spellings(before, delimiter, after).items():
                    ok = not check.accepts(grammar, ids)
                    what = f"{delimiter}{how} in a string of {where}"
                    check.expect(ok, f"{case['name']} rejects {what}")
            ok = check.accepts(grammar, check.ids(good))
            check.expect(ok, f"{case['name']} takes near misses in a string of {where}")
        if case["kind"] == "single_call":
            single = renders["single"]
            call = single[single.index("<|start|>") :]
            twice = single.removesuffix("<|call|>") + "<|end|>" + call
            ok = not check.accepts(grammar, check.ids(twice))
            check.expect(ok, f"{case['name']} rejects the same call twice")
        if case["kind"] in ("tool", "single_call"):
            for why, text in misses.items():
                ok = not check.accepts(grammar, check.ids(text))
                check.expect(ok, f"{case['name']} rejects {why}")
            # Spelled with ordinary tokens, `<|call|>` is text the engine
            # does not stop on.
            ids = check.ids(renders["single"])
            assert ids[-1] == check.tokenizer.convert_tokens_to_ids("<|call|>")
            spelled = ids[:-1] + check.ids("<|") + check.ids("call|>")
            ok = not check.accepts(grammar, spelled)
            check.expect(ok, f"{case['name']} rejects <|call|> in ordinary tokens")
            matcher = check.xgr.GrammarMatcher(grammar)
            assert all(matcher.accept_token(i) for i in ids)
            bitmask = check.xgr.allocate_token_bitmask(1, check.info.vocab_size)
            matcher.fill_next_token_bitmask(bitmask)
            row = bitmask[0].tolist()
            allowed = [i for i in range(check.info.vocab_size) if row[i // 32] >> (i % 32) & 1]
            check.expect(allowed == [check.eos], f"{case['name']} leaves only EOS after <|call|>")
        else:
            final = renders["final_json"].removesuffix(eos)
            for why, text in {
                "no thought": final.replace(
                    "<|channel|>analysis<|message|>Answer as JSON.<|end|>", ""
                ),
                "<|constrain|>json on the final channel": final.replace(
                    "final<|message|>", "final<|constrain|>json<|message|>"
                ),
                "text after the JSON": final + " Done.",
                "<|end|> closing the final channel": final + "<|end|>",
            }.items():
                ok = not check.accepts(grammar, check.ids(text))
                check.expect(ok, f"{case['name']} rejects {why}")

    print(f"{len(check.failures)} failures")
    return 1 if check.failures else 0


if __name__ == "__main__":
    sys.exit(main())
