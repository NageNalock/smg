//! Tables spelled from public response templates, read as Hugging Face's parser reads the same
//! outputs. A response template is the `response_template` a checkpoint's `tokenizer_config.json`
//! declares, or one of transformers' own; each table here spells one with the rows a template
//! needs: a framing state for the text outside every field, turn anchors, a start row, trimmed
//! and joined stays, and the stop tokens that are terminals. The calls are spelled where a call
//! syntax the crate has reads the template's (Gemma's, the JSON object, a call given as data whose
//! header names the tool), and left out where none does yet.
//!
//! Each case is a prompt, an output and the finish, with the message Hugging Face's parser gives
//! for them: `parse_response(output, template, prefix=prompt)` from transformers 5.18.0, the
//! output without the stop token's text, which the stop decoder hides (where the stop text would
//! change the message, the case says so). Every case replays whole, at every two-way split,
//! character by character and on thirty seeded plans, and the message the chat adapter folds from
//! the events is the recorded one at every cut: the reasoning, the content, and each call's name
//! and arguments as JSON values. The cases where the table reads an output on purpose otherwise
//! are listed with what it gives and why.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one replays after a prompt only"
)]
mod common;

use std::{fs, path::PathBuf};

use common::{bytes_of, chunkings, replay_after};
use serde_json::Value;
use symphony::{
    adapt,
    shape::{self, Arguments, Atom, CallShape, Class, Dialect, NameSource, Pattern, Slot},
    CallSyntax, Declared, Emits, Engine, EngineFinish, Format, Parser,
};

/// One output and the message it reads as.
struct Case {
    prompt: &'static str,
    output: &'static str,
    finish: EngineFinish,
    reasoning: Option<&'static str>,
    content: Option<&'static str>,
    /// Each call's name and its arguments as JSON.
    calls: &'static [(&'static str, &'static str)],
}

/// Python `regex`'s `\w`, as `tests/fixtures/shape/classes.json` records it: the class a
/// template's `\w` holds, which whoever lowers a template resolves into ranges.
#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a fixture that cannot be read ends the test with its path"
)]
fn python_word() -> Class {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shape/classes.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let tables: Value = serde_json::from_str(&text).expect("classes.json");
    let ranges = tables["word"]
        .as_array()
        .expect("the word table")
        .iter()
        .map(|range| {
            let mut chars = range.as_str().expect("a range").chars();
            match (chars.next(), chars.next(), chars.next()) {
                (Some(first), Some(last), None) => (first, last),
                _ => panic!("a range is two characters: {range}"),
            }
        });
    Class::new(ranges, false)
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`gpt_oss_template`): Harmony's
/// messages, the analysis channel a thought and the final channel the answer, each closed by
/// `<|end|>`, nothing kept outside them, the turn after `<|start|>assistant`. A call's open,
/// `<\|channel\|>commentary to=functions\.(?P<name>\w+).*?<\|message\|>`, is the terminal
/// `<|channel|>commentary to=functions.` and a header that names the tool; its arguments are one
/// JSON value; its close, `<|call|>`, is a stop token, so the stream's stop closes it. The stop
/// tokens are `<|return|>`, `<|endoftext|>` and `<|call|>` (the `generation_config.json` of
/// `openai/gpt-oss-20b`).
fn gpt_oss() -> Format {
    let word = python_word();
    let shape = shape::intern(CallShape {
        header: Pattern(
            [
                Atom::Run {
                    class: word.clone(),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Run {
                    class: Class::any(),
                    min: 0,
                    max: None,
                    greedy: false,
                    capture: None,
                },
                Atom::Lit("<|message|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word,
        hold_cap: 8192,
    });
    Format::new("gpt-oss test template")
        .terminal("analysis", "<|channel|>analysis<|message|>")
        .terminal("final", "<|channel|>final<|message|>")
        .terminal("end", "<|end|>")
        .terminal("call_open", "<|channel|>commentary to=functions.")
        .terminal("call_close", "<|call|>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "analysis", "reasoning")
        .transition("outside", "final", "content")
        .transition("reasoning", "end", "outside")
        .transition("content", "end", "outside")
        .transition("outside", "call_open", "call")
        .transition("call", "call_close", "outside")
        .calls(CallSyntax::Shaped(shape))
        .framing("outside")
        .closes_at_stop("call")
        .turn_anchor("<|start|>assistant")
        .trims("reasoning")
        .trims("content")
        .stop_tokens(["<|return|>", "<|endoftext|>", "<|call|>"])
}

/// The response template of `google/gemma-4-31B-it` (its `tokenizer_config.json`), as
/// transformers v5.18.0's tests write it (`gemma4_template`): a turn after `<|turn>model\n` or
/// after `<tool_response|>`, a thought, the calls in Gemma's spelling, and the content outside
/// them, stripped. The content closes at `<turn|>`, `<|tool_response>` or `<eos>`, and all three
/// are the stop tokens (its `generation_config.json`): the first and the last have no row, and
/// `<|tool_response>`, which a stream that does not stop on it carries, is a row back into the
/// content that drops the marker.
fn gemma4() -> Format {
    Format::new("gemma 4 template")
        .terminal("think_open", "<|channel>thought\n")
        .terminal("think_close", "<channel|>")
        .terminal("call_open", "<|tool_call>")
        .terminal("call_close", "<tool_call|>")
        .terminal("response_open", "<|tool_response>")
        .state("content", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("call", Emits::Arguments)
        .transition("content", "think_open", "reasoning")
        .transition("reasoning", "think_close", "content")
        .transition("content", "call_open", "call")
        .transition("call", "call_close", "content")
        .transition("content", "response_open", "content")
        .calls(CallSyntax::Gemma)
        .turn_anchor("<|turn>model\n")
        .turn_anchor("<tool_response|>")
        .trims("content")
        .trims("reasoning")
        .stop_tokens(["<eos>", "<turn|>", "<|tool_response>"])
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`ernie_template`): the turn after
/// `Assistant:`, the thought opened by `(?:^|<think>\s*)`, so a turn that starts with text starts
/// in it, the answer between `<response>\n` and `\n?</response>`, a call between `<tool_call>` and
/// `</tool_call>` as one JSON object, and nothing kept outside them. `</s>` is the stop token.
fn ernie() -> Format {
    Format::new("ernie test template")
        .terminal("think_open", "<think>")
        .terminal("think_close", "</think>")
        .terminal("answer_open", "<response>\n")
        .terminal("answer_close", "</response>")
        .terminal("call_open", "<tool_call>")
        .terminal("call_close", "</tool_call>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("outside", "think_open", "reasoning")
        .transition("reasoning", "think_close", "outside")
        .transition("outside", "answer_open", "content")
        .transition("content", "answer_close", "outside")
        .transition("outside", "call_open", "call")
        .transition("call", "call_close", "outside")
        .calls(CallSyntax::Json)
        .framing("outside")
        .turn_anchor("Assistant:")
        .starts_in("reasoning")
        .trims("reasoning")
        .trims("content")
        .stop_tokens(["</s>"])
}

/// transformers v5.18.0's `tests/utils/test_chat_parsing.py` (`inkling_template`): the turn after
/// `<|message_model|>`, each message a thought, an answer or a call after its own marker (an
/// author's name and other text before the marker are outside), all of them closed by
/// `<|end_message|>`, and nothing kept outside them. The thoughts and the answers repeat, joined
/// with `""` and not stripped: no join row and no trim row. `<|end_message|>` ends a message
/// inside the turn; the stop token, `<|content_model_end_sampling|>` (the `config.json` of
/// `thinkingmachines/Inkling`), ends the turn and is not a terminal here.
fn inkling() -> Format {
    Format::new("inkling test template")
        .terminal("thought", "<|content_thinking|>")
        .terminal("answer", "<|content_text|>")
        .terminal("call", "<|content_invoke_tool_json|>")
        .terminal("end", "<|end_message|>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("calls", Emits::Arguments)
        .transition("outside", "thought", "reasoning")
        .transition("outside", "answer", "content")
        .transition("outside", "call", "calls")
        .transition("reasoning", "end", "outside")
        .transition("content", "end", "outside")
        .transition("calls", "end", "outside")
        .calls(CallSyntax::Json)
        .framing("outside")
        .turn_anchor("<|message_model|>")
        .stop_tokens(["<|content_model_end_sampling|>"])
}

/// The response template of `meta-models/Muse-Glimmer-30B` (its `tokenizer_config.json`): the turn
/// after `<|start|>assistant`, a thought addressed to the model itself, an answer addressed to the
/// user, both stripped, and nothing kept outside them. Its calls take a header with attributes,
/// which no call syntax here reads yet, so they are left out. `<|end_of_text|>` and `<|eot|>` are
/// the stop tokens (its `generation_config.json`); `<|eom|>` ends a message inside the turn, and
/// the model goes on after it.
fn muse() -> Format {
    Format::new("muse glimmer template")
        .terminal("thought", "to=self<|message|>")
        .terminal("answer", "to=user<|message|>")
        .terminal("eot", "<|eot|>")
        .terminal("eom", "<|eom|>")
        .state("outside", Emits::Content)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .transition("outside", "thought", "reasoning")
        .transition("outside", "answer", "content")
        .transition("reasoning", "eom", "outside")
        .transition("content", "eot", "outside")
        .transition("content", "eom", "outside")
        .framing("outside")
        .turn_anchor("<|start|>assistant")
        .trims("reasoning")
        .trims("content")
        .stop_tokens(["<|end_of_text|>", "<|eot|>"])
}

/// The gpt-oss test template's cases. With the stop token's text fed, as transformers feeds it,
/// the answers of the first two would keep `<|return|>` (and the second its edge before it): the
/// stop decoder hides it here. The calls end at `<|call|>`, which the stream's stop stands for:
/// after a thought, right after the marker, with words before `<|message|>` that the header
/// skips, after an answer with spaces around the arguments, and a marker that names no tool
/// (its header needs a word right after it), which opens no call.
const GPT_OSS: &[Case] = &[
    Case {
        prompt: "<|start|>user<|message|>Weather in Paris?<|end|><|start|>assistant",
        output: "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
                 {\"city\": \"Paris\"}",
        finish: EngineFinish::Stop,
        reasoning: Some("Need the weather."),
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\"}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Paris\", \
                 \"days\": 3}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{\"city\": \"Paris\", \"days\": 3}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.get_weather json<|message|>{}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: None,
        calls: &[("get_weather", "{}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>final<|message|>Checking.<|end|><|start|>assistant<|channel|>\
                 commentary to=functions.lookup <|constrain|>json<|message|> {\"q\": [1, 2.5, \
                 null]} ",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Checking."),
        calls: &[("lookup", "{\"q\": [1, 2.5, null]}")],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>commentary to=functions.<|channel|>final<|message|>Hi",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>user<|message|>Weather?<|end|><|start|>assistant",
        output: "<|channel|>analysis<|message|>User asks for the weather.<|end|><|start|>\
                 assistant<|channel|>final<|message|>It is sunny.",
        finish: EngineFinish::Stop,
        reasoning: Some("User asks for the weather."),
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>analysis<|message|>\n  Plan.  \n<|end|><|start|>assistant<|channel|>\
                 final<|message|> Hi! \n",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan."),
        content: Some("Hi!"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "Stray text<|channel|>final<|message|>Hi<|end|>more stray",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: "<|channel|>analysis<|message|>Plan.<|end|><|start|>assistant<|chan",
        finish: EngineFinish::Length,
        reasoning: Some("Plan."),
        content: None,
        calls: &[],
    },
    Case {
        prompt: "hello ",
        output: "<|channel|>final<|message|>Hi",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant<|channel|>final<|message|>  ",
        output: " more.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("more."),
        calls: &[],
    },
];

/// The Gemma 4 template's cases: a thought and a call, edges, a turn after a tool's response and
/// after a later turn, an answer before a call, a call written inside a thought (the thought's
/// text, as the template reads it), a thought the prompt opened, and Python's whitespace.
const GEMMA4: &[Case] = &[
    Case {
        prompt: "<|turn>user\nWeather?<turn|>\n<|turn>model\n",
        output: "<|channel>thought\nThe user wants the weather.\n<channel|><|tool_call>\
                 call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
        finish: EngineFinish::Stop,
        reasoning: Some("The user wants the weather."),
        content: None,
        calls: &[("get_weather", "{\"city\":\"Paris\"}")],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "\n\nIt is sunny.\n\n",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>\
                 <|tool_response>response:get_weather{temp:20}<tool_response|>",
        output: "It is 20 degrees.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("It is 20 degrees."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                 response:f{}<tool_response|>Done.<turn|>\n<|turn>user\nThanks<turn|>\n<|turn>\
                 model\n",
        output: "Hello.",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hello."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "Let me check.\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,\
                 days:3}<tool_call|>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Let me check."),
        calls: &[("get_weather", "{\"city\":\"Paris\",\"days\":3}")],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "<|channel>thought\nPlan <|tool_call>call:f{}<tool_call|> more<channel|>Done",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan <|tool_call>call:f{}<tool_call|> more"),
        content: Some("Done"),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n<|tool_call>call:f{}<tool_call|><|tool_response>\
                 response:f{}<tool_response|><|channel>thought\n",
        output: "Plan.\n<channel|>Answer.",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan."),
        content: Some("Answer."),
        calls: &[],
    },
    Case {
        prompt: "<|turn>model\n",
        output: "\u{a0}\u{1c}Hi\u{2003}there\u{2003}\u{1f}",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi\u{2003}there"),
        calls: &[],
    },
];

/// The ERNIE test template's cases: transformers' tests' outputs without their stop text (a
/// thought then a call, a thought then an answer, `<think>` at the turn's first byte), an answer
/// or a close at the first byte, a prompt that writes the first byte, and a thought after an
/// answer.
const ERNIE: &[Case] = &[
    Case {
        prompt: "User: Weather?\nAssistant:",
        output: "The user is asking about the weather in Paris today. Let me check the available \
                 tools.\n</think>\n\n<tool_call>\n{\"name\": \"get_current_temperature\", \
                 \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n",
        finish: EngineFinish::Stop,
        reasoning: Some(
            "The user is asking about the weather in Paris today. Let me check the \
             available tools.",
        ),
        content: None,
        calls: &[("get_current_temperature", "{\"location\":\"Paris\"}")],
    },
    Case {
        prompt: "User: Hi! How are you?\nAssistant:",
        output:
            "The user just greeted me.\n\nFirst, I'll say hello back.\n</think>\n\n<response>\n\
             Hello! I'm doing well, thank you for asking.\n</response>\n",
        finish: EngineFinish::Stop,
        reasoning: Some("The user just greeted me.\n\nFirst, I'll say hello back."),
        content: Some("Hello! I'm doing well, thank you for asking."),
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "<think>some deliberation here</think>\n\n<tool_call>\n{\"name\": \
                 \"get_current_temperature\", \"arguments\": {\"location\": \"Paris\"}}\n\
                 </tool_call>\n",
        finish: EngineFinish::Stop,
        reasoning: Some("<think>some deliberation here"),
        content: None,
        calls: &[("get_current_temperature", "{\"location\":\"Paris\"}")],
    },
    Case {
        prompt: "Assistant:",
        output: "<response>\nHello!\n</response>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hello!"),
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "</think>\n<response>\nHi\n</response>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "Assistant: ",
        output: "<response>\nHi\n</response>",
        finish: EngineFinish::Stop,
        reasoning: Some("<response>\nHi\n</response>"),
        content: None,
        calls: &[],
    },
    Case {
        prompt: "Assistant:",
        output: "<response>\nHi\n</response><think>More.</think>",
        finish: EngineFinish::Stop,
        reasoning: Some("More."),
        content: Some("Hi"),
        calls: &[],
    },
];

/// The Inkling test template's cases: transformers' multi-message test without the stop token's
/// text, answers with spaces at their edges, thoughts joined with an answer after them, and text
/// outside the messages.
const INKLING: &[Case] = &[
    Case {
        prompt: "<|message_system|><|content_text|>Be brief.<|end_message|><|message_model|>",
        output: "<|content_thinking|>Consider the weather.<|end_message|><|message_model|>\
                 <|content_thinking|> Tokyo, probably.<|end_message|><|message_model|>\
                 <|content_text|>Checking the weather now.<|end_message|><|message_model|>\
                 get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\
                 \"args\":{\"city\":\"Tokyo\",\"units\":\"C\"}}<|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: Some("Consider the weather. Tokyo, probably."),
        content: Some("Checking the weather now."),
        calls: &[("get_weather", "{\"city\":\"Tokyo\",\"units\":\"C\"}")],
    },
    Case {
        prompt: "<|message_model|>",
        output: "<|content_text|> A <|end_message|><|message_model|><|content_text|> B \
                 <|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some(" A  B "),
        calls: &[],
    },
    Case {
        prompt: "<|message_model|>",
        output: "<|content_thinking|>one <|end_message|><|message_model|><|content_thinking|>\
                 two<|end_message|><|message_model|><|content_text|>\nHi\n<|end_message|>",
        finish: EngineFinish::Stop,
        reasoning: Some("one two"),
        content: Some("\nHi\n"),
        calls: &[],
    },
    Case {
        prompt: "<|message_model|>",
        output: "stray <|message_model|>author<|content_text|>Hi<|end_message|> tail",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
];

/// The Muse Glimmer template's cases: a thought then an answer, text outside the messages, and a
/// thought alone.
const MUSE: &[Case] = &[
    Case {
        prompt: "<|start|>user<|message|>Weather?<|eot|><|start|>assistant",
        output: "to=self<|message|>Plan the answer.<|eom|><|start|>assistant to=user<|message|>It \
                 is sunny.",
        finish: EngineFinish::Stop,
        reasoning: Some("Plan the answer."),
        content: Some("It is sunny."),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: " junk to=user<|message|>\nHi\n",
        finish: EngineFinish::Stop,
        reasoning: None,
        content: Some("Hi"),
        calls: &[],
    },
    Case {
        prompt: "<|start|>assistant",
        output: " to=self<|message|> think <|eom|>",
        finish: EngineFinish::Stop,
        reasoning: Some("think"),
        content: None,
        calls: &[],
    },
];

/// Replays `case` through `format` at every chunking, and checks that the message is the
/// recorded one at each and that the events account for the output's bytes.
#[expect(
    clippy::panic,
    clippy::expect_used,
    reason = "a check that fails fails the test that runs it"
)]
fn check(name: &str, format: &Format, case: &Case) {
    for (plan, cuts) in chunkings(case.output, usize::MAX) {
        let mut parser = Engine::new(format.clone(), Declared::default());
        let events = replay_after(
            &mut parser as &mut dyn Parser,
            case.prompt,
            case.output,
            &cuts,
            &case.finish,
            false,
        )
        .unwrap_or_else(|e| panic!("{name}: {:?}: {e}", case.output));
        let place = || format!("{name}: {:?} after {:?}, {plan}", case.output, case.prompt);
        let conserved: String = events.iter().map(bytes_of).collect();
        assert_eq!(conserved, case.output, "{}", place());
        let message = adapt::chat::message(0, &events).message;
        assert_eq!(
            message.reasoning_content.as_deref(),
            case.reasoning,
            "{}",
            place()
        );
        assert_eq!(message.content.as_deref(), case.content, "{}", place());
        let calls: Vec<(String, Value)> = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| {
                let arguments = call.function.arguments.unwrap_or_default();
                let value = serde_json::from_str(&arguments)
                    .unwrap_or_else(|e| panic!("{}: arguments {arguments:?}: {e}", place()));
                (call.function.name, value)
            })
            .collect();
        let expected: Vec<(String, Value)> = case
            .calls
            .iter()
            .map(|(name, arguments)| {
                let value = serde_json::from_str(arguments).expect("recorded arguments");
                (name.to_string(), value)
            })
            .collect();
        assert_eq!(calls, expected, "{}", place());
    }
}

#[test]
fn gpt_oss_test_template_reads_as_hugging_face_reads_it() {
    let format = gpt_oss();
    for case in GPT_OSS {
        check("gpt-oss", &format, case);
    }
}

#[test]
fn gemma4_template_reads_as_hugging_face_reads_it() {
    let format = gemma4();
    for case in GEMMA4 {
        check("gemma 4", &format, case);
    }
}

#[test]
fn ernie_test_template_reads_as_hugging_face_reads_it() {
    let format = ernie();
    for case in ERNIE {
        check("ernie", &format, case);
    }
}

#[test]
fn inkling_test_template_reads_as_hugging_face_reads_it() {
    let format = inkling();
    for case in INKLING {
        check("inkling", &format, case);
    }
}

#[test]
fn muse_glimmer_template_reads_as_hugging_face_reads_it() {
    let format = muse();
    for case in MUSE {
        check("muse", &format, case);
    }
}

#[test]
fn a_field_written_twice_reads_as_both_stays() {
    // The content's own close between two stays, in a stream that does not stop on it: Hugging
    // Face's parser keeps the last stay ("there"); a stream has sent the first by then, so the
    // table reads both, each trimmed.
    check(
        "gemma 4",
        &gemma4(),
        &Case {
            prompt: "<|turn>model\n",
            output: "Hi <|tool_response> there",
            finish: EngineFinish::Stop,
            reasoning: None,
            content: Some("Hithere"),
            calls: &[],
        },
    );
}

#[test]
fn every_table_here_is_valid_and_names_its_stop_tokens() {
    for (format, stops) in [
        (gpt_oss(), &[][..]),
        (gemma4(), &["<|tool_response>"][..]),
        (ernie(), &[][..]),
        (inkling(), &[][..]),
        (muse(), &["<|eot|>"][..]),
    ] {
        assert_eq!(format.validate(), Ok(()), "{}", format.name());
        for stop in stops {
            assert!(format.is_stop_token(stop), "{}: {stop}", format.name());
        }
    }
    // A stop token that is not a terminal is not kept, and a terminal that ends a message inside
    // the turn is not a stop token.
    assert!(gpt_oss().is_stop_token("<|call|>"));
    assert!(!gpt_oss().is_stop_token("<|return|>"));
    assert!(!gemma4().is_stop_token("<turn|>"));
    assert!(!inkling().is_stop_token("<|end_message|>"));
    assert!(!muse().is_stop_token("<|eom|>"));
}
