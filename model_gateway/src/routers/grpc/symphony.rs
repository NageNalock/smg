//! Symphony tables a parser setting can select.
//!
//! A tool or reasoning parser name of the form `symphony:<table>`, from a model card (worker label
//! or `WorkerSpec`) or from `--tool-call-parser` / `--reasoning-parser`, selects the row of that
//! name in [`TABLES`]. The card comes first: a card that names either parser decides alone, and the
//! flags select a table only for a model whose card names neither. Model-id detection never yields
//! such a name, so a model selects a table only when its configuration says so. A row also names
//! the parsers the gateway takes for the table's family without symphony, and the resolver answers
//! with them wherever a parser is looked up by name: preparation (tool constraints,
//! `skip_special_tokens`, the stop decoder) and the reading of the output are what they are for the
//! family today.

/// The parser-name prefix that selects a symphony table.
pub(crate) const PREFIX: &str = "symphony:";

/// A table a parser setting can select.
#[derive(Debug)]
pub(crate) struct Table {
    /// The name after [`PREFIX`].
    pub name: &'static str,
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
        tool_parser: Some("deepseek_v41"),
        reasoning_parser: Some("deepseek_v41"),
    },
    Table {
        name: "hy4",
        tool_parser: Some("hy_v4"),
        reasoning_parser: Some("hy_v4"),
    },
    Table {
        name: "qwen3",
        tool_parser: Some("qwen"),
        reasoning_parser: Some("qwen3"),
    },
];

/// The table a parser name selects, if it names one.
pub(crate) fn table(name: &str) -> Option<&'static Table> {
    let name = name.strip_prefix(PREFIX)?;
    TABLES.iter().find(|table| table.name == name)
}

/// Whether a parser name selects a table, for the checks that refuse unknown parser names.
pub(crate) fn known(name: &str) -> bool {
    table(name).is_some()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::{
        common::{Tool, ToolChoice},
        model_card::ModelCard,
    };
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::json;
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{
        routers::grpc::utils::ParserResolver,
        worker::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
    };

    fn tool(name: &str) -> Tool {
        serde_json::from_value(json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}
            }}}
        }))
        .expect("tool")
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

    /// The check every row passes, a new one included: its name is unique, and the parsers it
    /// names for its family are registered.
    #[test]
    fn every_table_names_registered_parsers() {
        let tools = ToolParserFactory::new();
        let reasoning = ReasoningParserFactory::new();
        for (position, row) in TABLES.iter().enumerate() {
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
}
