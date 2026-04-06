"""CAD/ESAD deliberate practice drill generation.

For each (headword, sense) in the generation plan, assembles a prompt
with contrastive sense scaffold + attestations + format anchors, calls
the LLM to generate target + contrastive sentence pairs.

Usage:
    python sdg/generate_cad_drills.py
    python sdg/generate_cad_drills.py --config sdg/conf/conf_cad_drill.yaml
"""

import argparse
import json
import os
import random
import time
import uuid
from pathlib import Path

import kagglehub
import openai
import pandas as pd
from jinja2 import Template
from omegaconf import OmegaConf
from tqdm.auto import tqdm

SDG_DIR = Path(__file__).parent


# ANSI colors
class C:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Use this to reason through how to generate diverse, unambiguous OA sentences for the target sense. Plan which attestations to expand, which diversity axes to cover, and how to make contrastive examples maximally confusable.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your reasoning about generation strategy"},
            },
            "required": ["thought"],
        },
    },
}

LOOKUP_FORM_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_form",
        "description": "Find real parallel examples containing a specific OA word form. Use this to verify that a form you want to use actually appears in real texts, and to see how it's translated in context.",
        "parameters": {
            "type": "object",
            "properties": {
                "form": {"type": "string", "description": "The OA word form to search for"},
                "n": {"type": "integer", "description": "Number of examples (default: 3)", "default": 3},
            },
            "required": ["form"],
        },
    },
}

LOOKUP_GRAMMAR_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_grammar",
        "description": "Look up Old Assyrian grammar rules by category. Returns full rule text with transformation guidance and real OA examples. Categories: verb_paradigm, verb_weak, verb_derived, clause_conditional, clause_relative, clause_temporal, clause_speech, modal, negation, noun_case, noun_plural, number, pronoun, preposition, particle, ventive, word_order, nonverbal, name_pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Grammar category to look up",
                },
            },
            "required": ["category"],
        },
    },
}

ALL_TOOLS = [THINK_TOOL, LOOKUP_FORM_TOOL, LOOKUP_GRAMMAR_TOOL]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_all_data(cfg):
    """Load all required datasets."""
    print(f"{C.CYAN}Loading datasets...{C.RESET}")

    senses_df = pd.read_parquet(cfg.paths.senses)
    examples_df = pd.read_parquet(cfg.paths.examples)
    plan_df = pd.read_parquet(cfg.paths.plan)
    print(f"{C.DIM}  Senses: {len(senses_df):,} rows, {senses_df['headword'].nunique():,} headwords{C.RESET}")
    print(f"{C.DIM}  Examples: {len(examples_df):,} headwords{C.RESET}")
    print(f"{C.DIM}  Plan: {len(plan_df):,} (headword, sense) batches → {plan_df['n_to_generate'].sum():,} target examples{C.RESET}")

    input_dir = Path(kagglehub.dataset_download(cfg.datasets.dpc))
    ono_df = pd.read_parquet(input_dir / "ono.parquet")
    train_df = pd.read_parquet(input_dir / "train.parquet")
    train_df = train_df[train_df.language == "en"].reset_index(drop=True)
    print(f"{C.DIM}  Onomasticon: {len(ono_df):,} name pairs{C.RESET}")
    print(f"{C.DIM}  Training data: {len(train_df):,} English sentences (for lookup_form){C.RESET}")

    # Grammar rule index (category → rules)
    grammar_index_path = Path("data/supl_data/grammar_rule_index.json")
    grammar_rules_path = Path("data/supl_data/grammar_rules_by_section.json")
    grammar_index = json.loads(grammar_index_path.read_text()) if grammar_index_path.exists() else {}
    grammar_rules = json.loads(grammar_rules_path.read_text()) if grammar_rules_path.exists() else {}
    print(f"{C.DIM}  Grammar: {len(grammar_index)} categories, {len(grammar_rules)} rules{C.RESET}")

    return {
        "senses_df": senses_df,
        "examples_df": examples_df,
        "plan_df": plan_df,
        "ono_df": ono_df,
        "train_df": train_df,
        "grammar_index": grammar_index,
        "grammar_rules": grammar_rules,
    }


def load_templates(cfg):
    system_tmpl = Template((SDG_DIR / "prompts" / f"{cfg.prompts.system}.j2").read_text())
    user_tmpl = Template((SDG_DIR / "prompts" / f"{cfg.prompts.user}.j2").read_text())
    return system_tmpl, user_tmpl


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------


def lookup_form(form: str, train_df: pd.DataFrame, n: int = 3) -> str:
    """Find parallel examples containing a specific word form."""
    examples = []
    for _, row in train_df.sample(frac=1.0, random_state=None).iterrows():
        if f"{form.lower()} " in f"{row['transliteration'].lower()} ":
            examples.append(row)
            if len(examples) == n:
                break

    if not examples:
        return f"No examples found containing '{form}'."

    md = f"Examples containing '{form}':\n\n"
    for i, row in enumerate(examples):
        md += f"Example {i + 1}:\n"
        md += f"  Transliteration: {row['transliteration']}\n"
        md += f"  Translation: {row['translation']}\n\n"
    return md


def lookup_grammar(category: str, grammar_index: dict, grammar_rules: dict, max_rules: int = 5) -> str:
    """Look up grammar rules by category. Returns sampled rules with examples."""
    if category not in grammar_index:
        available = ", ".join(sorted(grammar_index.keys()))
        return f"Unknown category '{category}'. Available: {available}"

    entries = grammar_index[category]
    seen = set()
    deduped = []
    for e in entries:
        if e["section"] not in seen:
            seen.add(e["section"])
            deduped.append(e)

    sampled = random.sample(deduped, min(max_rules, len(deduped)))

    md = f"## Grammar: {category} ({len(deduped)} rules, showing {len(sampled)})\n\n"
    for entry in sampled:
        rule = grammar_rules.get(entry["section"], {})
        md += f"### {entry['title']}\n"
        md += f"{rule.get('rule', entry['summary'])[:500]}\n"
        examples = rule.get("examples", [])[:3]
        if examples:
            md += "Examples:\n"
            for ex in examples:
                md += f"  {ex['transliteration']} → {ex['translation']} [{ex.get('grammar', '')}]\n"
        md += "\n"
    return md


def execute_tool(tool_call, context: dict) -> str:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "think":
        return "Thought recorded."
    elif name == "lookup_form":
        return lookup_form(args["form"], context["train_df"], args.get("n", 3))
    elif name == "lookup_grammar":
        return lookup_grammar(args["category"], context["grammar_index"], context["grammar_rules"])
    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Prompt Assembly
# ---------------------------------------------------------------------------


def get_contrastive_senses(all_senses: list[dict], target_sense_id: str, max_k: int) -> list[dict]:
    """Pick top-k most confusable alternative senses (by attestation count)."""
    alternatives = [s for s in all_senses if s["sense_id"] != target_sense_id]
    alternatives.sort(key=lambda s: -s["n_att"])
    return alternatives[:max_k]


def get_raw_text_excerpt(raw_text: str, max_chars: int) -> str:
    if not raw_text or raw_text == "N/A":
        return ""
    if len(raw_text) <= max_chars:
        return raw_text
    return raw_text[:max_chars] + "\n[... truncated]"


def assemble_user_prompt(
    headword: str,
    sense_id: str,
    data: dict,
    cfg,
    user_tmpl: Template,
) -> str:
    """Assemble the user prompt for one (headword, sense_id) generation unit."""
    senses_df = data["senses_df"]
    examples_df = data["examples_df"]
    plan_df = data["plan_df"]
    ono_df = data["ono_df"]
    gen_cfg = cfg.generation

    hw_rows = senses_df[senses_df["headword"] == headword]
    first = hw_rows.iloc[0]

    all_senses = json.loads(first["all_senses_summary"])
    target_row = hw_rows[hw_rows["sense_id"] == sense_id].iloc[0]
    target_attestations = json.loads(target_row["attestations"])

    plan_row = plan_df[(plan_df["headword"] == headword) & (plan_df["sense_id"] == sense_id)]
    n_to_generate = int(plan_row.iloc[0]["n_to_generate"]) if len(plan_row) > 0 else 16

    contrastive_senses = []
    if len(all_senses) > 1:
        contrastive_senses = get_contrastive_senses(all_senses, sense_id, gen_cfg.max_contrastive_senses)

    ex_row = examples_df[examples_df["headword"] == headword]
    training_examples = json.loads(ex_row.iloc[0]["examples"]) if len(ex_row) > 0 else []

    names = ono_df.sample(min(gen_cfg.n_names, len(ono_df))).to_dict("records")

    return user_tmpl.render(
        headword=headword,
        pos=first["pos"] or "",
        grammatical_info=first["grammatical_info"] or "",
        entry_overview=first["entry_overview"] or "",
        n_senses=len(all_senses),
        all_senses=all_senses,
        target_sense_id=sense_id,
        target_gloss=target_row["gloss"],
        target_domain=target_row["domain"],
        target_attestations=target_attestations,
        n_to_generate=n_to_generate,
        contrastive_senses=contrastive_senses,
        training_examples=training_examples,
        names=names,
        raw_entry_text_excerpt=get_raw_text_excerpt(first["raw_entry_text"] or "", gen_cfg.raw_text_max_chars),
    )


# ---------------------------------------------------------------------------
# LLM Call with Retry
# ---------------------------------------------------------------------------


def call_llm_with_retry(llm_client, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return llm_client.chat.completions.create(**kwargs)
        except Exception as e:
            wait_time = 2**attempt
            print(f"{C.YELLOW}[Retry]{C.RESET} {attempt + 1}/{max_retries} in {wait_time}s... {C.DIM}{e}{C.RESET}")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


def run_generation(cfg, headword, sense_id, data, llm_client, system_prompt, user_tmpl) -> list[dict]:
    """Run the generation agent for one (headword, sense_id) batch.

    Returns the parsed JSON output (list of sense groups with examples).
    """
    user_prompt = assemble_user_prompt(headword, sense_id, data, cfg, user_tmpl)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    max_tool_calls = cfg.agent.max_tool_calls

    for step in range(max_tool_calls):
        is_final_step = step == max_tool_calls - 1
        print(f"{C.DIM}  Step {step + 1}/{max_tool_calls}{C.RESET}")

        llm_params = {
            "model": cfg.api.model,
            "messages": messages,
            "tools": ALL_TOOLS,
            "tool_choice": "none" if is_final_step else "auto",
        }

        response = call_llm_with_retry(llm_client, max_retries=cfg.api.max_retries, **llm_params)
        response_message = response.choices[0].message
        messages.append(response_message.model_dump())

        # Log reasoning
        if response_message.content:
            preview = response_message.content  # [:300]
            print(f"{C.MAGENTA}[LLM]{C.RESET} {preview}{'...' if len(response_message.content) > 300 else ''}")

        # Check for JSON output
        if response_message.content and "```json" in response_message.content.lower():
            try:
                json_text = response_message.content.split("```json")[1].split("```")[0]
                result = json.loads(json_text)
                if not isinstance(result, list):
                    result = [result]

                # Count total examples
                total_examples = sum(len(group.get("examples", [])) for group in result)
                target_examples = sum(len(group.get("examples", [])) for group in result if group.get("is_target", False))
                contrastive_examples = total_examples - target_examples
                print(f"{C.GREEN}[OK]{C.RESET} {target_examples} target + {contrastive_examples} contrastive = {total_examples} examples")
                return result

            except (json.JSONDecodeError, IndexError) as e:
                print(f"{C.RED}[Error]{C.RESET} JSON parse error: {e}")
                if is_final_step:
                    return []
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your JSON output could not be parsed.\n\nError: {e}\n\nPlease output valid JSON.",
                    }
                )
                continue

        # Handle tool calls
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tc_args = tool_call.function.arguments
                print(f"{C.YELLOW}[Tool Call]{C.RESET} {tool_call.function.name}({tc_args})")
                tool_result = execute_tool(tool_call, data)
                print(f"{C.CYAN}[Tool Result]{C.RESET} {tool_result}")
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
        else:
            if is_final_step:
                print(f"{C.RED}[Error]{C.RESET} Final step without JSON output")
                return []
            messages.append(
                {
                    "role": "user",
                    "content": "Continue. When ready, output your final ```json output.",
                }
            )

        # Countdown reminders
        remaining = max_tool_calls - step - 1
        if remaining == 1:
            messages.append(
                {
                    "role": "user",
                    "content": "LAST STEP. You MUST produce your final ```json output NOW.",
                }
            )
        elif 1 < remaining <= 3:
            messages.append(
                {
                    "role": "user",
                    "content": f"{remaining} steps remaining. Produce your final ```json output soon.",
                }
            )

    print(f"{C.RED}[Error]{C.RESET} Max tool calls reached")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_output_key(headword: str, sense_id: str) -> str:
    """Create a filesystem-safe key for (headword, sense_id)."""
    safe_hw = headword.replace(" ", "_").replace("/", "_").replace("'", "")
    safe_sid = sense_id.replace(" ", "_").replace("/", "_").replace("'", "")
    return f"{safe_hw}__{safe_sid}"


def main():
    parser = argparse.ArgumentParser(description="CAD/ESAD deliberate practice drill generation")
    parser.add_argument("--config", type=str, default="sdg/conf/conf_cad_drill.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    run_id = uuid.uuid4().hex[:8]
    print(f"{C.CYAN}Run ID: {run_id}{C.RESET}")

    # Load data
    data = load_all_data(cfg)
    system_tmpl, user_tmpl = load_templates(cfg)
    system_prompt = system_tmpl.render()

    plan_df = data["plan_df"]

    if cfg.generation.get("polysemous_only", False):
        poly_hws = plan_df.groupby("headword").size()
        poly_hws = set(poly_hws[poly_hws > 1].index)
        plan_df = plan_df[plan_df["headword"].isin(poly_hws)].reset_index(drop=True)
        data["plan_df"] = plan_df
        print(f"{C.CYAN}Polysemous only: {len(poly_hws)} headwords, {len(plan_df)} batches, {plan_df['n_to_generate'].sum():,} target examples{C.RESET}")

    # Output directory — includes run_id so multiple runs don't overwrite
    output_dir = Path(cfg.output_path) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{C.DIM}  Output dir: {output_dir}{C.RESET}")

    # Setup LLM client
    llm_client = openai.OpenAI(base_url=cfg.api.base_url, api_key=os.environ.get("INFERENCE_NVIDIA_API_KEY"))

    # Process plan
    total_generated = 0
    skipped = 0

    plan_df = plan_df.sample(frac=1.0).reset_index(drop=True)
    for idx, (_, plan_row) in enumerate(tqdm(plan_df.iterrows(), total=len(plan_df), desc="Generating")):
        headword = plan_row["headword"]
        sense_id = plan_row["sense_id"]
        n_to_gen = plan_row["n_to_generate"]
        output_key = make_output_key(headword, sense_id)
        out_path = output_dir / f"{output_key}.json"

        # Skip if already done
        if out_path.exists():
            skipped += 1
            continue

        print(f"\n{C.CYAN}{'=' * 60}{C.RESET}")
        print(f'{C.CYAN}[{idx + 1}/{len(plan_df)}] {headword} → "{plan_row["gloss"][:50]}" ({n_to_gen} target){C.RESET}')

        try:
            sense_groups = run_generation(cfg, headword, sense_id, data, llm_client, system_prompt, user_tmpl)

            # Save result
            result = {
                "run_id": run_id,
                "headword": headword,
                "sense_id": sense_id,
                "gloss": plan_row["gloss"],
                "domain": plan_row["domain"],
                "n_attestations": int(plan_row["n_attestations"]),
                "n_requested": int(n_to_gen),
                "sense_groups": sense_groups,
            }
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

            n = sum(len(g.get("examples", [])) for g in sense_groups)
            total_generated += n
            print(f"{C.GREEN}[Saved]{C.RESET} {n} examples → {out_path.name}")

        except Exception as e:
            print(f"{C.RED}[Error]{C.RESET} Failed: {e}")
            continue

    print(f"\n{C.GREEN}Done!{C.RESET} Generated {total_generated:,} examples ({skipped:,} skipped as already done)")


if __name__ == "__main__":
    main()
