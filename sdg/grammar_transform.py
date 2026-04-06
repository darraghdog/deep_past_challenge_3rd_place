"""Grammar-guided transformation agent for generating synthetic OA training data.

Takes seed sentences from training data, samples grammar rules from GOA,
and uses an LLM to apply applicable rules as transformations — producing
novel (transliteration, translation) pairs grounded in real attested forms.

Usage:
    uv run python sdg/grammar_transform.py
    uv run python sdg/grammar_transform.py --config sdg/conf/conf_transform.yaml
"""

import argparse
import json
import os
import random
import sys
import time
import unicodedata
import uuid
from pathlib import Path

import kagglehub
import openai
import pandas as pd
from jinja2 import Template
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from seed import get_oare_for_seed, get_seeds, load_oare

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
        "description": ("Use this to reason through which grammar rules apply to the seed sentence and what each transformation would look like. Plan the OA form changes and English translation changes before producing output."),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string", "description": "Your reasoning about rule applicability and transformations"},
            },
            "required": ["thought"],
        },
    },
}

LOOKUP_FORM_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_form",
        "description": ("Find real parallel examples containing a specific OA word form. Use this to verify that a form you want to use in a transformation actually appears in real texts, and to see how it's translated in context."),
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

ALL_TOOLS = [THINK_TOOL, LOOKUP_FORM_TOOL]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_all_data(cfg):
    """Load all required datasets."""
    print(f"{C.CYAN}Loading datasets...{C.RESET}")

    # Training data for lookup_form
    input_dir = Path(kagglehub.dataset_download(cfg.datasets.dpc))
    train_df = pd.read_parquet(input_dir / "train.parquet")
    train_df = train_df[train_df.language == "en"].reset_index(drop=True)
    print(f"{C.DIM}  Training data: {len(train_df):,} English sentences{C.RESET}")

    # OARE annotations
    oare_df = load_oare()
    print(f"{C.DIM}  OARE: {len(oare_df):,} word annotations{C.RESET}")

    # Grammar rules pool
    rules_path = Path(cfg.rules_path)
    with open(rules_path) as f:
        all_rules = json.load(f)
    print(f"{C.DIM}  Grammar rules: {len(all_rules):,}{C.RESET}")

    # Morphological annotations guide
    morph_path = Path(cfg.morph_path)
    morph_text = morph_path.read_text() if morph_path.exists() else ""

    return {
        "train_df": train_df,
        "oare_df": oare_df,
        "all_rules": all_rules,
        "morph_text": morph_text,
    }


# ---------------------------------------------------------------------------
# Tool Execution
# ---------------------------------------------------------------------------


def _strip_diacritics(s: str) -> str:
    """Strip diacritics for fuzzy form matching (š→s, ṣ→s, ṭ→t, á→a, ú→u, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c)
    )


def lookup_form(form: str, train_df: pd.DataFrame, n: int = 3) -> str:
    """Find parallel examples containing a specific word form."""
    norm_form = _strip_diacritics(form)
    examples = []
    for _, row in train_df.sample(frac=1.0, random_state=None).iterrows():
        tokens = [_strip_diacritics(t.strip(".,;:!?()")) for t in row["transliteration"].split()]
        if norm_form in tokens:
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


def execute_tool(tool_call, context: dict) -> str:
    """Execute a tool call and return the result string."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "think":
        return "Thought recorded."
    elif name == "lookup_form":
        return lookup_form(args["form"], context["train_df"], args.get("n", 3))
    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------


def build_oare_text(doc_oare: pd.DataFrame, max_lines: int = 150) -> str:
    """Format OARE annotations as text for the prompt."""
    lines = []
    for _, row in doc_oare.head(max_lines).iterrows():
        g = str(row.grammar) if pd.notna(row.grammar) else "-"
        gl = str(row.gloss) if pd.notna(row.gloss) else "-"
        lem = str(row.lemma) if pd.notna(row.lemma) else "-"
        lines.append(f"L{str(row.line_num):>4s} {str(row.word):22s} | {lem:20s} | {gl:30s} | {g}")
    return "\n".join(lines)


def build_prompts(seed_row, doc_oare, sampled_rules, morph_text):
    """Build system and user prompts from templates."""
    system_template = Template((SDG_DIR / "prompts" / "system_transform.j2").read_text())
    user_template = Template((SDG_DIR / "prompts" / "user_transform.j2").read_text())

    system_prompt = system_template.render(
        morphological_annotations=morph_text,
    )

    oare_text = build_oare_text(doc_oare)

    user_prompt = user_template.render(
        seed_transliteration=seed_row["transliteration"],
        seed_translation=seed_row["translation"],
        seed_genre=seed_row.get("document_type", "unknown"),
        oare_annotations=oare_text,
        grammar_rules=sampled_rules,
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


def call_llm_with_retry(llm_client, max_retries=3, **kwargs):
    """Call LLM with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return llm_client.chat.completions.create(**kwargs)
        except Exception as e:
            wait_time = 2**attempt
            print(f"{C.YELLOW}[Retry]{C.RESET} {attempt + 1}/{max_retries} in {wait_time}s... {C.DIM}{e}{C.RESET}")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {max_retries} attempts")


def run_transform(cfg, seed_row, context, llm_client) -> list[dict]:
    """Run the transformation agent for one seed sentence.

    Returns list of generated (transliteration, translation) pairs.
    """
    oare_df = context["oare_df"]
    all_rules = context["all_rules"]
    morph_text = context["morph_text"]

    # Get OARE annotations for this seed's document
    doc_oare = get_oare_for_seed(seed_row, oare_df)

    # Sample grammar rules
    n_rules = cfg.agent.n_rules
    sampled_rules = random.sample(all_rules, min(n_rules, len(all_rules)))

    # Build prompts
    system_prompt, user_prompt = build_prompts(seed_row, doc_oare, sampled_rules, morph_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # print(f"{C.CYAN}System prompt:{C.RESET}")
    # print(system_prompt)
    print(f"{C.CYAN}User prompt:{C.RESET}")
    print(user_prompt)

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

        # Log reasoning text (full, no truncation)
        if response_message.content:
            print(f"{C.MAGENTA}[Reasoning]{C.RESET}")
            print(response_message.content)

        # Log tool calls
        if response_message.tool_calls:
            for tc in response_message.tool_calls:
                print(f"{C.YELLOW}[Tool Call]{C.RESET} {tc.function.name}({tc.function.arguments})")

        # Check for JSON output
        if response_message.content and "```json" in response_message.content.lower():
            try:
                json_text = response_message.content.split("```json")[1].split("```")[0]
                pairs = json.loads(json_text)
                if not isinstance(pairs, list):
                    pairs = [pairs]
                for p in pairs:
                    p["seed_oare_id"] = seed_row["oare_id"]
                    p["seed_sentence_id"] = int(seed_row["sentence_id"])
                print(f"{C.GREEN}[OK]{C.RESET} Generated {len(pairs)} pairs")
                return pairs
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
                tool_result = execute_tool(tool_call, context)
                print(f"{C.CYAN}[Tool Response]{C.RESET} {tool_call.function.name}:")
                print(tool_result)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
        else:
            if is_final_step:
                print(f"{C.RED}[Error]{C.RESET} Final step without JSON output")
                return []
            messages.append(
                {
                    "role": "user",
                    "content": "Continue. When ready, output your final JSON with the transformed pairs.",
                }
            )

        # Countdown reminders when running low on steps
        remaining = max_tool_calls - step - 1
        if remaining == 1:
            messages.append(
                {
                    "role": "user",
                    "content": "⚠️ LAST STEP. You MUST produce your final ```json output NOW. No more tool calls after this.",
                }
            )
        elif 1 < remaining <= 3:
            messages.append(
                {
                    "role": "user",
                    "content": f"⚠️ {remaining} steps remaining. Start wrapping up — produce your final ```json output soon.",
                }
            )

    print(f"{C.RED}[Error]{C.RESET} Max tool calls reached")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Grammar-guided transformation agent")
    parser.add_argument("--config", type=str, default="sdg/conf/conf_transform.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # Load data
    context = load_all_data(cfg)

    # Get seeds
    seeds = get_seeds(n=cfg.n_seeds)
    print(f"\n{C.CYAN}Seeds: {len(seeds):,} sentences{C.RESET}")

    # Output directory — one JSON per seed, named by UUID
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{C.DIM}  Output dir: {output_dir}{C.RESET}")

    # Setup LLM client
    llm_client = openai.OpenAI(base_url=cfg.api.base_url, api_key=os.environ.get("INFERENCE_NVIDIA_API_KEY"))

    # Process seeds
    total_generated = 0
    for idx, (_, seed_row) in enumerate(tqdm(seeds.iterrows(), total=len(seeds), desc="Seeds")):
        seed_id = str(uuid.uuid4())

        print(f"\n{C.CYAN}{'=' * 60}{C.RESET}")
        print(f"{C.CYAN}[{idx + 1}/{len(seeds)}] {seed_id}{C.RESET}")
        print(f"{C.DIM}  T: {seed_row['transliteration']}{C.RESET}")
        print(f"{C.DIM}  E: {seed_row['translation']}{C.RESET}")

        try:
            pairs = run_transform(cfg, seed_row, context, llm_client)

            # Save result
            result = {
                "seed_id": seed_id,
                "seed_oare_id": seed_row["oare_id"],
                "seed_sentence_id": int(seed_row["sentence_id"]),
                "seed_transliteration": seed_row["transliteration"],
                "seed_translation": seed_row["translation"],
                "seed_genre": seed_row.get("document_type", "unknown"),
                "generated_pairs": pairs,
            }
            out_path = output_dir / f"{seed_id}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

            n = len(pairs)
            total_generated += n
            print(f"{C.GREEN}[Saved]{C.RESET} {n} pairs → {out_path.name}")

        except Exception as e:
            print(f"{C.RED}[Error]{C.RESET} Seed failed: {e}")
            continue

    print(f"\n{C.GREEN}Done!{C.RESET} Generated {total_generated:,} pairs in {output_dir}")


if __name__ == "__main__":
    main()
