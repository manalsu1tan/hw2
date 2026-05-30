from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, extract_answer_from_tags, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return list(dataset)


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=example["question"]) for example in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    generations: list[dict[str, Any]] = []
    reward_sums = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

    for prompt, request_output in zip(prompts, outputs, strict=True):
        responses = [completion.text for completion in request_output.outputs]
        ground_truth = getattr(request_output, "ground_truth", None)
        if ground_truth is None:
            ground_truth = ""
        if len(responses) == 1:
            scored_response = responses[0]
        else:
            voted_answer = majority_vote_tagged_answers(responses)
            scored_response = f"<answer>{voted_answer}</answer>" if voted_answer is not None else responses[0]
        reward_info = reward_fn(scored_response, ground_truth)
        for key in reward_sums:
            reward_sums[key] += reward_info.get(key, 0.0)
        generations.append(
            {
                "prompt": prompt,
                "responses": responses,
                "scored_response": scored_response,
                "model_answer": extract_answer_from_tags(scored_response),
                "ground_truth": ground_truth,
                "reward_info": reward_info,
            }
        )

    n = max(len(generations), 1)
    return {
        "metrics": {key: value / n for key, value in reward_sums.items()},
        "generations": generations,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    _run_baseline(output_path=output_path, use_cot=False, n=1)


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    _run_baseline(output_path=output_path, use_cot=True, n=1)


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    _run_baseline(output_path=output_path, use_cot=True, n=k)


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def _extract_gsm8k_answer(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _run_baseline(output_path: Path, use_cot: bool, n: int) -> None:
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples("test")[:DEFAULT_VALIDATION_SIZE]
    prompts = build_prompts(examples, str(get_prompt_template(use_cot)))
    ground_truths = [_extract_gsm8k_answer(example["answer"]) for example in examples]
    sampling_params = SamplingParams(
        n=n,
        temperature=0.0 if n == 1 else 0.7,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    model = LLM(model=DEFAULT_MODEL_NAME, dtype="bfloat16")
    outputs = model.generate(prompts, sampling_params)

    generations: list[dict[str, Any]] = []
    reward_sums = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    tie_count = 0
    for prompt, ground_truth, request_output in zip(prompts, ground_truths, outputs, strict=True):
        responses = [completion.text for completion in request_output.outputs]
        if n == 1:
            scored_response = responses[0]
        else:
            answers = [answer for answer in (extract_answer_from_tags(response) for response in responses) if answer]
            if answers:
                counts = {answer: answers.count(answer) for answer in set(answers)}
                top_count = max(counts.values())
                tie_count += sum(1 for count in counts.values() if count == top_count) > 1
            voted_answer = majority_vote_tagged_answers(responses)
            scored_response = f"<answer>{voted_answer}</answer>" if voted_answer is not None else responses[0]
        reward_info = answer_tag_reward_fn(scored_response, ground_truth)
        for key in reward_sums:
            reward_sums[key] += reward_info.get(key, 0.0)
        generations.append(
            {
                "prompt": prompt,
                "responses": responses,
                "scored_response": scored_response,
                "ground_truth": ground_truth,
                "reward_info": reward_info,
            }
        )

    total = max(len(generations), 1)
    results = {
        "model": DEFAULT_MODEL_NAME,
        "use_cot": use_cot,
        "self_consistency_k": n,
        "metrics": {key: value / total for key, value in reward_sums.items()},
        "tie_rate": tie_count / total if n > 1 else None,
        "generations": generations,
    }
    write_evaluation_results(results, output_path)
