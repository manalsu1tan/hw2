from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    full_sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []

    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(output_ids))
        full_sequences.append(prompt_ids + output_ids)

    max_len = max(len(sequence) - 1 for sequence in full_sequences)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0) or 0

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []
    for sequence, prompt_len, response_len in zip(full_sequences, prompt_lengths, response_lengths, strict=True):
        sequence_len = len(sequence) - 1
        pad_len = max_len - sequence_len
        input_ids.append(sequence[:-1] + [pad_token_id] * pad_len)
        labels.append(sequence[1:] + [pad_token_id] * pad_len)
        response_mask.append([False] * (prompt_len - 1) + [True] * response_len + [False] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    return (tensor * mask.to(dtype=tensor.dtype)).sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("number of rollout responses must be divisible by group_size")

    reward_infos = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor([info["reward"] for info in reward_infos], dtype=torch.float32)
    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        advantages = centered

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "reward_min": raw_rewards.min().item(),
        "reward_max": raw_rewards.max().item(),
        "format_reward_mean": float(sum(info.get("format_reward", 0.0) for info in reward_infos) / len(reward_infos)),
        "answer_reward_mean": float(sum(info.get("answer_reward", 0.0) for info in reward_infos) / len(reward_infos)),
    }
    return advantages.reshape(-1), raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    broadcast_advantages = advantages.expand_as(policy_log_probs)
    unclipped_objective = ratios * broadcast_advantages
    clipped_objective = clipped_ratios * broadcast_advantages
    loss = -torch.minimum(unclipped_objective, clipped_objective)
    metadata = {
        "clip_fraction": (clipped_objective < unclipped_objective).to(torch.float32).mean(),
        "mean_ratio": ratios.mean(),
    }
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    masked_loss = per_token_loss * response_mask.to(per_token_loss.dtype)
    per_example_loss = masked_loss.sum(dim=1) / response_mask.sum(dim=1).clamp_min(1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward(retain_graph=True)
    metadata = {
        **metadata,
        "unscaled_loss": per_example_loss.mean().detach(),
        "loss": loss.detach(),
    }
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    logs: list[dict[str, Any]] = []
    for idx, (prompt, response, ground_truth, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        item: dict[str, Any] = {
            "index": idx,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": reward_info.get("reward"),
            "format_reward": reward_info.get("format_reward"),
            "answer_reward": reward_info.get("answer_reward"),
        }
        if token_entropies is not None:
            item["token_entropy"] = token_entropies[idx]
        logs.append(item)
    return logs


def train_grpo(*args, **kwargs) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    if args:
        raise TypeError("train_grpo expects keyword arguments so experiment settings are explicit")

    policy: torch.nn.Module = kwargs["policy"]
    tokenizer = kwargs["tokenizer"]
    reward_fn: Callable[[str, str], dict[str, float]] = kwargs["reward_fn"]
    train_prompts: Sequence[str] = kwargs["train_prompts"]
    train_ground_truths: Sequence[str] = kwargs["train_ground_truths"]
    optimizer: torch.optim.Optimizer = kwargs.get("optimizer") or torch.optim.Adam(
        policy.parameters(),
        lr=kwargs.get("learning_rate", 1e-5),
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    n_grpo_steps = kwargs.get("n_grpo_steps", 8)
    rollout_batch_size = kwargs.get("rollout_batch_size", 32)
    group_size = kwargs.get("group_size", 8)
    advantage_eps = kwargs.get("advantage_eps", 1e-6)
    normalize_by_std = kwargs.get("normalize_by_std", True)
    sampling_temperature = kwargs.get("sampling_temperature", 1.0)
    sampling_min_tokens = kwargs.get("sampling_min_tokens", 4)
    sampling_max_tokens = kwargs.get("sampling_max_tokens", 256)
    epochs_per_rollout_batch = kwargs.get("epochs_per_rollout_batch", 1)
    train_batch_size = kwargs.get("train_batch_size", rollout_batch_size)
    gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 16)
    cliprange = kwargs.get("cliprange", 1.0)
    max_grad_norm = kwargs.get("max_grad_norm", 1.0)
    return_token_entropy = kwargs.get("return_token_entropy", False)
    device = kwargs.get("device") or next(policy.parameters()).device

    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")

    policy.to(device)
    policy.train()
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    history: list[dict[str, Any]] = []

    for step in range(n_grpo_steps):
        indices = random.sample(range(len(train_prompts)), k=n_prompts_per_rollout_batch)
        prompt_batch = [train_prompts[idx] for idx in indices]
        ground_truth_batch = [train_ground_truths[idx] for idx in indices]
        repeated_prompts = [prompt for prompt in prompt_batch for _ in range(group_size)]
        repeated_ground_truths = [ground_truth for ground_truth in ground_truth_batch for _ in range(group_size)]

        tokenized_prompts = tokenizer(repeated_prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        prompt_input_ids = tokenized_prompts["input_ids"].to(device)
        prompt_attention_mask = tokenized_prompts.get("attention_mask")
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.to(device)

        with torch.no_grad():
            generated = policy.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                do_sample=True,
                temperature=sampling_temperature,
                min_new_tokens=sampling_min_tokens,
                max_new_tokens=sampling_max_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        rollout_responses = tokenizer.batch_decode(generated[:, prompt_input_ids.shape[1] :], skip_special_tokens=True)

        tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        with torch.no_grad():
            old_log_probs = get_response_log_probs(policy, input_ids, labels, return_token_entropy=False)["log_probs"].detach()
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=list(repeated_ground_truths),
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        advantages = advantages.to(device).unsqueeze(1)

        update_logs: list[dict[str, float]] = []
        for _ in range(epochs_per_rollout_batch):
            order = torch.randperm(rollout_batch_size, device=device)
            for start in range(0, rollout_batch_size, train_batch_size):
                optimizer.zero_grad(set_to_none=True)
                train_indices = order[start : start + train_batch_size]
                for micro_start in range(0, len(train_indices), micro_train_batch_size):
                    micro_indices = train_indices[micro_start : micro_start + micro_train_batch_size]
                    scored = get_response_log_probs(
                        policy,
                        input_ids[micro_indices],
                        labels[micro_indices],
                        return_token_entropy=return_token_entropy,
                    )
                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=scored["log_probs"],
                        response_mask=response_mask[micro_indices],
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        advantages=advantages[micro_indices],
                        old_log_probs=old_log_probs[micro_indices],
                        cliprange=cliprange,
                    )
                    update_logs.append(
                        {
                            "loss": float(loss),
                            "clip_fraction": float(metadata["clip_fraction"].detach().cpu()),
                        }
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                update_logs[-1]["grad_norm"] = float(grad_norm)

        history.append(
            {
                "step": step,
                **reward_metadata,
                "raw_reward_mean": float(raw_rewards.mean()),
                "updates": update_logs,
                "sample_generations": log_generations(
                    repeated_prompts[: min(4, len(repeated_prompts))],
                    rollout_responses[: min(4, len(rollout_responses))],
                    list(repeated_ground_truths)[: min(4, len(repeated_ground_truths))],
                    [
                        reward_fn(response, ground_truth)
                        for response, ground_truth in zip(
                            rollout_responses[: min(4, len(rollout_responses))],
                            list(repeated_ground_truths)[: min(4, len(repeated_ground_truths))],
                            strict=True,
                        )
                    ],
                ),
            }
        )

    return {"history": history, "final_step": n_grpo_steps}
