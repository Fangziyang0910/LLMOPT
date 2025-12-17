"""
统一批量运行 experiments 下的数据集脚本（支持 CLI 参数，提供默认值）。

使用示例：
    uv run experiments/run_dataset.py
    uv run experiments/run_dataset.py --dataset optmath --run-name 20251214_optmath --start-index 0 --end-index 10
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from agents.core.llm import get_llm
from agents.runner import run_agent
from agents.schema import ContextSchema
from experiments.manager import (
    ComplexORManager,
    IndustryORManager,
    MamoManager,
    NL4COPManager,
    NL4LPManager,
    NL4OptManager,
    OptiBenchManager,
    OptmathManager,
)

ALLOWED_RESULT_KEYS = {
    "original_question",
    "consensus",
    "consensus_reached",
    "blueprint_audit_round",
    "negotiation_round",
    "controversy",
    "blueprint_audit_passed",
    "blueprint_audit_feedback",
    "modeling_phase",
    "code",
    "code_run_success",
    "code_run_output",
    "code_run_count",
    "code_audit_count",
    "code_audit_passed",
    "code_audit_feedback",
    "code_run_history",
}


def filter_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """去除对话类字段，仅保留指定键值。"""
    return {key: value for key, value in result.items() if key in ALLOWED_RESULT_KEYS}


def validate_index_range(start_index: int, end_index: int | None, total: int) -> int:
    if start_index >= total:
        raise ValueError(f"START_INDEX ({start_index}) must be less than total problem count ({total}).")

    resolved_end = end_index if end_index is not None else total
    if resolved_end > total:
        raise ValueError(f"END_INDEX ({resolved_end}) must be less than or equal to total problem count ({total}).")
    if resolved_end <= start_index:
        raise ValueError(f"END_INDEX ({resolved_end}) must be greater than START_INDEX ({start_index}).")

    return resolved_end


def build_context_glm() -> ContextSchema:
    return ContextSchema(
        llm=get_llm("zai", "glm-4.6", 1, extra_body={"thinking": {"type": "disabled"}}),
        tools_llm=get_llm("zai", "glm-4.6", 1, extra_body={"thinking": {"type": "disabled"}}),
        code_llm=get_llm("iflow", "qwen3-coder-plus", 0),
    )


def build_context_deepseek() -> ContextSchema:
    return ContextSchema(
        llm=get_llm("deepseek", "deepseek-reasoner", 1, max_tokens=65536),
        tools_llm=get_llm("deepseek", "deepseek-chat", 1),
        code_llm=get_llm("deepseek", "deepseek-chat", 0),
    )


CONTEXT_REGISTRY: Dict[str, Callable[[], ContextSchema]] = {
    "glm": build_context_glm,
    "deepseek": build_context_deepseek,
}


DATASET_REGISTRY = {
    "optmath": (OptmathManager, "Optmath"),
    "optibench": (OptiBenchManager, "OptiBench"),
    "nl4opt": (NL4OptManager, "NL4Opt"),
    "nl4lp": (NL4LPManager, "NL4LP"),
    "nl4cop": (NL4COPManager, "NL4COP"),
    "complexor": (ComplexORManager, "ComplexOR"),
    "industryor": (IndustryORManager, "IndustryOR"),
    "mamo_easy": (lambda: MamoManager(difficulty="easy"), "Mamo-easy"),
    "mamo_complex": (lambda: MamoManager(difficulty="complex"), "Mamo-complex"),
}


def get_manager_and_label(dataset: str) -> Tuple[Any, str]:
    entry = DATASET_REGISTRY.get(dataset)
    if entry is None:
        supported = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown DATASET: {dataset}. Supported: {supported}")

    manager_factory, label = entry
    return manager_factory(), label


def main(dataset: str, run_name: str, start_index: int, end_index: int | None, context_name: str) -> None:
    manager, label = get_manager_and_label(dataset)
    total = manager.get_total_count()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    context_builder = CONTEXT_REGISTRY.get(context_name)
    if context_builder is None:
        supported = ", ".join(sorted(CONTEXT_REGISTRY.keys()))
        raise ValueError(f"Unknown context: {context_name}. Supported: {supported}")
    context = context_builder()
    resolved_end_index = validate_index_range(start_index, end_index, total)

    for index in range(start_index, resolved_end_index):
        try:
            problem_id = manager.get_id(index)
            question = manager.get_input(index)

            output_path = results_dir / f"{problem_id}.json"
            if output_path.exists():
                print(f"Skipping {label} item {problem_id} ({index + 1}/{total}): result exists.")
                continue

            print(f"Running {label} item {problem_id} ({index + 1}/{total})")
            result = run_agent(question=question, context=context)
            filtered = filter_result(result)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Error running {label} item at index {index} ({index + 1}/{total}): {exc}")
            traceback.print_exc()
            continue

    print(f"结果已保存到: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量运行指定数据集，并将每题结果保存为 JSON。")
    parser.add_argument("--context", default="deepseek", choices=sorted(CONTEXT_REGISTRY.keys()), help="上下文配置预设（选择不同的 llm/tools_llm/code_llm）。")
    parser.add_argument("--dataset", default="industryor", choices=sorted(DATASET_REGISTRY.keys()), help="数据集名称（mamo 用 mamo_easy/mamo_complex 区分难度）。")
    parser.add_argument("--run-name", default="20251210_IndustryOR_OPT-Agent_deepseek-v3.2", help="结果目录名（保存到 results/<run-name>/）。")
    parser.add_argument("--start-index", type=int, default=0, help="起始索引（含）。")
    parser.add_argument("--end-index", type=int, default=None, help="结束索引（不含），不传则跑到最后。")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        run_name=args.run_name,
        start_index=args.start_index,
        end_index=args.end_index,
        context_name=args.context,
    )
