"""读取结果目录，提取 code_run_output、题面与标准答案后汇总。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set
from pydantic import BaseModel, Field

# 导入各 benchmark manager
from experiments.manager import (  # type: ignore
    ComplexORManager,
    IndustryORManager,
    MamoManager,
    NL4COPManager,
    NL4LPManager,
    NL4OptManager,
    OptiBenchManager,
    OptmathManager,
)
from agents.core.llm import get_llm


PROJECT_ROOT = Path(__file__).parent.parent


MANAGER_BUILDERS = {
    "nl4lp": NL4LPManager,
    "complexor": ComplexORManager,
    "nl4opt": NL4OptManager,
    "optibench": OptiBenchManager,
    "optmath": OptmathManager,
    "industryor": IndustryORManager,
    "nl4cop": NL4COPManager,
    "mamo_easy": lambda: MamoManager(difficulty="easy"),
    "mamo_complex": lambda: MamoManager(difficulty="complex"),
}


CONFIG = {
    "benchmark": "industryor",
    "results_dir": PROJECT_ROOT / "results" / "20251210_IndustryOR_OPT-Agent_deepseek-v3.2",
    "output": PROJECT_ROOT / "results" / "validated_industryor_20251210.jsonl",
    # LLM 配置：供应商、模型、temperature
    "llm_provider": "deepseek",
    "llm_model": "deepseek-chat",
    "llm_temperature": 0.3,
}


def build_manager(benchmark: str):
    if benchmark not in MANAGER_BUILDERS:
        raise ValueError(f"未知 benchmark: {benchmark}")
    builder = MANAGER_BUILDERS[benchmark]
    return builder() if callable(builder) else builder


def build_id_index_map(manager: Any) -> Dict[str, int]:
    total = manager.get_total_count()
    return {str(manager.get_id(i)): i for i in range(total)}


def iter_result_files(results_dir: Path) -> Iterable[Path]:
    """遍历结果文件（*.json），按文件名排序。"""
    return sorted(results_dir.glob("*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_record(manager: Any, id_index: Dict[str, int], file_path: Path) -> Dict[str, Any]:
    record_id = file_path.stem
    result_data = load_json(file_path)
    predicted = result_data.get("code_run_output", "")

    index = id_index.get(record_id)
    if index is None:
        raise KeyError(f"结果文件名 {record_id} 未在数据集中找到对应 id")

    gt_answer = manager.get_answer(index)

    return {
        "id": record_id,
        "code_run_output": predicted,
        "gt_answer": gt_answer,
    }


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


class EvalResult(BaseModel):
    analysis: str = Field(..., description="Reasoning about correctness.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1.")
    is_correct: bool = Field(..., description="Whether model answer matches reference.true or false")


def judge_with_llm(llm: Any, record_id: str, predicted: str, reference: str) -> Dict[str, Any]:
    system_prompt = """你将得到两个答案：模型答案与参考答案。仅从答案是否一致的角度判断模型答案是否正确。
    若数值有微小差异（可能由计算精度导致），且不影响本质结果，则视为正确。对于目标值一致，但是具体取值不一样的情况，应该是存在多重最优解，应该视为争取。以目标值作为评判的主要标准。
    请用中文回答，输出分析、置信度（0-1），以及正确性判断（true或false）
    输出遵循如下格式：
    {
        "analysis": "分析结果",
        "confidence": "置信度",
        "is_correct": "正确性判断（true或false）"
    }
    请直接输出JSON字符串，不要添加任何其他内容。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"id: {record_id}\n\n"
                f"Model answer:\n{predicted}\n\n"
                f"Reference answer:\n{reference}"
            ),
        },
    ]

    model_with_struct = llm.with_structured_output(EvalResult)
    structured: EvalResult = model_with_struct.invoke(messages)
    return {
        "analysis": structured.analysis,
        "confidence": float(structured.confidence),
        "is_correct": bool(structured.is_correct),
        "id": record_id,
    }


def load_existing_ids(output_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rid = data.get("id")
                if rid is not None:
                    ids.add(str(rid))
            except Exception as e:
                print(f"[warn] 读取已存在结果时跳过第 {line_num} 行: {e}")
                continue
    return ids


def append_record(output_path: Path, rec: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False))
        f.write("\n")


def main() -> None:
    benchmark = CONFIG["benchmark"]
    results_dir = Path(CONFIG["results_dir"])
    output_path = Path(CONFIG["output"]) if CONFIG.get("output") else None

    if not results_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_dir}")

    manager = build_manager(benchmark)
    id_index = build_id_index_map(manager)
    llm = get_llm(
        CONFIG["llm_provider"],
        CONFIG["llm_model"],
        CONFIG["llm_temperature"],
    )

    files = list(iter_result_files(results_dir))
    total_files = len(files)
    if total_files == 0:
        print("结果目录为空，未执行校验。")
        return

    existing_ids: Set[str] = set()
    if output_path:
        existing_ids = load_existing_ids(output_path)
        print(f"已存在结果条数: {len(existing_ids)}，将跳过这些 id。")

    processed = 0
    skipped = 0

    for idx, fp in enumerate(files, 1):
        record_id = fp.stem
        if record_id in existing_ids:
            print(f"[{idx}/{total_files}] 跳过已存在 id: {record_id}")
            skipped += 1
            continue

        rec = extract_record(manager, id_index, fp)
        eval_result = judge_with_llm(
            llm=llm,
            record_id=rec["id"],
            predicted=_to_text(rec["code_run_output"]),
            reference=_to_text(rec["gt_answer"]),
        )
        rec.update(eval_result)

        if output_path:
            append_record(
                output_path,
                {
                    "id": rec["id"],
                    "is_correct": rec["is_correct"],
                    "confidence": rec["confidence"],
                    "analysis": rec["analysis"],
                },
            )
        processed += 1
        existing_ids.add(record_id)

        print(
            f"[{idx}/{total_files}] 已完成 id={record_id} "
            f"is_correct={rec.get('is_correct')} conf={rec.get('confidence')}"
        )

    print(
        f"完成。总文件 {total_files}，处理 {processed}，跳过 {skipped}。"
        + (f" 结果文件: {output_path}" if output_path else "")
    )


if __name__ == "__main__":
    main()
