"""统计验证结果的正确率。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).parent.parent

CONFIG = {
    # 已验证结果文件列表（JSONL，每行含 id/analysis/confidence/is_correct）
    "validated_paths": [
        PROJECT_ROOT / "results" / "validated_industryor_20251210.jsonl",
    ],
    # 可选：汇总结果输出为 JSON；为 None 时仅打印
    "output": None,
}


def iter_records(paths: List[Path]) -> Iterable[Dict]:
    for path in paths:
        if not path.exists():
            print(f"[warn] 文件不存在，跳过: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"[warn] 解析失败 {path}:{line_num}，跳过。原因: {e}")
                    continue


def main() -> None:
    paths = [Path(p) for p in CONFIG["validated_paths"]]
    total = 0
    correct = 0
    conf_sum = 0.0

    for rec in iter_records(paths):
        total += 1
        if rec.get("is_correct"):
            correct += 1
        try:
            conf_sum += float(rec.get("confidence", 0) or 0)
        except Exception:
            pass

    if total == 0:
        print("无有效记录，无法统计。")
        return

    acc = correct / total
    avg_conf = conf_sum / total
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "avg_confidence": avg_conf,
        "files": [str(p) for p in paths],
    }

    print(
        f"统计完成：\n"
        f"  样本数: {total}\n"
        f"  正确数: {correct}\n"
        f"  准确率: {acc:.4f}\n"
        f"  平均置信度: {avg_conf:.4f}"
    )

    output = CONFIG.get("output")
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {out_path}")


if __name__ == "__main__":
    main()

