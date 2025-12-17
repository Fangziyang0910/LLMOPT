"""
NL4LP 数据集管理器

从 benchmark/NL4LP.jsonl 加载题目与答案，并提供统一访问接口。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class NL4LPManager:
    """
    NL4LP 数据集管理器
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        """
        初始化 NL4LP 管理器

        Args:
            dataset_path: 数据集文件路径，默认使用 benchmark/NL4LP.jsonl
        """
        if dataset_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.dataset_path = project_root / "benchmark" / "NL4LP.jsonl"
        else:
            self.dataset_path = Path(dataset_path)

        self._dataset = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据集并校验必要字段。"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")

        records: List[Dict[str, Any]] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"第{line_num}行JSON解析错误: {e.msg}", e.doc, e.pos
                    ) from e

                missing = {key for key in ("id", "question", "answer") if key not in record}
                if missing:
                    raise KeyError(f"记录缺少必要字段 {missing}: 行 {line_num}")

                records.append(record)

        if not records:
            raise ValueError(f"数据集为空: {self.dataset_path}")

        return records

    def _validate_index(self, index: int) -> None:
        """验证索引有效性。"""
        if not (0 <= index < len(self._dataset)):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self._dataset) - 1}]")

    def _get_record(self, index: int) -> Dict[str, Any]:
        """内部获取记录。"""
        self._validate_index(index)
        return self._dataset[index]

    def get_total_count(self) -> int:
        """获取数据集记录总数。"""
        return len(self._dataset)

    def get_id(self, index: int) -> str:
        """获取记录 ID。"""
        record = self._get_record(index)
        return str(record["id"])

    def get_input(self, index: int) -> str:
        """统一接口：获取输入文本（题面描述）。"""
        record = self._get_record(index)
        question = record.get("question")
        if question is None:
            raise KeyError("记录中不存在字段: question")
        return str(question).strip()

    def get_answer(self, index: int) -> Any:
        """获取答案（统一转字符串返回）。"""
        record = self._get_record(index)
        answer = record.get("answer")
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer.strip()
        return str(answer).strip()
