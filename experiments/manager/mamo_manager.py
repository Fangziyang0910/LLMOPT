"""
Mamo 数据集管理器

支持按难度（easy / complex）加载数据集，并提供统一数据访问接口。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MamoManager:
    """
    Mamo 数据集管理器

    通过难度选择加载对应数据集，供实验脚本按索引访问。
    """

    def __init__(self, dataset_path: Optional[str] = None, difficulty: str = "easy") -> None:
        """
        初始化 Mamo 管理器

        Args:
            dataset_path: 数据集文件路径；如未指定则根据 difficulty 选择 benchmark 下的默认文件
            difficulty: "easy" 或 "complex"，用于选择默认数据集文件
        """
        difficulty = difficulty.lower().strip()
        if difficulty not in {"easy", "complex"}:
            raise ValueError(f"未知难度: {difficulty}，仅支持 easy 或 complex")

        if dataset_path is None:
            project_root = Path(__file__).parent.parent.parent
            filename = "Mamo_easy.jsonl" if difficulty == "easy" else "Mamo_complex.jsonl"
            self.dataset_path = project_root / "benchmark" / filename
        else:
            self.dataset_path = Path(dataset_path)

        self._dataset = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据集（逐行 JSON），并校验必要字段。"""
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

                missing = {key for key in ("id", "Question") if key not in record}
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
        """统一接口：获取输入文本。"""
        record = self._get_record(index)
        question = record.get("Question")
        return str(question).strip() if question is not None else ""

    def get_answer(self, index: int) -> str:
        """获取答案文本。"""
        record = self._get_record(index)
        answer = record.get("Answer")
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer.strip()
        return json.dumps(answer, ensure_ascii=False)

