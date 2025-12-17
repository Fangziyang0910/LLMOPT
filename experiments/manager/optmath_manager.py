"""
Optmath 数据集管理器

提供从 Optmath 数据集中提取问题描述和答案的基础功能。
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Any, Optional


class OptmathManager:
    """
    Optmath 数据集管理器

    用于从 Optmath 数据集中提取问题描述和答案。
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        初始化 Optmath 管理器

        Args:
            dataset_path: 数据集文件路径，默认为项目根目录下的 benchmark/Optmath.jsonl
        """
        if dataset_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.dataset_path = project_root / "benchmark" / "Optmath.jsonl"
        else:
            self.dataset_path = Path(dataset_path)

        self._dataset = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据集（逐行 JSON）"""
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
                        f"第{line_num}行JSON解析错误: {e.msg}",
                        e.doc,
                        e.pos,
                    ) from e

                missing = {key for key in ("id", "en_question", "en_answer") if key not in record}
                if missing:
                    raise KeyError(f"记录缺少必要字段 {missing}: 行 {line_num}")

                records.append(record)

        if not records:
            raise ValueError(f"数据集为空: {self.dataset_path}")

        return records

    def _validate_index(self, index: int) -> None:
        """验证索引有效性"""
        if not (0 <= index < len(self._dataset)):
            raise IndexError(f"索引{index}超出范围，数据集大小: {len(self._dataset)}")

    def _get_record(self, index: int) -> Dict[str, Any]:
        """根据索引获取记录"""
        self._validate_index(index)
        return self._dataset[index]

    def get_total_count(self) -> int:
        """获取数据集记录总数"""
        return len(self._dataset)

    def get_id(self, index: int) -> str:
        """获取记录ID"""
        record = self._get_record(index)
        return str(record["id"])

    def get_answer(self, index: int) -> Union[str, int, float]:
        """获取英文答案（统一转字符串返回）"""
        record = self._get_record(index)
        answer = record.get("en_answer")
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer.strip()
        return json.dumps(answer, ensure_ascii=False)

    def get_input(self, index: int) -> str:
        """统一接口：获取英文输入文本"""
        record = self._get_record(index)
        question = record.get("en_question")
        return str(question).strip() if question is not None else ""
