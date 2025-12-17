"""
NL4COP数据集管理器

提供从NL4COP数据集中按索引提取数据的核心功能。
"""

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class NL4COPManager:
    """
    NL4COP数据集管理器
    
    用于从NL4COP数据集中按索引提取数据。
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        """初始化NL4COP管理器"""
        if dataset_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.dataset_path = project_root / "benchmark" / "NL4COP"
        else:
            self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_path}")

        self._dataset = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """加载所有问题文件并按problem_id排序"""
        problems: List[Dict[str, Any]] = []

        json_files = sorted(glob.glob(str(self.dataset_path / "*.json")))

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    problem_data: Dict[str, Any] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"警告: 无法加载文件 {file_path}: {e}")
                continue

            problem_id = problem_data.get("problem_id") or Path(file_path).stem
            problem_data["problem_id"] = problem_id
            problems.append(problem_data)

        problems.sort(key=lambda item: item["problem_id"])
        return problems

    def _validate_index(self, index: int) -> None:
        """验证索引有效性"""
        if not (0 <= index < len(self._dataset)):
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self._dataset) - 1}]")

    def get_total_count(self) -> int:
        """获取数据集问题总数"""
        return len(self._dataset)

    def get_id(self, index: int) -> str:
        """根据索引获取问题ID"""
        self._validate_index(index)
        return str(self._dataset[index]["problem_id"])

    def get_input(self, index: int) -> str:
        """统一接口：获取输入文本"""
        self._validate_index(index)
        problem = self._dataset[index]

        description = str(problem["problem_description"]).strip()
        solution_format = str(problem["solution_format_description"]).strip()
        return f"{description}\n\n{solution_format}"

    def get_answer(self, index: int) -> str:
        """获取标准答案（reference_solution 字段）。"""
        self._validate_index(index)
        problem = self._dataset[index]
        answer = problem.get("reference_solution")
        if answer is None:
            return ""
        if isinstance(answer, str):
            return answer.strip()
        return json.dumps(answer, ensure_ascii=False)
