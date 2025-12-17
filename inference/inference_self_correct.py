from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Optional, Tuple

import subprocess

from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import generate_prompt
from prompts.self_correction_prompt import self_correction as build_self_correction_prompt


# load model and tokenizer
path = ""
path_t = ""
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path_t)


# inference to get five elements
def infer_five_elem(question: str, feedback: Optional[str] = None) -> Optional[str]:
    messages = [
        {"role": "user", "content": generate_prompt.Q2F(question, feedback)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")

    if "```text" in response:
        return response.split("```text")[1].split("```")[0]
    elif "```plaintext" in response:
        return response.split("```plaintext")[1].split("```")[0]
    elif "```" in response:
        return response.split("```")[1].split("```")[0]
    else:
        return None


# inference to get pyomo python code
def infer_code(five_elem: str, feedback: Optional[str] = None) -> str:
    messages = [
        {"role": "user", "content": generate_prompt.F2C(five_elem, feedback)}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ans = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return ans.split("```python")[1].split("```")[0].replace('print("\\\\\n', 'print("').replace('print(f"\\\\\n', 'print(f"')


def build_code_feedback(prev_code: str, error_feedback: str) -> str:
    prev_code = (prev_code or "").strip()
    error_feedback = (error_feedback or "").strip()
    return (
        "这是你之前构建的代码，如下：\n"
        f"{prev_code}\n\n"
        "但是存在错误，错误反馈如下：\n"
        f"{error_feedback}\n\n"
        "请你修正。"
    )


# execute the code
def test_code(code_str: str) -> Tuple[str, str]:
    ans = subprocess.run(
        [sys.executable, "-c", code_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return str(ans.stdout.decode("gbk", errors="ignore")), str(
        ans.stderr.decode("gbk", errors="ignore")
    )


@dataclass(frozen=True)
class SelfCorrectionResult:
    five_ok: Optional[bool]
    code_ok: Optional[bool]
    analysis: str


def parse_self_correction(text: str) -> SelfCorrectionResult:
    five_ok: Optional[bool] = None
    code_ok: Optional[bool] = None
    analysis = ""

    normalized = text.replace("\r\n", "\n")
    for line in normalized.splitlines():
        lower = line.strip().lower()
        if five_ok is None and lower.startswith("the five-element is"):
            if "true" in lower:
                five_ok = True
            elif "false" in lower:
                five_ok = False
        if code_ok is None and lower.startswith("the code is"):
            if "true" in lower:
                code_ok = True
            elif "false" in lower:
                code_ok = False

    if five_ok is None or code_ok is None:
        matches = re.findall(r"\b(true|false)\b", normalized, flags=re.IGNORECASE)
        values = [m.lower() == "true" for m in matches[:2]]
        if five_ok is None:
            five_ok = values[0] if len(values) >= 1 else True
        if code_ok is None:
            code_ok = values[1] if len(values) >= 2 else True

    if "Analysis:" in normalized:
        analysis = normalized.split("Analysis:", 1)[1].strip()
    else:
        analysis = normalized.strip()

    return SelfCorrectionResult(
        five_ok=five_ok,
        code_ok=code_ok,
        analysis=analysis,
    )


def self_correction_judge(
    question: str,
    five: str,
    code: str,
    output: str,
    error: str,
) -> SelfCorrectionResult:
    judge_prompt = build_self_correction_prompt(question, five, code, output, error)
    messages = [{"role": "user", "content": judge_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=8192)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.replace("\\\\n", "\n").replace("&#39;","'").replace("&lt;", "<").replace("&gt;", ">").replace("\\\\\"","\"")
    return parse_self_correction(response)


def run_self_correct(question: str) -> dict:
    five_elem: Optional[str] = None
    code_str = ""
    out_log = ""
    err_log = ""
    judge: Optional[SelfCorrectionResult] = None

    self_correction_times = 0
    need_new_five = True
    five_feedback: Optional[str] = None
    code_feedback: Optional[str] = None

    attempt = 0
    for attempt in range(1, 12 + 1):
        if need_new_five:
            five_elem = infer_five_elem(question, five_feedback)
            code_feedback = None

        five_text = five_elem or ""
        code_str = infer_code(five_text, code_feedback)
        out_log, err_log = test_code(code_str)
        judge = self_correction_judge(question, five_text, code_str, out_log, err_log)

        if attempt >= 12:
            break

        if judge.five_ok is True and judge.code_ok is True:
            break

        self_correction_times += 1

        if judge.five_ok is False or (judge.five_ok is False and judge.code_ok is False):
            five_feedback = (
                "这是你之前构建的五要素，如下：\n"
                f"{five_text.strip()}\n\n"
                "但是存在错误，错误反馈如下：\n"
                f"{judge.analysis.strip()}\n\n"
                "请你修正。"
            )
            need_new_five = True
            code_feedback = None
            continue

        if judge.five_ok is True and judge.code_ok is False:
            error_feedback = (
                "运行输出如下：\n"
                f"{out_log}\n\n"
                "运行错误如下：\n"
                f"{err_log}\n\n"
                "自修正分析如下：\n"
                f"{judge.analysis}"
            )
            code_feedback = build_code_feedback(code_str, error_feedback)
            need_new_five = False
            five_feedback = None
            continue
        break

    return {
        "five_elem": five_elem or "",
        "code": code_str,
        "run_output": out_log,
        "run_error": err_log,
        "self_correction_times": self_correction_times,
        "attempts": attempt,
        "final_judge": {
            "five_ok": judge.five_ok if judge is not None else None,
            "code_ok": judge.code_ok if judge is not None else None,
            "analysis": judge.analysis if judge is not None else "",
        },
    }
