"""Graph Agent evaluation framework.

Usage examples:
1) Run end-to-end evaluation on a QA csv:
   python multi_agent/eval_framework.py \
       --input-csv multi_agent/eval_data/qa_dataset.csv \
       --output-dir multi_agent/output/eval \
       --text-only

2) Only summarize an existing case metrics csv:
   python multi_agent/eval_framework.py \
       --input-csv multi_agent/output/eval/case_metrics.csv \
       --output-dir multi_agent/output/eval \
       --summarize-only

Input CSV expected columns:
- required: question
- optional: case_id, reference, modality, difficulty, domain_tag
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

try:
    from .graph_agent import create_research_graph
except ImportError:
    from graph_agent import create_research_graph


REQUIRED_QUESTION_COL = "question"
TEXT_MODALITY = "text"


@dataclass
class CaseMetric:
    run_id: str
    case_id: str
    question: str
    reference: str
    modality: str
    difficulty: str
    domain_tag: str
    latency_ms: float
    tool_calls_total: int
    tool_calls_repeat: int
    used_fallback: int
    iteration_count_max: int
    answer_text: str
    retrieved_context_count: int
    success: Optional[int]
    success_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "question": self.question,
            "reference": self.reference,
            "modality": self.modality,
            "difficulty": self.difficulty,
            "domain_tag": self.domain_tag,
            "latency_ms": round(self.latency_ms, 3),
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_repeat": self.tool_calls_repeat,
            "used_fallback": self.used_fallback,
            "iteration_count_max": self.iteration_count_max,
            "answer_text": self.answer_text,
            "retrieved_context_count": self.retrieved_context_count,
            "success": self.success,
            "success_score": None if self.success_score is None else round(self.success_score, 4),
        }


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fa5 ]+", "", lowered)
    return lowered


def _char_f1(reference: str, prediction: str) -> float:
    ref = _normalize_text(reference)
    pred = _normalize_text(prediction)
    if not ref or not pred:
        return 0.0

    ref_chars = list(ref)
    pred_chars = list(pred)

    ref_count: Dict[str, int] = {}
    for c in ref_chars:
        ref_count[c] = ref_count.get(c, 0) + 1

    overlap = 0
    for c in pred_chars:
        if ref_count.get(c, 0) > 0:
            overlap += 1
            ref_count[c] -= 1

    precision = overlap / max(1, len(pred_chars))
    recall = overlap / max(1, len(ref_chars))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_tool_calls_from_messages(messages: Iterable[Any]) -> List[str]:
    tool_keys: List[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args", {}) or {}
            query = str(args.get("query", ""))
            tool_keys.append(f"{name}::{query}")
    return tool_keys


def _extract_answer_and_context_count(state_values: Dict[str, Any]) -> tuple[str, int]:
    answer_text = ""
    context_count = 0

    messages = state_values.get("messages", [])
    if messages:
        last = messages[-1]
        answer_text = last.content if hasattr(last, "content") else str(last)

    agent_answers = state_values.get("agent_answers", [])
    if isinstance(agent_answers, list):
        context_count = sum(1 for x in agent_answers if isinstance(x, dict) and x.get("context"))

    return answer_text, context_count


def _summarize_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    total_cases = int(len(df))
    latency_series = df["latency_ms"].dropna()

    success_series = df["success"].dropna()
    task_success_rate = float(success_series.mean()) if not success_series.empty else None

    total_tool_calls = int(df["tool_calls_total"].sum()) if "tool_calls_total" in df else 0
    total_tool_repeat = int(df["tool_calls_repeat"].sum()) if "tool_calls_repeat" in df else 0
    repeat_retrieval_rate = (
        float(total_tool_repeat / total_tool_calls) if total_tool_calls > 0 else None
    )

    fallback_series = df["used_fallback"].dropna()
    fallback_rate = float(fallback_series.mean()) if not fallback_series.empty else None

    latency_p50 = float(np.percentile(latency_series, 50)) if not latency_series.empty else None
    latency_p95 = float(np.percentile(latency_series, 95)) if not latency_series.empty else None

    by_difficulty: Dict[str, Dict[str, Any]] = {}
    if "difficulty" in df.columns:
        for diff, sub in df.groupby("difficulty"):
            lat = sub["latency_ms"].dropna()
            succ = sub["success"].dropna()
            by_difficulty[str(diff)] = {
                "sample_size": int(len(sub)),
                "task_success_rate": float(succ.mean()) if not succ.empty else None,
                "fallback_rate": float(sub["used_fallback"].mean()) if not sub.empty else None,
                "latency_p95_ms": float(np.percentile(lat, 95)) if not lat.empty else None,
            }

    return {
        "sample_size": total_cases,
        "task_success_rate": task_success_rate,
        "repeat_retrieval_rate": repeat_retrieval_rate,
        "fallback_rate": fallback_rate,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "total_tool_calls": total_tool_calls,
        "total_repeat_tool_calls": total_tool_repeat,
        "by_difficulty": by_difficulty,
    }


def _validate_input_columns(df: pd.DataFrame) -> None:
    if REQUIRED_QUESTION_COL not in df.columns:
        raise ValueError(f"input csv must include column: {REQUIRED_QUESTION_COL}")


def _run_one_case(
    app: Any,
    run_id: str,
    row: pd.Series,
    success_threshold: float,
) -> CaseMetric:
    case_id = str(row.get("case_id") or f"case_{uuid.uuid4().hex[:8]}")
    question = str(row.get("question") or "").strip()
    reference = str(row.get("reference") or "").strip()
    modality = str(row.get("modality") or TEXT_MODALITY).strip().lower()
    difficulty = str(row.get("difficulty") or "unknown").strip().lower()
    domain_tag = str(row.get("domain_tag") or "default").strip().lower()

    config = {"configurable": {"thread_id": f"eval_{run_id}_{case_id}"}}
    inputs = {
        "originalQuery": question,
        "messages": [HumanMessage(content=question)],
    }

    start = time.perf_counter()
    used_fallback = 0
    seen_tool_keys: set[str] = set()
    repeat_calls = 0
    total_calls = 0
    iteration_count_max = 0

    # Stream updates to capture route-level telemetry without changing graph logic.
    for event in app.stream(inputs, config=config, stream_mode="updates"):
        event_str = str(event)
        if "fallback_response" in event_str:
            used_fallback = 1

        event_json = json.loads(json.dumps(event, default=str))
        for node_update in event_json.values():
            if isinstance(node_update, dict):
                iteration_count_max = max(
                    iteration_count_max,
                    _safe_int(node_update.get("iteration_count"), iteration_count_max),
                )

                # Prefer explicit retrieval_keys if present in state updates.
                retrieval_keys = node_update.get("retrieval_keys")
                if isinstance(retrieval_keys, list):
                    for key in retrieval_keys:
                        key_s = str(key)
                        total_calls += 1
                        if key_s in seen_tool_keys:
                            repeat_calls += 1
                        else:
                            seen_tool_keys.add(key_s)

                # Fallback: parse tool_calls from message-like string dumps.
                msg_dump = str(node_update.get("messages", ""))
                if "tool_calls" in msg_dump and "text_retrieval" in msg_dump:
                    pass

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    state_snapshot = app.get_state(config)
    state_values = state_snapshot.values if hasattr(state_snapshot, "values") else {}
    answer_text, context_count = _extract_answer_and_context_count(state_values)

    success = None
    success_score = None
    if reference:
        success_score = _char_f1(reference, answer_text)
        success = 1 if success_score >= success_threshold else 0

    return CaseMetric(
        run_id=run_id,
        case_id=case_id,
        question=question,
        reference=reference,
        modality=modality,
        difficulty=difficulty,
        domain_tag=domain_tag,
        latency_ms=latency_ms,
        tool_calls_total=total_calls,
        tool_calls_repeat=repeat_calls,
        used_fallback=used_fallback,
        iteration_count_max=iteration_count_max,
        answer_text=answer_text,
        retrieved_context_count=context_count,
        success=success,
        success_score=success_score,
    )


def _run_eval(
    input_csv: Path,
    output_dir: Path,
    text_only: bool,
    success_threshold: float,
) -> tuple[Path, Path]:
    df = pd.read_csv(input_csv)
    _validate_input_columns(df)

    if text_only and "modality" in df.columns:
        df = df[df["modality"].fillna(TEXT_MODALITY).str.lower() == TEXT_MODALITY].copy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    app = create_research_graph()

    case_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        metric = _run_one_case(
            app=app,
            run_id=run_id,
            row=row,
            success_threshold=success_threshold,
        )
        case_rows.append(metric.to_dict())

    case_df = pd.DataFrame(case_rows)
    summary = _summarize_metrics(case_df)
    summary["run_id"] = run_id
    summary["input_csv"] = str(input_csv)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")

    output_dir.mkdir(parents=True, exist_ok=True)
    case_out = output_dir / f"case_metrics_{run_id}.csv"
    summary_out = output_dir / f"summary_{run_id}.json"

    case_df.to_csv(case_out, index=False, encoding="utf-8")
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return case_out, summary_out


def _summarize_existing_case_csv(input_csv: Path, output_dir: Path) -> Path:
    df = pd.read_csv(input_csv)
    required = {"latency_ms", "tool_calls_total", "tool_calls_repeat", "used_fallback"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"case metrics csv missing columns: {missing}")

    summary = _summarize_metrics(df)
    summary["run_id"] = str(df.get("run_id", pd.Series(["unknown"]))[0])
    summary["input_csv"] = str(input_csv)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"summary_from_case_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph_agent with reproducible KPI outputs.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input csv path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./multi_agent/output/eval"),
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only evaluate rows with modality=text (if modality column exists).",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.55,
        help="Threshold for char-F1 success decision when reference exists.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run inference; summarize an existing case_metrics csv only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.summarize_only:
        out = _summarize_existing_case_csv(args.input_csv, args.output_dir)
        print(f"[OK] summary generated: {out}")
        return

    case_out, summary_out = _run_eval(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        text_only=args.text_only,
        success_threshold=args.success_threshold,
    )
    print(f"[OK] case metrics: {case_out}")
    print(f"[OK] summary: {summary_out}")


if __name__ == "__main__":
    main()
