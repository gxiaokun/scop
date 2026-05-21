from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import random
from collections import defaultdict

import instructor
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm import tqdm


# ============================================================
# Structured output schema
# ============================================================

class TemporalQARepairOutput(BaseModel):
    """
    Structured repair result for one temporal QA record.
    """

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(
        ...,
        description=(
            "A brief repair note in 1-2 sentences. "
            "State whether the answer was changed, and summarize "
            "the main question repair. Do not provide detailed reasoning."
        ),
        min_length=1,
        max_length=360,
    )

    repaired_answer: List[str] = Field(
        ...,
        description=(
            "The repaired answer list. Preserve original answer items "
            "unless a temporal-format correction is clearly necessary."
        ),
    )

    rewritten_question: str = Field(
        ...,
        description=(
            "A natural, grammatical, semantically precise rewritten "
            "temporal QA question."
        ),
        min_length=5,
    )

    @field_validator("thought")
    @classmethod
    def normalize_thought(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @field_validator("rewritten_question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        if not value.endswith("?"):
            value += "?"
        return value

    @field_validator("repaired_answer")
    @classmethod
    def validate_answer_list(cls, value: List[str]) -> List[str]:
        if not isinstance(value, list):
            raise ValueError("repaired_answer must be a list.")
        if any(not isinstance(item, str) for item in value):
            raise ValueError("Every repaired_answer item must be a string.")
        return [item.strip() for item in value]



SYSTEM_PROMPT = """
You are a strict temporal QA dataset repair assistant. 

You are highly sensitive to the formatting of temporal expressions and should only modify the time format in answers when an explicit correction is required. You will infer the intent of each question by analyzing the provided gold_events, answer_type, question_type, and temporal_relation, and polish the phrasing where necessary to ensure it is natural, fluent, and semantically precise.
Since the refined questions and answers will be deployed in a QA system, please ensure the final output is articulated as naturally as possible while strictly preserving the original query events, reference events, answer slots, temporal relations, and factual integrity.

Rules of rewrite question text:[

Before rewriting, first understand the original question internally:
- What temporal operation is it asking about?
- Is it asking about chronological rank, duration comparison, longest/shortest ranking, before/after offset, union, intersection, overlap, combined time periods, entity retrieval, or time interval retrieval?
- What answer form does the gold answer imply?

Do not output this analysis.
Only output the rewritten question.

Guidelines:
**Most Important** The temporal granularity must be kept at the day level.
1. Whenever possible, follow the word order direction of the specific events in gold_events.
2. If the question contains a specific date, preserve it verbatim. Do not modify its format, granularity, or value in any way.
3. Make the question more natural and less template-like.
4. Keep the original temporal meaning.
5. Use the gold answer only to understand the expected question form.
6. Do not insert, reveal, or change the gold answer.
7. Do not collapse a temporal-reasoning question into a generic fact question.
8. If the original question involves combined, union, intersection, overlap, or shared time periods, preserve that temporal operation.
9. If the original question involves chronological rank or ordering, preserve the rank/order question form.
10. If the original question involves before/after offsets, preserve the offset conditions.
11. If the original question involves duration comparison, preserve the comparison meaning.
12. You may naturally rephrase entities and event descriptions when it improves fluency, as long as the question still clearly refers to the same events.
13. Avoid excessive paraphrasing that changes the task being asked.
14. Do not add new dates, durations, calculations, or external facts.
15. Output only the rewritten question in the structured field.
]

Rules of rewrite answer text:[
First understand what temporal granularity the question is asking for.
Then repair only the gold answer so that it matches the question intent and the gold events.

Important:
1. Do not use model predictions.
2. Do not infer new facts.
3. Do not change the factual event, entity, or relation.
4. Do not change entity answers.
5. Do not invent or remove answers.
6. Use the gold events as the factual reference.
7. If the current gold answer already matches the question intent, keep it unchanged.
8. If the question asks for a year, answer with years such as "1983".
9. If the question asks when something happened, started, ended, ceased, or finished, answer with dates such as "1983-01-01".
10. If the question asks for time periods or periods, answer with intervals such as "1983-01-01 to 1983-01-01".
11. If the current gold answer contains multiple start dates, begin dates, end dates, finish dates, cease dates, or stop dates, keep all of them. Do not reduce them to only the first or only the last date, even if the question contains words like "first", "begin", "start", "end", "finish", "cease", or "stop".
12. Do not merge, delete, deduplicate, reorder, or add duplicate answers. Only change the temporal granularity of existing answer items when needed.
]

Your thought must be brief.

Return only:
- thought
- repaired_answer
- rewritten_question
""".strip()


FEW_SHOTS: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
    (
     {"question": "Which organisation is award received by Keiju Kobayashi from 1985-01-01 to 1985-01-01?",
        "current_answer": [
        "Medal with Purple Ribbon"
        ],
        "gold_events": [
        "Keiju Kobayashi|award received|Medal with Purple Ribbon|1985-01-01|1985-01-01"
        ],
        "question_level": "simple",
        "answer_type": "object",
        "temporal_relation": "timeline",
    },{
        "thought": (
                "The answer is an entity and does not need to be changed. The temporal granularity must be kept at the day level. Refer to the word order of the gold events in the event field, and only make minor adjustments."
            ),
            "repaired_answer": ["Medal with Purple Ribbon"],
            "rewritten_question": (
                "Keiju Kobayashi received an award from which organisation on 1985-01-01?"
            ),
    }
    ),
    (
        {
            "question": (
                "Which organisation is member of sports teamed by Michael Nelson before Conal Platt member of sports team Lincoln City F.C.?"
            ),
            "current_answer": ["Bury F.C."],
            "gold_events": [
                "Michael Nelson|member of sports team|Bury F.C.|2001-01-01|2003-01-01",
                "Conal Platt|member of sports team|Lincoln City F.C.|2011-01-01|2011-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "object",
            "temporal_relation": "X < Y",
        },
        {
            "thought": (
                "The answer is an entity answer and remains unchanged. "
                "The question was rewritten to express the before relation naturally "
                "and remove malformed wording."
            ),
            "repaired_answer": ["Bury F.C."],
            "rewritten_question": (
                "Which team did Michael Nelson play for before Conal Platt played for Lincoln City F.C.?"
            ),
        },
    ),
    (
        {
            "question": (
                "Who nominated for Nobel Prize in Chemistry Giuseppe Cardone member of sports team Vicenza Calcio?"
            ),
            "current_answer": ["Giuseppe Cardone"],
            "gold_events": [
                "Karl August Folkers|nominated for|Nobel Prize in Chemistry|1962-01-01|1962-01-01",
                "Giuseppe Cardone|member of sports team|Vicenza Calcio|1999-01-01|1999-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "subject",
            "temporal_relation": "X > Y",
        },
        {
            "thought": (
                "The answer is a subject entity and remains unchanged. The question was repaired to explicitly express the intended after relation."
            ),
            "repaired_answer": ["Giuseppe Cardone"],
            "rewritten_question": (
                "Who played for Vicenza Calcio after Karl August Folkers was nominated for the Nobel Prize in Chemistry?"
            ),
        },
    ),
    (
        {
            "question": (
                "63188 days Mike Thompson position held United States representative, in which organisation, Copley Medal winner?"
            ),
            "current_answer": ["Jacques Charles François Sturm"],
            "gold_events": [
                "Copley Medal|winner|Jacques Charles François Sturm|1840-01-01|1840-01-01",
                "Mike Thompson|position held|United States representative|2013-01-01|2013-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "object",
            "temporal_relation": "duration_before",
        },
        {
            "thought": (
                "The answer is an entity answer and remains unchanged. "
                "The question was rewritten to preserve the exact day offset in natural wording."
            ),
            "repaired_answer": ["Jacques Charles François Sturm"],
            "rewritten_question": (
                "Who won the Copley Medal 63,188 days before Mike Thompson served as a United States representative?"
            ),
        },
    ),
    (
        {
            "question": (
                "Who award received Ordem do Mérito Cultural 40178 days Theodor Klaehn educated at University of Rostock?"
            ),
            "current_answer": ["Ana Miranda"],
            "gold_events": [
                "Theodor Klaehn|educated at|University of Rostock|1905-01-01|1907-01-01",
                "Ana Miranda|award received|Ordem do Mérito Cultural|2017-01-01|2017-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "subject",
            "temporal_relation": "duration_after",
        },
        {
            "thought": (
                "The answer remains unchanged. "
                "The question was rewritten to preserve the duration-after constraint clearly."
            ),
            "repaired_answer": ["Ana Miranda"],
            "rewritten_question": (
                "Who received the Ordem do Mérito Cultural 40,178 days after Theodor Klaehn studied at the University of Rostock?"
            ),
        },
    ),
    (
        {
            "question": (
                "Which organisation is position helded Robert Le Gall during the period when Dan Aykroyd award received Golden Raspberry Award for Worst Supporting Actor?"
            ),
            "current_answer": ["abbot"],
            "gold_events": [
                "Robert Le Gall|position held|abbot|1983-01-01|2001-01-01",
                "Dan Aykroyd|award received|Golden Raspberry Award for Worst Supporting Actor|1991-01-01|1991-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "object",
            "temporal_relation": "X di Y",
        },
        {
            "thought": (
                "The answer is an entity or literal answer and remains unchanged. "
                "The question was rewritten to remove malformed wording and preserve the containment relation."
            ),
            "repaired_answer": ["abbot"],
            "rewritten_question": (
                "Which position did Robert Le Gall hold when Dan Aykroyd received the Golden Raspberry Award for Worst Supporting Actor?"
            ),
        },
    ),
    (
        {
            "question": (
                "Who position held member of the German Bundestag during Noël Mamère position held member of the French National Assembly?"
            ),
            "current_answer": ["Thomas Gambke"],
            "gold_events": [
                "Thomas Gambke|position held|member of the German Bundestag|2009-01-01|2013-01-01",
                "Noël Mamère|position held|member of the French National Assembly|2012-01-01|2017-01-01",
            ],
            "question_type": "timeline_position_retrieval_temporal_constrained_retrieval",
            "answer_type": "subject",
            "temporal_relation": "X o Y",
        },
        {
            "thought": (
                "The answer remains unchanged. "
                "The question was rewritten to express the temporal overlap more precisely."
            ),
            "repaired_answer": ["Thomas Gambke"],
            "rewritten_question": (
                "Who served as a member of the German Bundestag during a period that overlapped with Noël Mamère's service in the French National Assembly?"
            ),
        },
    ),
    (
        {
            "question": (
                "Based on the start time, what is the chronological rank of the event involving 'Barry Miller' among the following events: Barry Miller member of sports team Doncaster Rovers F.C. Brett Williams member of sports team Stoke City F.C. and Nikolai Bogolyubov award received Order of Lenin?"
            ),
            "current_answer": ["3"],
            "gold_events": [
                "Nikolai Bogolyubov|award received|Order of Lenin|1967-01-01|1967-01-01",
                "Brett Williams|member of sports team|Stoke City F.C.|1993-01-01|1993-01-01",
                "Barry Miller|member of sports team|Doncaster Rovers F.C.|2000-01-01|2003-01-01",
            ],
            "question_type": "timeline_position_retrieval*3",
            "answer_type": "relation_ranking",
            "temporal_relation": "rank_start_time",
        },
        {
            "thought": (
                "The rank answer remains unchanged. "
                "The question was rewritten to make the ranking criterion easier to read."
            ),
            "repaired_answer": ["3"],
            "rewritten_question": (
                "Based on their start times, what is the chronological rank of the event involving Barry Miller among the following events: Barry Miller being a member of sports team Doncaster Rovers F.C., Brett Williams being a member of sports team Stoke City F.C., and Nikolai Bogolyubov receiving the award Order of Lenin?"
            ),
        },
    ),
]


# ============================================================
# Regex helpers
# ============================================================

YEAR_RE = re.compile(r"^\d{3,4}$")
DATE_RE = re.compile(r"^\d{3,4}-\d{2}-\d{2}$")
INTERVAL_RE = re.compile(
    r"^\d{3,4}-\d{2}-\d{2}\s+to\s+\d{3,4}-\d{2}-\d{2}$"
)
DURATION_RE = re.compile(r"^\d+\s+days$")
INTEGER_RE = re.compile(r"^\d+$")

COMPARISON_SET = {"longer", "shorter", "equal"}


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def normalize_question_text(text: str, fallback: str) -> str:
    repaired = normalize_text(text)

    if not repaired:
        repaired = normalize_text(fallback)

    if not repaired.endswith("?"):
        repaired += "?"

    return repaired


def normalize_answer_list(answer: Any) -> List[str]:
    if answer is None:
        return []
    if not isinstance(answer, list):
        raise ValueError("answer must be a list.")
    return [normalize_text(item) for item in answer]


def is_year(text: str) -> bool:
    return bool(YEAR_RE.fullmatch(normalize_text(text)))


def is_date(text: str) -> bool:
    return bool(DATE_RE.fullmatch(normalize_text(text)))


def is_interval(text: str) -> bool:
    return bool(INTERVAL_RE.fullmatch(normalize_text(text)))


def is_duration(text: str) -> bool:
    return bool(DURATION_RE.fullmatch(normalize_text(text)))


def is_integer_string(text: str) -> bool:
    return bool(INTEGER_RE.fullmatch(normalize_text(text)))


def is_comparison(text: str) -> bool:
    return normalize_text(text).lower() in COMPARISON_SET


def is_temporal_like(text: str) -> bool:
    return (
        is_year(text)
        or is_date(text)
        or is_interval(text)
        or is_duration(text)
    )


def lower_str(value: Any) -> str:
    return str(value or "").strip().lower()


def answer_must_remain_unchanged(
    answer_type: Any,
    temporal_relation: Any,
) -> bool:
    answer_type_l = lower_str(answer_type)
    temporal_relation_l = lower_str(temporal_relation)

    if answer_type_l in {"subject", "object", "entity", "literal"}:
        return True

    if "ranking" in answer_type_l or temporal_relation_l.startswith("rank_"):
        return True

    if "comparison" in answer_type_l or temporal_relation_l == "duration_compare":
        return True

    return False


def temporal_answer_change_allowed(
    answer_type: Any,
    question_type: Any,
) -> bool:
    answer_type_l = lower_str(answer_type)
    question_type_l = lower_str(question_type)

    tokens = [
        "timestamp",
        "date",
        "year",
        "interval",
        "range",
        "duration",
        "time_period",
        "union",
        "intersection",
    ]

    return any(token in answer_type_l for token in tokens) or any(
        token in question_type_l for token in tokens
    )


def validate_repaired_answer(
    original_answer: List[str],
    repaired_answer: List[str],
    question_type: Any,
    answer_type: Any,
    temporal_relation: Any,
) -> None:
    if len(original_answer) != len(repaired_answer):
        raise ValueError(
            "Answer count changed: "
            f"original={len(original_answer)}, repaired={len(repaired_answer)}"
        )

    if answer_must_remain_unchanged(answer_type, temporal_relation):
        if original_answer != repaired_answer:
            raise ValueError(
                "Entity/rank/comparison answer was changed, which is not allowed."
            )
        return

    if original_answer == repaired_answer:
        return

    if not temporal_answer_change_allowed(answer_type, question_type):
        raise ValueError(
            "Answer changed, but this answer type is not allowed to change."
        )

    for item in repaired_answer:
        if not is_temporal_like(item):
            raise ValueError(
                f"Changed answer is not a valid temporal format: {item}"
            )


# ============================================================
# Prompt construction
# ============================================================

def to_prompt_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "question": item.get("question", ""),
        "current_answer": item.get("answer", []),
        "gold_events": item.get("events", []),
        "question_type": item.get("question_type"),
        "answer_type": item.get("answer_type"),
        "temporal_relation": item.get("temporal_relation"),
    }


def build_messages(
    item: Dict[str, Any],
    validation_feedback: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for shot_input, shot_output in FEW_SHOTS:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    shot_input,
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    shot_output,
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        )

    current_payload = json.dumps(
        to_prompt_payload(item),
        ensure_ascii=False,
        indent=2,
    )

    if validation_feedback:
        current_payload += (
            "\n\nThe previous output failed validation for this reason:\n"
            f"{validation_feedback}\n"
            "Regenerate a valid result. Keep the answer unchanged unless a "
            "temporal-format correction is clearly necessary."
        )

    messages.append(
        {
            "role": "user",
            "content": current_payload,
        }
    )

    return messages


# ============================================================
# File I/O
# ============================================================

def atomic_save_json(path: str, data: Any) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def load_data(
    input_path: str,
    checkpoint_path: str,
    resume: bool,
) -> List[Dict[str, Any]]:
    if resume and os.path.exists(checkpoint_path):
        print(f"checkpoint, resume from {checkpoint_path}...")
        with open(checkpoint_path, "r", encoding="utf-8") as file:
            return json.load(file)

    print(f"Read from input file {input_path}...")
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    return data


# ============================================================
# Client
# ============================================================

def build_client(
    api_key: str,
    base_url: str,
    timeout: int,
):
    if not api_key:
        raise ValueError(
            "API key is missing. Pass --api-key or set OPENAI_API_KEY."
        )

    raw_client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=1,
    )

    return instructor.from_openai(
        raw_client,
        mode=instructor.Mode.JSON,
    )


# ============================================================
# Single-item repair
# ============================================================

def repair_one_qa(
    client,
    item: Dict[str, Any],
    model_name: str,
    instructor_max_retries: int,
    validation_feedback: Optional[str] = None,
) -> TemporalQARepairOutput:
    result: TemporalQARepairOutput = client.chat.completions.create(
        model=model_name,
        response_model=TemporalQARepairOutput,
        max_retries=instructor_max_retries,
        messages=build_messages(
            item=item,
            validation_feedback=validation_feedback,
        ),
    )
    return result


def process_one_item(
    client,
    idx: int,
    item: Dict[str, Any],
    model_name: str,
    instructor_max_retries: int,
    local_retry: int,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    original_question = item.get("question", "")
    original_answer = normalize_answer_list(item.get("answer", []))

    if not original_question:
        return idx, None, "Missing question"

    last_error: Optional[str] = None

    for _ in range(local_retry + 1):
        try:
            result = repair_one_qa(
                client=client,
                item=item,
                model_name=model_name,
                instructor_max_retries=instructor_max_retries,
                validation_feedback=last_error,
            )

            repaired_answer = normalize_answer_list(result.repaired_answer)
            rewritten_question = normalize_question_text(
                result.rewritten_question,
                fallback=original_question,
            )

            validate_repaired_answer(
                original_answer=original_answer,
                repaired_answer=repaired_answer,
                question_type=item.get("question_type"),
                answer_type=item.get("answer_type"),
                temporal_relation=item.get("temporal_relation"),
            )

            final_answer = (
                item.get("answer", [])
                if repaired_answer == original_answer
                else repaired_answer
            )

            repaired_payload = {
                "question": rewritten_question,
                "answer": final_answer,
                "repair_thought": normalize_text(result.thought),
                "answer_changed": final_answer != item.get("answer", []),
                "question_changed": rewritten_question != original_question,
            }

            return idx, repaired_payload, None

        except Exception as exc:
            last_error = str(exc)

    return idx, None, last_error

# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair temporal QA questions and conservatively repair answer formats."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the repaired output JSON file.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to checkpoint file. "
            "Default: <output>.qa_repair_checkpoint"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Limit the number of samples to process, stratified by question_level. "
            "Useful for debugging. Default: process all."
        ),
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help=(
            "API key. If omitted, read from OPENAI_API_KEY."
        ),
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help=(
            "OpenAI-compatible base URL. "
            "Default: OPENAI_BASE_URL or https://api.openai.com/v1"
        ),
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help=(
            "Model name. Default: OPENAI_MODEL or gpt-5-mini."
        ),
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("TEMPORAL_QA_MAX_WORKERS", "16")),
        help="Number of worker threads. Default: 16.",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=int(os.getenv("TEMPORAL_QA_SAVE_INTERVAL", "200")),
        help="Save checkpoint every N completed samples. Default: 200.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if it exists. Default: enabled.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume from checkpoint.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TEMPORAL_QA_TIMEOUT", "120")),
        help="Request timeout in seconds. Default: 120.",
    )

    parser.add_argument(
        "--local-retry",
        type=int,
        default=int(os.getenv("TEMPORAL_QA_LOCAL_RETRY", "1")),
        help="Extra local retry count per sample. Default: 1.",
    )

    parser.add_argument(
        "--instructor-max-retries",
        type=int,
        default=int(os.getenv("TEMPORAL_QA_INSTRUCTOR_MAX_RETRIES", "3")),
        help="Instructor internal structured-output retries. Default: 3.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("TEMPORAL_QA_SLEEP_SECONDS", "0")),
        help="Optional sleep after each completed future. Default: 0.",
    )

    parser.add_argument(
        "--keep-original-question",
        action="store_true",
        default=True,
        help="Keep original question in output. Default: enabled.",
    )

    parser.add_argument(
        "--drop-original-question",
        action="store_false",
        dest="keep_original_question",
        help="Do not keep original question in output.",
    )

    parser.add_argument(
        "--keep-original-answer",
        action="store_true",
        default=True,
        help="Keep original answer in output. Default: enabled.",
    )

    parser.add_argument(
        "--drop-original-answer",
        action="store_false",
        dest="keep_original_answer",
        help="Do not keep original answer in output.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint
        else args.output + ".qa_repair_checkpoint"
    )

    client = build_client(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    data = load_data(
        input_path=args.input,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )


    if args.test_size is not None and args.test_size < len(data):
        target_size = args.test_size
        

        level_groups = defaultdict(list)
        for dp in data:
            lvl = str(dp.get("question_level", "unknown")).lower()
            level_groups[lvl].append(dp)
            
        sampled_dataset = []
        remaining_to_sample = target_size
        

        random.seed(42) 

        allocations = {}
        for lvl, group in level_groups.items():
            alloc_size = int((len(group) / len(data)) * target_size)
            alloc_size = min(alloc_size, len(group))
            allocations[lvl] = alloc_size
            
            sampled_dataset.extend(random.sample(group, alloc_size))
            remaining_to_sample -= alloc_size
            
   
        if remaining_to_sample > 0:
            remaining_pool = [dp for dp in data if dp not in sampled_dataset]
            sampled_dataset.extend(random.sample(remaining_pool, min(remaining_to_sample, len(remaining_pool))))
            

        random.shuffle(sampled_dataset)
        data = sampled_dataset
        
        print(f"\n[Debug Subset] Stratified sampled test_size={target_size} | Base Allocations: {allocations}")
    # ==========================================

    success_count = 0
    fail_count = 0
    skipped_count = 0
    futures = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, item in enumerate(data):
            if item.get("_qa_repaired") is True:
                skipped_count += 1
                continue

            futures.append(
                executor.submit(
                    process_one_item,
                    client,
                    idx,
                    item,
                    args.model,
                    args.instructor_max_retries,
                    args.local_retry,
                )
            )

        print(f"all samples num: {len(data)}")
        print(f"skip processed num: {skipped_count}")
        print(f"pending num: {len(futures)}\n")

        with tqdm(
            total=len(futures),
            desc="Repairing temporal QA",
            unit="sample",
        ) as progress:
            completed_in_session = 0

            for future in as_completed(futures):
                idx, repaired_payload, error_msg = future.result()
                completed_in_session += 1

                if error_msg is None and repaired_payload is not None:
                    item = data[idx]

                    if (
                        args.keep_original_question
                        and "original_question_before_repair" not in item
                    ):
                        item["original_question_before_repair"] = item.get(
                            "question", ""
                        )

                    if (
                        args.keep_original_answer
                        and "original_answer_before_repair" not in item
                    ):
                        item["original_answer_before_repair"] = item.get(
                            "answer", []
                        )

                    item["question"] = repaired_payload["question"]
                    item["answer"] = repaired_payload["answer"]
                    item["qa_repair_thought"] = repaired_payload["repair_thought"]
                    item["qa_repair_answer_changed"] = repaired_payload[
                        "answer_changed"
                    ]
                    item["qa_repair_question_changed"] = repaired_payload[
                        "question_changed"
                    ]
                    item["_qa_repaired"] = True

                    success_count += 1

                else:
                    fail_count += 1
                    data[idx]["qa_repair_error"] = error_msg
                    print(
                        f"\n[ERROR] index={idx}, id={data[idx].get('id')} failed: {error_msg}"
                    )

                progress.update(1)
                progress.set_postfix(
                    success=success_count,
                    fail=fail_count,
                    skipped=skipped_count,
                )

                if completed_in_session % args.save_interval == 0:
                    atomic_save_json(checkpoint_path, data)

                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

    atomic_save_json(checkpoint_path, data)

    clean_data = []
    for item in data:
        clean_item = dict(item)
        clean_item.pop("_qa_repaired", None)
        clean_data.append(clean_item)

    atomic_save_json(args.output, clean_data)

    print("\Done")
    print(f"output file: {args.output}")
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"success: {success_count}, "
        f"fail: {fail_count}, "
        f"skipped: {skipped_count}"
    )


if __name__ == "__main__":
    main()