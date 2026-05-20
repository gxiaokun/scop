from pydantic import BaseModel, Field, model_validator
from typing import List, Tuple, Optional
import re


from src.config.rag_config import TemporalDatasets
from src.utils.llm_utils import LLMTaskExecutor
from src.llm.fewshots import MULTI_EXTR_FEW_SHOTS, CORN_EXTR_FEW_SHOTS

from src.config.rag_config import RAGConfig


class ExtractionSchema(BaseModel):
    anchor_triples: List[Tuple[str, str, str]] = Field(
        default_factory=list,
        description=(
            "Known complete event triples from the question. "
            "Each triple must be [subject, relation, object]. "
            "Do not include '?' or pure temporal expressions."
        ),
    )

    search_triple: Optional[Tuple[str, str, str]] = Field(
        default=None,
        description=(
            "The unique target event triple being asked about. "
            "Use exactly one '?' for the unknown subject or object. "
            "Use null if the question asks for time or temporal computation instead of an unknown entity."
        ),
    )


SYSTEM_PROMPT = """
You are a retrieval-oriented temporal question triple conversion system.

Your task is to convert a temporal question into a structured event representation for downstream knowledge-base event retrieval and temporal reasoning.

Output fields:
1. anchor_triples:
   Known complete event triples from the question.
   Each triple must be [subject, relation, object].
   anchor_triples must not contain "?".

2. search_triple:
   The target event triple being asked about.
   If it exists, it must contain exactly one "?" in subject or object.
   The relation must never be "?".
   If the question asks for time or temporal computation instead of an unknown entity, search_triple must be null.

Core principle:
Triples express only event content.
Temporal expressions and temporal relations must not enter any triple slot.

Do not put the following into subject, relation, or object:
- before / after / first / last
- same day / same month / same year
- on DATE / in MONTH / in YEAR
- pure dates, months, years, time points, or time intervals
- temporal placeholders such as time, date, year, month, day

I. Determine the question type first

A. Unknown-entity question:
If the question asks for an unknown participant or value, such as who / which country / with whom / to whom / whom / what award / which sports team / which organization, construct search_triple with exactly one "?".

Examples:
- Who visited France? -> ["?", "visit", "France"]
- Which country did Oman negotiate with? -> ["Oman", "negotiate with", "?"]

Temporal expressions such as on / in / before / after + date/month/year are only filtering conditions. They do not change the question type.

B. Time-answer question:
If the question asks for time, such as when / what year / what month / what day / at what time / during which period / from when to when, set search_triple = null.
The known event being asked about should be placed in anchor_triples.

C. Temporal-computation question:
If the question asks for chronological rank, duration comparison, longer/shorter, total duration, average duration, combined time periods, union, or intersection of periods, set search_triple = null.
All known events involved in the computation should be placed in anchor_triples.

II. When to produce anchor_triples

anchor_triples should contain only known complete events.

Include an anchor triple when:
1. The complete event is directly expressed in the question.
2. A reference event can be minimally recovered from a parallel structure in the same sentence.

Minimal recovery is allowed only when:
- the reference item is introduced by before / after / same day / same month / same year or similar structures;
- the reference item is not a pure time expression;
- the missing subject, relation, or object can be directly copied from the main event template;
- no new entity, relation, modifier, or commonsense inference is needed.

Do not create an anchor if before / after / on / in is followed only by a pure time expression, such as before 2005, after 19 April 2008, or in August 2008.

III. When to produce search_triple

If the question is an unknown-entity question:
- construct exactly one search_triple;
- replace the unknown subject or object with "?";
- "?" may appear only in subject or object;
- relation must be a normal relation phrase and must not contain "?".
If the question is a time-answer or temporal-computation question:
- search_triple must be null.

IV. Passive sentence handling

Convert passive constructions into active semantic direction.
Examples:
- Who was accused by Zawahiri? -> ["Zawahiri", "accuse", "?"]
- Which country was negotiated with by China? -> ["China", "negotiate with", "?"]
Do not reverse the event direction incorrectly because of passive voice.

V. Extraction rules

1. Do not regenerate the question or answer it.
2. Do not supplement facts using commonsense, background knowledge, or expected answers.
3. Preserve entity names and relation phrases from the question as much as possible.
4. Avoid unnecessary semantic rewriting.
5. Do not automatically turn ordinary noun phrases, background descriptions, locative adjuncts, or causal adjuncts into events.
6. If it is uncertain whether a reference item can be recovered as an event, prefer not extracting it.
7. Across the entire output, at most one "?" may appear, and it may appear only in search_triple.
8. Pure temporal expressions must never independently become anchor_triples.
"""


def run_extraction(source_question: str) -> ExtractionSchema:

    llm_executor = LLMTaskExecutor()
    rag_config = RAGConfig()

    dataset_name = rag_config.DATASET_NAME
    FEWSHOTS = (
        MULTI_EXTR_FEW_SHOTS
        if dataset_name == TemporalDatasets.MULTITQ
        else CORN_EXTR_FEW_SHOTS
    )

    input_str = f"source_question: {source_question}"

    return llm_executor.execute_structured_task(
        system_prompt=SYSTEM_PROMPT,
        few_shots=FEWSHOTS,
        current_input_data=input_str,
        response_model=ExtractionSchema,
    )
