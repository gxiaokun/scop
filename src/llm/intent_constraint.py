from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Tuple, Any
import json
from openai.types.chat import ChatCompletionMessageParam

from src.config.llm_config import LLMConfig
from src.llm.fewshots import TEMPORAL_CONSTRAINT_FEW_SHOTS
from src.utils.llm_utils import LLMTaskExecutor

Triple = Tuple[str, str, str]


class TemporalConstraint(BaseModel):
    
    anchor_kind: Literal["event", "explicit_time", "explicit_interval"] = Field(
        description=(
            "Type of temporal anchor. event means relying on an event in anchor_triples; "
            "explicit_time means relying on an explicit time expression; "
            "explicit_interval means relying on an explicit time interval."
        )
    )

    relation: Literal["before", "after", "equal", "inside", "overlap"] = Field(
        description=(
            "Temporal relationship of search_triple relative to the anchor. "
            "before/after indicate temporal order; equal indicates equality at a specified granularity; "
            "inside indicates full containment in an interval; overlap indicates interval overlap."
        )
    )

    anchor_index: Optional[int] = Field(
        default=None,
        description="Index of the used anchor_triples entry when anchor_kind='event'; null otherwise."
    )

    time_text: Optional[str] = Field(
        default=None,
        description="Explicit time expression from the question when anchor_kind='explicit_time'; null otherwise."
    )

    interval_start: Optional[str] = Field(
        default=None,
        description="Start of the explicit interval when anchor_kind='explicit_interval'; null otherwise."
    )

    interval_end: Optional[str] = Field(
        default=None,
        description="End of the explicit interval when anchor_kind='explicit_interval'; null otherwise."
    )

    granularity: Optional[Literal["day", "month", "year"]] = Field(
        default=None,
        description=(
            "Required when relation='equal'. "
            "Use 'day' for same day/on DATE, 'month' for same month/in MONTH YEAR, "
            "and 'year' for same year/in YEAR. "
            "Must be null for before/after/inside/overlap."
        ),
    )

    offset_days: Optional[int] = Field(
        default=None,
        description=(
            "Positive day offset used only when the question explicitly contains N days before/after; "
            "null otherwise."
        )
    )


class Ranking(BaseModel):
    rank: Literal["asc", "desc"] = Field(
        description=(
            "Temporal ranking direction for search_triple candidates. "
            "asc means earliest to latest; desc means latest to earliest."
        )
    )

    rank_k: int = Field(
        description=(
            "1-based ranking position after temporal sorting. "
            "Normally 1 in the current task. "
            "first/earliest -> 1 with rank='asc'; "
            "last/latest -> 1 with rank='desc'. "
            "Use a larger integer only if the question explicitly asks for another ordinal."
        )
    )


class TemporalConstraintSchema(BaseModel):
    constraints: List[TemporalConstraint] = Field(
        default_factory=list,
        description=(
            "List of temporal constraints imposed on search_triple, "
            "output in the order they appear in the question."
        ),
    )

    ranking: Optional[Ranking] = Field(
        default=None,
        description=(
            "Temporal ranking requirement imposed on search_triple candidates. "
            "Use null if there is no ordinal ranking requirement. "
            "Use {'rank': 'asc', 'rank_k': k} for the kth earliest candidate. "
            "Use {'rank': 'desc', 'rank_k': k} for the kth latest candidate. "
            "In the current data, k is usually 1 for first/earliest or last/latest."
        ),
    )


TEMPORAL_CONSTRAINT_SYSTEM_PROMPT = """
You are a temporal constraint parsing system.

Input:
1. question: the original question.
2. decomposed_triples:
   - anchor_triples: already decomposed anchor triples.
   - search_triple: the target triple to be retrieved or filtered.

Task:
Determine the temporal constraints imposed on search_triple.
Do not re-extract, rewrite, or modify any triples.
Do not answer the question.
Only parse constraints and ranking requirements that apply to search_triple.

Output:
1. constraints:
   A list of temporal constraints imposed on search_triple.
   Output all constraints in the order they appear in the question.

2. ranking:
   A ranking object if search_triple itself has an ordinal temporal ranking requirement.
   Otherwise output null.

Temporal relations:
- before:
  search_triple occurs before the anchor.
  Use for "before X", "earlier than X", and "N days before X".

- after:
  search_triple occurs after the anchor.
  Use for "after X", "later than X", and "N days after X".

- equal:
  search_triple matches the anchor at a specific temporal granularity.
  Use for "on DATE", "in MONTH YEAR", "in YEAR", "same day", "same month", or "same year".
  Every equal constraint must include a non-null granularity:
  "day" for same day/on DATE;
  "month" for same month/in MONTH YEAR;
  "year" for same year/in YEAR.

- inside:
  search_triple falls completely inside an interval.
  Use for "during", "during the time that", "from A to B", and "between A and B".

- overlap:
  search_triple overlaps with an interval, but complete containment is not required.
  Use for "while", "at the same time as", and simultaneous interval cases.

Anchor selection:
- Use anchor_kind = "event" when the constraint depends on one of the anchor_triples.
  Fill the correct 0-based anchor_index.

- Use anchor_kind = "explicit_time" when the constraint depends on an explicit time expression in the question.
  Fill time_text. Use granularity only when the relation is equal.

- Use anchor_kind = "explicit_interval" when the constraint depends on an explicit interval in the question.
  Fill interval_start and interval_end.

Granularity rule:
- granularity is required when relation = "equal".
- Never output relation = "equal" with granularity = null.
- Use "day" for same day / on DATE / on a specific day.
- Use "month" for same month / in MONTH YEAR / in a specific month.
- Use "year" for same year / in YEAR / in a specific year.
- For relation = "before", "after", "inside", or "overlap", granularity must be null.

Offset rule:
- Fill offset_days only when the question explicitly contains a fixed offset such as "7305 days after" or "27758 days before".
- offset_days is always a positive integer.
- The direction is determined by relation, not by the sign of offset_days.

Ranking rule:
- ranking is not a temporal constraint. It describes ordering over search_triple candidates.
- Set ranking only when the ordinal expression modifies the target search_triple candidates.
- Do not set ranking when the ordinal word or number is part of an entity name, award name, class name, event name, parliament name, or anchor description.
- Use {"rank": "asc", "rank_k": 1} for first / earliest.
- Use {"rank": "desc", "rank_k": 1} for last / latest.
- If the question explicitly asks for another ordinal, such as second, third, or fifth, keep the same rule:
  {"rank": "asc", "rank_k": N} means the N-th earliest;
  {"rank": "desc", "rank_k": N} means the N-th latest.
- If there is no ordinal ranking requirement on search_triple, ranking must be null.

Judgment rules:
1. Identify constraints only for search_triple.
2. Do not create constraints for temporal expressions that do not restrict search_triple.
3. If multiple temporal conditions apply to search_triple, output all of them.
4. If a constraint depends on anchor_triples, choose the correct anchor_index.
5. If the question contains both event anchors and explicit time anchors, output both.
6. If the question contains no temporal constraint, constraints must be [].
7. Prefer equal for same day/month/year and explicit date/month/year constraints, and always fill the correct granularity.
8. Prefer inside for during/from/between interval constraints.
9. Prefer overlap for while/at-the-same-time simultaneous interval constraints.
10. Prioritize correctness of relation, anchor_kind, anchor_index, granularity, offset_days, and ranking format.
"""


def run_temporal_constraint_parse(
    question: str,
    decomposed_triples: dict,
    model_role: str = "gpt",
) -> TemporalConstraintSchema:

    llm_executor = LLMTaskExecutor()
    
    anchor_triples = decomposed_triples.get("anchor_triples", [])
    search_triple = decomposed_triples.get("search_triple", None)

    input_data = {
        "question": question,
        "decomposed_triples": {
            "anchor_triples": anchor_triples,
            "search_triple": search_triple,
        },
    }

    return llm_executor.execute_structured_task(
        system_prompt=TEMPORAL_CONSTRAINT_SYSTEM_PROMPT,
        few_shots=TEMPORAL_CONSTRAINT_FEW_SHOTS,
        current_input_data=input_data,
        response_model=TemporalConstraintSchema
    )
