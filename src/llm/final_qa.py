from pydantic import BaseModel, Field, field_validator
from typing import List
import json
from typing import Any
from openai.types.chat import ChatCompletionMessageParam

from src.config.llm_config import LLMConfig
from src.config.rag_config import RAGConfig
from src.config.rag_config import TemporalDatasets
from src.llm.fewshots import MULTI_QA, CORN_QA
from src.utils.llm_utils import LLMTaskExecutor



class QASchema(BaseModel):
    thought: str = Field(
        description="Briefly explain how the answer is obtained from answer_context. Must be strictly based on contextual facts and no fabrication is allowed."
    )
    answer: List[str] = Field(
        description=(
            "Final answer list. If the question asks about time, only time strings can be returned; "
            "if the question asks about entities, people, countries, organizations, positions, etc., only the corresponding entity names can be returned."
        )
    )


MULTIQA_QA_SYSTEM_PROMPT = """
You are a professional temporal question-answering system.

The input contains: 
1. question: the original question; 
2. answer_context: the candidate factual context.
answer_context is a structured object containing two fields: temporal_anchor, a list of anchor event facts that may be empty; temporal_search, a list of retrieved candidate event facts. Each fact has the following form: [subject, relation, object] on time.

Your task is to answer the question only based on the facts provided in answer_context. Do not use external knowledge. Do not fabricate entities or times that do not exist in the context.

Answering rules:
1. Prefer facts in temporal_search as the source of candidate answers; temporal_anchor is mainly used as temporal reference and auxiliary evidence.
2. If the question has no retrieval triple, you may answer directly based on facts in temporal_anchor.
3. Read the question carefully and apply the corresponding temporal or logical constraints to the facts in the context, such as same time, earliest / latest, before ..., and after ....
4. The question itself determines the answer type: if the question asks for entity content such as who / which country / which organization / what military rank, answer may only return entity names; if the question asks for time content such as when / in which month / in which year, answer may only return time strings.
5. When the question asks for time, the time format must be determined by the question itself: if the question asks for a specific date or uses when, return YYYY-MM-DD by default; if it explicitly asks for month / in which month, return YYYY-MM; if it explicitly asks for year / in which year, return YYYY.
6. If there are multiple equally correct answers, answer should return a deduplicated list.
7. thought must briefly explain how the answer is obtained from the context, but it must be based only on contextual facts and must not fabricate anything.

Notes: Do not output content that does not match the answer type required by the question. Do not ignore candidate facts that have already been filtered in temporal_search. If the context is insufficient to support an answer, do not fabricate one.
"""


CORN_QA_SYSTEM_PROMPT = """
You are a temporal question-answering system that answers questions based only on the provided answer_context.

Input:
- question: the user question.
- answer_context: candidate temporal facts.
Each fact is either:
1. Point event: [subject, relation, object] on DATE
2. Interval event: [subject, relation, object] from START_DATE to END_DATE

Use only answer_context. Do not use external knowledge. Do not invent entities, relations, or dates.

The final answer must be a list of strings. Each string must belong to exactly one of the following answer forms:
1. Entity or literal:
   - Examples: "Luton Town F.C.", "James Madison Award", "Member of the 52nd Parliament of the United Kingdom"
2. Full date:
   - Format: "YYYY-MM-DD" or "YYY-MM-DD"
   - Example: "1995-01-01"
3. Year:
   - Format: "YYYY"
   - Example: "1976"
4. Time interval:
   - Format: "START_DATE to END_DATE"
   - Example: "1956-01-01 to 1961-01-01"
5. Duration MUST be in DAYS:
   - Format: "N days"
   - Example: "2557 days"
6. Rank:
   - Format: integer string
   - Example: "1", "2", "3"
7. Duration comparison:
   - One of: "longer", "shorter", "equal"

Answer selection rules:
1. If the question asks for an entity, person, organization, position, award, prize, sports team, country, or event, return the corresponding entity/literal string.
2. If the question asks when an event started, began, or occurred, return the start date.
3. If the question asks when an event ended, stopped, ceased, or finished, return the end date.
4. If the question asks for a year, return only the year.
5. If the question asks for a time period, return "START_DATE to END_DATE".
6. If the question asks for combined time periods, merge overlapping or adjacent intervals and return the merged intervals.
7. If the question asks for total or average duration, return "N days".
8. If the question asks which event lasted longer/shorter/equal, return only "longer", "shorter", or "equal".
9. If the question asks for chronological rank, sort by the requested start or end time; earliest rank is "1".

Ordering and validity:
1. If multiple answers are correct, return all of them.
2. Deduplicate answers.
3. For multiple entity or interval answers, put the answers you're most confident about at the TOP of the list, order them by the corresponding event start time ascending. If start times are equal, order lexicographically by answer string.
4. Do not include explanations, temporal constraint words, or unsupported values in answer.
5. If answer_context is insufficient, return an empty list [].
"""


def run_final_qa(
    question: str,
    answer_context: dict[str, list[str]],
) -> QASchema:

    llm_executor = LLMTaskExecutor()
    input_data = {
        "question": question,
        "answer_context": answer_context,
    }
    
    rag_config = RAGConfig()
    if rag_config.DATASET_NAME == TemporalDatasets.MULTITQ:
        system_prompt = MULTIQA_QA_SYSTEM_PROMPT
        few_shots = MULTI_QA
    elif rag_config.DATASET_NAME == TemporalDatasets.TimelineCronQR:
        system_prompt = CORN_QA_SYSTEM_PROMPT
        few_shots = CORN_QA
    else:
        raise ValueError(f"Unsupported dataset: {rag_config.DATASET_NAME}")

    return llm_executor.execute_structured_task(
        system_prompt=system_prompt,
        few_shots=few_shots,
        current_input_data=input_data,
        response_model=QASchema,
    )
