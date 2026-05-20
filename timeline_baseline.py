import json
import os
import time
import argparse
import sys
import json
import random
from collections import defaultdict
from typing import List, Dict, Any, Set, Literal
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Dict, Any, List, Set
from openai.types.chat import ChatCompletionMessageParam  # 引入严格类型

import instructor
from pydantic import BaseModel, Field

from src.utils.baseline_embed_build import VectorDatabase
from src.config.base_config import BaseConfig, logger
from src.config.llm_config import LLMConfig
from src.config.rag_config import RAGConfig, TemporalDatasets
from src.utils.llm_utils import LLMTaskExecutor
from src.utils.llm_utils import embed_fn

CORN_LEVELS = ["simple", "medium", "complex"]


class FilterResult(BaseModel):
    indices: List[int] = Field(
        description="List of exact integer indices of the most relevant contexts."
    )


class AnswerResult(BaseModel):
    answer_list: List[str] = Field(
        description=(
            "The extracted exact answers in a list. "
            "Return [] if the answer cannot be determined."
        )
    )


class AgentStepResult(BaseModel):
    action: Literal["Search", "Finish"] = Field(
        description="MUST be exactly 'Search' or 'Finish'."
    )
    search_query: str = Field(
        description=("Short keyword-style retrieval query if action is Search. "),
        default="",
    )
    answer_list: List[str] = Field(
        description=(
            "Exact answer strings only if action is Finish. "
            "Return [] if insufficient evidence."
        ),
        default_factory=list,
    )


class PseudoDocumentResult(BaseModel):
    pseudo_document: str = Field(
        description=(
            "A concise pseudo-document or hypothetical evidence passage for retrieval. "
            "Return only the generated text without explanations."
        )
    )


LLM_ONLY_SYSTEM_PROMPT = """
You are a temporal knowledge graph QA assistant.

Answer the question directly.

Rules:
1. Return answer_list as a list of exact answer strings.
2. If the answer cannot be determined, return [].
3. Do not provide explanations.
4. Do not output full sentences.
"""

COT_ONLY_SYSTEM_PROMPT = """
You are a temporal QA assistant.

Return only the final answer_list.
Rules:
1. First, internally identify the subject entity, relation, answer type, and temporal constraint.
2. Internally reason step by step before deciding the final answer.
3. Return answer_list as a list of exact answer strings.
4. If the answer cannot be determined, return [].
5. Do not output explanations, reasoning traces, full sentences, or uncertainty markers.
"""


ANSWER_SYSTEM_PROMPT = """
You are a question answering assistant.

Answer using only the provided Context.

Rules:
1. Answer using only the provided Context.
2. Only output an answer when the Context clearly supports it.
3. If unsure or insufficient evidence, return [].
4. Do not output anything else.
"""


FILTER_SYSTEM_PROMPT = """
You are a temporal evidence selector.

You will receive:
1. Original question: the final QA task.
2. Current retrieval query: the query used to retrieve candidate contexts for the current step.
3. Candidate contexts.

Your task:
Select the contexts that are most useful for answering the Original question.
The Current retrieval query indicates the current retrieval hop, but it must not override the Original question.

Ranking criteria:
1. Contexts satisfying the Original question's seed entity, target answer type, relation, and temporal constraint.
2. Contexts relevant to the Current retrieval query's entity, relation, bridge entity, or temporal clue.
3. Contexts matching temporal constraints:
   before, after, during, in, from, to, first, last, earliest, latest, start time, end time, interval.
4. Contexts that provide a bridge entity for multi-hop reasoning.
5. Contexts that disambiguate candidates with the same relation but different time.
6. Prefer diverse evidence covering necessary hops over duplicate sentences.

Hard negatives:
- same entity but wrong relation for the Original question
- same relation but wrong entity for the Original question
- correct current retrieval query but incompatible with the Original question's temporal constraint
- temporally ambiguous context when a more specific temporal context exists
- evidence that is only related to the Current retrieval query but useless for answering the Original question

Return exactly the integer indices of selected contexts, ordered from most useful to least useful.
Do not explain.
"""


REACT_TKGQA_PROMPT = """
You are a question-answering agent using ReAct.

You can use Search to retrieve evidence from a temporal knowledge graph text database.
At each step, choose either Search or Finish.

General behavior:
- Use the Question and retrieved Observations to decide the next action.
- If the current Observations are insufficient, issue another Search query.
- Finish when the retrieved Observations appear sufficient to answer.
- Before finishing, prefer answers that are directly supported by Observations matching the requested relation and any explicit time expression in the Question.

Rules:
1. Use only retrieved Observations.
2. The first valid action must be Search.
3. Search queries should be short keyword queries.
4. Do not simply repeat the full question as the search query.
5. Search queries should contain useful keywords related to the question.
6. Return answer_list as exact answer strings.
7. Return [] if the answer is not supported.
8. Do not explain.

Output format:
Return only one valid JSON object with exactly these keys:
{"action": "Search" or "Finish", "search_query": "...", "answer_list": [...]}

Do not write labels such as Action:, Search_Query:, or Answer_List:.
"""


IRCOT_TKGQA_PROMPT = """
You are a question-answering agent using Interleaved Retrieval Chain-of-Thought.

You solve the question through an iterative process in which retrieval and internal reasoning alternate.
The agent may retrieve evidence, inspect the retrieved Observations, consider whether more information could be useful, and then either retrieve again or finish.

At every step, choose exactly one action:
- Search: issue a short retrieval query to obtain additional Observations.
- Finish: return the final answer_list based only on the retrieved Observations.

General behavior:
- Begin with a Search action before attempting to answer.
- Use retrieved Observations to form an internal sense of what information is available.
- When the available Observations do not appear sufficient, issue another Search query.
- New Observations may influence the wording of later Search queries.
- For questions that appear to require more than one piece of information, multiple Search steps may be used.
- Finish only when the currently retrieved Observations appear adequate to produce an answer.
- Search queries should remain short and relevant to the Question or to previously retrieved Observations.

Rules:
1. Use only retrieved Observations when producing the final answer.
2. The first valid action must be Search.
3. Search queries should be short keyword-style queries.
4. Do not simply repeat the full Question as the search query.
5. Search for entities, relations, candidate terms, or other clues that may be relevant.
6. Do not provide explanations or explicit reasoning traces.
7. Return answer_list as exact answer strings when finishing.
8. Return [] if the answer is not supported by the retrieved Observations.
9. Keep search_query empty when action is Finish.
10. Keep answer_list empty when action is Search.

Output format:
Return only one valid JSON object with exactly these keys:
{"action": "Search" or "Finish", "search_query": "...", "answer_list": [...]}

The output must contain no extra labels, sections, commentary, or prose.
Do not write labels such as Action:, Search_Query:, or Answer_List:.
"""


HYDE_DOCUMENT_SYSTEM_PROMPT = """
You generate hypothetical evidence passages for temporal knowledge graph question answering retrieval.

Given a question, write one concise evidence-like passage that could plausibly help retrieve the relevant temporal KG facts.

Rules:
1. Preserve the question's core entities, relation cues, answer focus, and temporal expressions.
2. Prefer compact factual phrasing similar to an evidence snippet, not a reasoning trace.
3. Do not output bullet lists, explanations, uncertainty markers, or answer_list.
4. Keep the passage concise, preferably under 80 words.
"""

QUERY2DOC_SYSTEM_PROMPT = """
You generate pseudo-documents for query expansion in temporal knowledge graph question answering retrieval.

Given a question, write one concise pseudo-document that expands the query with likely relevant entities, relation cues, temporal clues, and answer focus.

Rules:
1. Preserve the original question intent and temporal constraints.
2. Produce natural evidence-like text that can improve retrieval.
3. Do not output bullet lists, explanations, uncertainty markers, or answer_list.
4. Keep the pseudo-document concise, preferably under 80 words.
"""


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    return "".join(ch for ch in s if not ch.isspace())


def to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(item) for item in x if item is not None]
    return [str(x)]


def _is_no_answer(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, list) and len(x) == 0:
        return True
    if isinstance(x, str) and normalize_text(x) in ("", "no answer", "n/a", "none"):
        return True
    if (
        isinstance(x, list)
        and len(x) == 1
        and normalize_text(x[0]) in ("", "no answer", "n/a", "none")
    ):
        return True
    return False


def is_hit(gold_answer: Any, llm_answer: Any) -> bool:
    if _is_no_answer(gold_answer) and _is_no_answer(llm_answer):
        return True

    gold_list = [
        normalize_text(s) for s in to_str_list(gold_answer) if normalize_text(s)
    ]
    pred_list = [
        normalize_text(s) for s in to_str_list(llm_answer) if normalize_text(s)
    ]

    if not gold_list or not pred_list:
        return False

    top1_pred = pred_list[0]
    gold_set = set(gold_list)

    return top1_pred in gold_set


def call_llm_no_context(query: str) -> List[str]:

    executor = LLMTaskExecutor()

    current_input = f"Question:\n{query}\n\nReturn answer_list only."

    try:

        resp: AnswerResult = executor.execute_structured_task(
            system_prompt=LLM_ONLY_SYSTEM_PROMPT,
            few_shots=[],
            current_input_data=current_input,
            response_model=AnswerResult,
        )
        return resp.answer_list

    except Exception as e:

        logger.error(f"LLM-only failed: {e}")
        return [f"LLM_ERROR: {str(e)[:20]}"]


def call_llm_cot_no_context(query: str) -> List[str]:

    executor = LLMTaskExecutor()
    current_input = (
        f"Question:\n{query}\n\nReason internally and return answer_list only."
    )

    try:
        resp: AnswerResult = executor.execute_structured_task(
            system_prompt=COT_ONLY_SYSTEM_PROMPT,
            few_shots=[],
            current_input_data=current_input,
            response_model=AnswerResult,
        )
        return resp.answer_list

    except Exception as e:
        logger.error(f"CoT-only failed: {e}")
        return [f"LLM_ERROR: {str(e)[:20]}"]


def generate_pseudo_document(
    question: str,
    system_prompt: str,
    method_name: str,
) -> str:

    executor = LLMTaskExecutor()
    current_input = (
        f"Question:\n{question}\n\n"
        "Generate exactly one concise retrieval-oriented passage."
    )

    try:
        resp: PseudoDocumentResult = executor.execute_structured_task(
            system_prompt=system_prompt,
            few_shots=[],
            current_input_data=current_input,
            response_model=PseudoDocumentResult,
        )
        pseudo_doc = " ".join(str(resp.pseudo_document).split()).strip()
        return pseudo_doc

    except Exception as e:
        logger.warning(
            f"{method_name} pseudo-document generation failed, fallback to original question. Error: {e}"
        )
        return ""


def call_llm_for_filter(
    original_question: str,
    retrieval_query: str,
    contexts: List[str],
    final_k: int = 10,
) -> List[str]:

    if len(contexts) <= final_k:
        return contexts

    indexed_contexts = "\n".join([f"[{i}] {c}" for i, c in enumerate(contexts)])

    user_prompt = f"""
Original question:
{original_question}

Current retrieval query:
{retrieval_query}

Candidate contexts:
{indexed_contexts}

Select exactly {final_k} indices.

Selection objective:
Choose contexts that are useful for answering the Original question.
The Current retrieval query indicates the current retrieval hop, but it must not override the Original question's entity, relation, answer type, or temporal constraints.
"""

    try:

        executor = LLMTaskExecutor()
        resp: FilterResult = executor.execute_structured_task(
            system_prompt=FILTER_SYSTEM_PROMPT,
            few_shots=[],
            current_input_data=user_prompt,
            response_model=FilterResult,
        )

        valid_indices = [
            idx
            for idx in resp.indices
            if isinstance(idx, int) and 0 <= idx < len(contexts)
        ]

        seen = set()
        unique_indices = [
            idx for idx in valid_indices if not (idx in seen or seen.add(idx))
        ]

        filtered_contexts = [contexts[i] for i in unique_indices[:final_k]]

        if len(filtered_contexts) < final_k:
            used_indices = set(unique_indices)
            for i in range(len(contexts)):
                if i not in used_indices:
                    filtered_contexts.append(contexts[i])
                    if len(filtered_contexts) == final_k:
                        break

        return filtered_contexts

    except Exception as e:
        logger.warning(
            f"LLM filter failed, falling back to original top {final_k} contexts. Error: {e}"
        )
        return contexts[:final_k]


def call_llm_for_answer(
    query: str,
    contexts: List[str],
) -> List[str]:

    context_text = (
        "\n".join([f"[E{i}] {c}" for i, c in enumerate(contexts)])
        if contexts
        else "None"
    )

    user_prompt = f"""
Context:
{context_text}

Question:
{query}

Return answer_list only.
"""

    try:

        executor = LLMTaskExecutor()
        resp: AnswerResult = executor.execute_structured_task(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            few_shots=[],
            current_input_data=user_prompt,
            response_model=AnswerResult,
        )

        return resp.answer_list

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        return [f"LLM_ERROR: {str(e)[:20]}"]


def retrieve_contexts(
    query: str,
    vector_db: VectorDatabase,
    k: int,
) -> List[str]:

    raw = vector_db.search(query, k=k)
    return [item["sentence"] for item in raw]


def retrieve_contexts_with_optional_filter(
    original_question: str,
    retrieval_query: str,
    vector_db: VectorDatabase,
    use_filter: bool,
    filter_retrieve_k: int,
    final_k: int,
) -> List[str]:

    if use_filter:
        broad_contexts = retrieve_contexts(
            query=retrieval_query,
            vector_db=vector_db,
            k=filter_retrieve_k,
        )

        return call_llm_for_filter(
            original_question=original_question,
            retrieval_query=retrieval_query,
            contexts=broad_contexts,
            final_k=final_k,
        )

    return retrieve_contexts(
        query=retrieval_query,
        vector_db=vector_db,
        k=final_k,
    )


def run_rag_answer(
    question: str,
    vector_db: VectorDatabase,
    final_k: int,
) -> List[str]:

    contexts = retrieve_contexts(
        query=question,
        vector_db=vector_db,
        k=final_k,
    )

    return call_llm_for_answer(
        query=question,
        contexts=contexts,
    )


def run_rag_filter_answer(
    question: str,
    vector_db: VectorDatabase,
    filter_retrieve_k: int,
    final_k: int,
) -> List[str]:

    broad_contexts = retrieve_contexts(
        query=question,
        vector_db=vector_db,
        k=filter_retrieve_k,
    )

    contexts = call_llm_for_filter(
        original_question=question,
        retrieval_query=question,
        contexts=broad_contexts,
        final_k=final_k,
    )

    return call_llm_for_answer(
        query=question,
        contexts=contexts,
    )


def run_hyde_rag_answer(
    question: str,
    vector_db: VectorDatabase,
    final_k: int,
) -> Dict[str, Any]:

    pseudo_doc = generate_pseudo_document(
        question=question,
        system_prompt=HYDE_DOCUMENT_SYSTEM_PROMPT,
        method_name="HyDE-RAG",
    )

    retrieval_query = pseudo_doc if pseudo_doc else question
    contexts = retrieve_contexts(
        query=retrieval_query,
        vector_db=vector_db,
        k=final_k,
    )

    answer_list = call_llm_for_answer(
        query=question,
        contexts=contexts,
    )

    return {
        "answer_list": answer_list,
        "rag_debug": {
            "retrieval_variant": "hyde_rag",
            "pseudo_document": pseudo_doc,
            "retrieval_query": retrieval_query,
            "num_contexts": len(contexts),
        },
    }


def run_query2doc_rag_answer(
    question: str,
    vector_db: VectorDatabase,
    final_k: int,
) -> Dict[str, Any]:

    pseudo_doc = generate_pseudo_document(
        question=question,
        system_prompt=QUERY2DOC_SYSTEM_PROMPT,
        method_name="Query2doc-RAG",
    )

    retrieval_query = f"{question}\n{pseudo_doc}" if pseudo_doc else question

    contexts = retrieve_contexts(
        query=retrieval_query,
        vector_db=vector_db,
        k=final_k,
    )

    answer_list = call_llm_for_answer(
        query=question,
        contexts=contexts,
    )

    return {
        "answer_list": answer_list,
        "rag_debug": {
            "retrieval_variant": "query2doc_rag",
            "pseudo_document": pseudo_doc,
            "retrieval_query": retrieval_query,
            "num_contexts": len(contexts),
        },
    }


def run_hyde_rag_filter_answer(
    question: str,
    vector_db: VectorDatabase,
    filter_retrieve_k: int,
    final_k: int,
) -> Dict[str, Any]:

    pseudo_doc = generate_pseudo_document(
        question=question,
        system_prompt=HYDE_DOCUMENT_SYSTEM_PROMPT,
        method_name="HyDE-RAG+Filter",
    )

    retrieval_query = pseudo_doc if pseudo_doc else question

    broad_contexts = retrieve_contexts(
        query=retrieval_query,
        vector_db=vector_db,
        k=filter_retrieve_k,
    )

    contexts = call_llm_for_filter(
        original_question=question,
        retrieval_query=retrieval_query,
        contexts=broad_contexts,
        final_k=final_k,
    )

    answer_list = call_llm_for_answer(
        query=question,
        contexts=contexts,
    )

    return {
        "answer_list": answer_list,
        "rag_debug": {
            "retrieval_variant": "hyde_rag_filter",
            "pseudo_document": pseudo_doc,
            "retrieval_query": retrieval_query,
            "num_broad_contexts": len(broad_contexts),
            "num_contexts": len(contexts),
            "filter_retrieve_k": filter_retrieve_k,
            "final_k": final_k,
        },
    }


def run_query2doc_rag_filter_answer(
    question: str,
    vector_db: VectorDatabase,
    filter_retrieve_k: int,
    final_k: int,
) -> Dict[str, Any]:

    pseudo_doc = generate_pseudo_document(
        question=question,
        system_prompt=QUERY2DOC_SYSTEM_PROMPT,
        method_name="Query2doc-RAG+Filter",
    )

    retrieval_query = f"{question}\n{pseudo_doc}" if pseudo_doc else question

    broad_contexts = retrieve_contexts(
        query=retrieval_query,
        vector_db=vector_db,
        k=filter_retrieve_k,
    )

    contexts = call_llm_for_filter(
        original_question=question,
        retrieval_query=retrieval_query,
        contexts=broad_contexts,
        final_k=final_k,
    )

    answer_list = call_llm_for_answer(
        query=question,
        contexts=contexts,
    )

    return {
        "answer_list": answer_list,
        "rag_debug": {
            "retrieval_variant": "query2doc_rag_filter",
            "pseudo_document": pseudo_doc,
            "retrieval_query": retrieval_query,
            "num_broad_contexts": len(broad_contexts),
            "num_contexts": len(contexts),
            "filter_retrieve_k": filter_retrieve_k,
            "final_k": final_k,
        },
    }


def call_agentic_loop(
    query: str,
    vector_db: Any,
    mode: str,
    use_filter: bool,
    filter_retrieve_k: int,
    final_k: int,
    max_steps: int = 4,
) -> Dict[str, Any]:

    llm_config = LLMConfig()

    client = llm_config.get_client()

    if "react" in mode:
        system_prompt = REACT_TKGQA_PROMPT
    else:
        system_prompt = IRCOT_TKGQA_PROMPT

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                "Generate the first short Search query for retrieving relevant evidence. "
            ),
        },
    ]

    seen_queries: Set[str] = set()
    search_queries: List[str] = []
    num_searches = 0
    first_action = None
    forced_search_before_finish = 0

    agent_debug: Dict[str, Any] = {
        "agent_variant": "search_first_prompt_balanced_no_initial_observation",
        "mode": mode,
        "use_filter": use_filter,
        "final_k": final_k,
        "filter_retrieve_k": filter_retrieve_k,
        "max_steps": max_steps,
        "first_action": None,
        "num_searches": 0,
        "search_queries": [],
        "forced_search_before_finish": 0,
        "finalized_by": None,
    }

    for step in range(max_steps):
        try:

            resp: AgentStepResult = client.chat.completions.create(
                model=llm_config.get_model_name(),
                messages=messages,
                temperature=llm_config.TEMPERATURE or 0.0,
                response_model=AgentStepResult,
                max_tokens=llm_config.MAX_TOKENS,
                timeout=llm_config.TIMEOUT,
                # parallel_tool_calls=False,
            )

            if first_action is None:
                first_action = resp.action
                agent_debug["first_action"] = first_action

            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action": resp.action,
                            "search_query": resp.search_query,
                            "answer_list": resp.answer_list,
                        },
                        ensure_ascii=False,
                    ),
                }
            )

            if resp.action == "Finish" and num_searches == 0:
                forced_search_before_finish += 1
                agent_debug["forced_search_before_finish"] = forced_search_before_finish
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: Finish is not allowed before any Search. "
                            "Generate a short Search query first. Return JSON only."
                        ),
                    }
                )
                continue

            if resp.action == "Finish":
                agent_debug["num_searches"] = num_searches
                agent_debug["search_queries"] = search_queries
                agent_debug["finalized_by"] = "agent_finish"
                return {
                    "answer_list": resp.answer_list if resp.answer_list else [],
                    "agent_debug": agent_debug,
                }

            if resp.action == "Search":
                search_query = resp.search_query.strip()

                if not search_query:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Observation: Empty search_query. Generate a short keyword query. Return JSON only."
                            ),
                        }
                    )
                    continue

                search_query_key = " ".join(search_query.lower().split())

                if search_query_key in seen_queries:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Observation: Repeated search_query. Generate a different short query. Return JSON only."
                            ),
                        }
                    )
                    continue

                seen_queries.add(search_query_key)
                search_queries.append(search_query)
                num_searches += 1
                agent_debug["num_searches"] = num_searches
                agent_debug["search_queries"] = search_queries

                final_obs = retrieve_contexts_with_optional_filter(
                    original_question=query,
                    retrieval_query=search_query,
                    vector_db=vector_db,
                    use_filter=use_filter,
                    filter_retrieve_k=filter_retrieve_k,
                    final_k=final_k,
                )

                obs_text = (
                    "\n".join(
                        [f"[E{num_searches}-{i}] {c}" for i, c in enumerate(final_obs)]
                    )
                    if final_obs
                    else "No relevant information found."
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{query}\n\n"
                            f"Search query:\n{search_query}\n\n"
                            f"Observation:\n{obs_text}\n\n"
                            "Use the Observation to decide whether to Search again or Finish. Return JSON only."
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"Agent Loop  {step + 1} failed: {e}")
            agent_debug["num_searches"] = num_searches
            agent_debug["search_queries"] = search_queries
            agent_debug["finalized_by"] = "agent_loop_error"
            agent_debug["error"] = str(e)[:200]
            return {
                "answer_list": [f"LLM_ERROR: {str(e)[:20]}"],
                "agent_debug": agent_debug,
            }

    try:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "No more search steps. Use the retrieved Observations only. "
                    "Finish with answer_list if the evidence supports the answer; otherwise return []. Return JSON only."
                ),
            }
        )

        resp: AgentStepResult = client.chat.completions.create(
            model=llm_config.get_model_name(),
            messages=messages,
            temperature=llm_config.TEMPERATURE,
            response_model=AgentStepResult,
            max_tokens=256,
            timeout=llm_config.TIMEOUT,
        )

        agent_debug["num_searches"] = num_searches
        agent_debug["search_queries"] = search_queries
        agent_debug["finalized_by"] = "final_check"

        if resp.action == "Finish" and num_searches > 0:
            return {
                "answer_list": resp.answer_list if resp.answer_list else [],
                "agent_debug": agent_debug,
            }

        return {
            "answer_list": [],
            "agent_debug": agent_debug,
        }

    except Exception as e:
        logger.error(f"Agent final check failed: {e}")
        agent_debug["num_searches"] = num_searches
        agent_debug["search_queries"] = search_queries
        agent_debug["finalized_by"] = "final_check_error"
        agent_debug["error"] = str(e)[:200]
        return {
            "answer_list": [],
            "agent_debug": agent_debug,
        }


def process_single_item(
    data_point: Dict[str, Any],
    vector_db: VectorDatabase,
    mode: str,
    filter_retrieve_k: int,
    final_k: int,
    max_steps: int,
) -> Dict[str, Any]:

    q_id = data_point.get("id")
    question = data_point.get("question")
    gold_answer = data_point.get("answer")
    level = data_point.get("question_level", "unknown")
    agent_debug = None
    rag_debug = None

    if mode == "llm":
        llm_answer = call_llm_no_context(question)

    elif mode == "cot":
        llm_answer = call_llm_cot_no_context(question)

    elif mode == "rag":
        llm_answer = run_rag_answer(
            question=question,
            vector_db=vector_db,
            final_k=final_k,
        )

    elif mode == "rag_filter":
        llm_answer = run_rag_filter_answer(
            question=question,
            vector_db=vector_db,
            filter_retrieve_k=filter_retrieve_k,
            final_k=final_k,
        )

    elif mode == "hyde_rag":
        baseline_result = run_hyde_rag_answer(
            question=question,
            vector_db=vector_db,
            final_k=final_k,
        )
        llm_answer = baseline_result.get("answer_list", [])
        rag_debug = baseline_result.get("rag_debug")

    elif mode == "hyde_rag_filter":
        baseline_result = run_hyde_rag_filter_answer(
            question=question,
            vector_db=vector_db,
            filter_retrieve_k=filter_retrieve_k,
            final_k=final_k,
        )
        llm_answer = baseline_result.get("answer_list", [])
        rag_debug = baseline_result.get("rag_debug")

    elif mode == "query2doc_rag":
        baseline_result = run_query2doc_rag_answer(
            question=question,
            vector_db=vector_db,
            final_k=final_k,
        )
        llm_answer = baseline_result.get("answer_list", [])
        rag_debug = baseline_result.get("rag_debug")

    elif mode == "query2doc_rag_filter":
        baseline_result = run_query2doc_rag_filter_answer(
            question=question,
            vector_db=vector_db,
            filter_retrieve_k=filter_retrieve_k,
            final_k=final_k,
        )
        llm_answer = baseline_result.get("answer_list", [])
        rag_debug = baseline_result.get("rag_debug")

    elif mode in ["react", "react_filter", "ircot", "ircot_filter"]:
        use_filter = mode.endswith("_filter")

        agent_result = call_agentic_loop(
            query=question,
            vector_db=vector_db,
            mode=mode,
            use_filter=use_filter,
            filter_retrieve_k=filter_retrieve_k,
            final_k=final_k,
            max_steps=max_steps,
        )
        llm_answer = agent_result.get("answer_list", [])
        agent_debug = agent_result.get("agent_debug")

    else:
        llm_answer = []

    result = {
        "id": q_id,
        "question": question,
        "answer": gold_answer,
        "question_level": level,
        "llm_answer": llm_answer,
    }

    if agent_debug is not None:
        result["agent_debug"] = agent_debug

    if rag_debug is not None:
        result["rag_debug"] = rag_debug

    return result


def evaluate_json_results(json_path: str) -> None:
    if not os.path.exists(json_path):
        print("no result file found, skip evaluation.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            print("error parsing result file, skip evaluation.")
            return

    overall_total = 0
    overall_hit = 0

    level_total = {lv: 0 for lv in CORN_LEVELS}
    level_hit = {lv: 0 for lv in CORN_LEVELS}

    for rec in records:
        gold = rec.get("answer")
        pred = rec.get("llm_answer")
        level = str(rec.get("question_level", "unknown")).lower()

        if gold is None or pred is None:
            continue

        correct = is_hit(gold, pred)

        overall_total += 1
        if correct:
            overall_hit += 1

        if level in CORN_LEVELS:
            level_total[level] += 1
            if correct:
                level_hit[level] += 1

    if overall_total == 0:
        print("no valid records to evaluate.")
        return

    def safe_ratio(hit: int, total: int) -> float:
        return hit / total if total > 0 else 0.0

    print(f"\n===== final evaluation metrics: {json_path} =====")
    print(f"Sample count: {overall_total}")
    print(f"Overall: {safe_ratio(overall_hit, overall_total):.4f}")

    for lv in CORN_LEVELS:
        print(
            f"{lv.capitalize():7s}: "
            f"{safe_ratio(level_hit[lv], level_total[lv]):.4f} "
            f"({level_hit[lv]}/{level_total[lv]})"
        )


def main():
    parser = argparse.ArgumentParser(description="TimelineCronQR Baseline Evaluation")

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "rag",
            "rag_filter",
            "hyde_rag",
            "hyde_rag_filter",
            "query2doc_rag",
            "query2doc_rag_filter",
            "react",
            "react_filter",
            "ircot",
            "ircot_filter",
        ],
        required=True,
        help="choice",
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="recalc for result file",
    )

    parser.add_argument(
        "--final_k",
        type=int,
        default=10,
        help=(
            "The final number of contexts fed into the LLM or each Agent Observation. "
            "In non-filter mode, vector retrieval directly fetches final_k items; "
            "In filter mode, it first retrieves filter_retrieve_k items, and then filters them down to final_k items."
        ),
    )

    parser.add_argument(
        "--filter_retrieve_k",
        type=int,
        default=50,
        help=(
            "Used only for rag_filter / hyde_rag_filter / query2doc_rag_filter / react_filter / ircot_filter. "
            "Indicates the number of rough retrievals before LLM filtering."
        ),
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum number of iteration steps for ReAct / IRCoT.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of concurrent threads.",
    )

    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Ignore existing result files and run again.",
    )

    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help=(
            "Run only the first N samples in the test split, used for debugging. "
            "If specified, results will be written to the RUNS_DIR/testsize_N/ subdirectory. "
            "If not specified, the full test split will be run, and the original output directory will be used."
        ),
    )

    args = parser.parse_args()
    BaseConfig(log_level="INFO", max_workers=args.max_workers)
    RAGConfig(dataset_name=TemporalDatasets.TimelineCronQR.value)
    LLMConfig()

    if args.test_size is not None and args.test_size <= 0:
        print(
            "test_size must be a positive integer; if you want to run the full test split, please do not pass --test_size."
        )
        return

    RUNS_DIR = os.getenv("EXPERIMENT_OUTPUT_DIR") or ""

    output_dir = os.path.join(RUNS_DIR, "corn_baseline")
    if args.test_size is not None:
        output_dir = os.path.join(
            RUNS_DIR, "corn_baseline", f"testsize_{args.test_size}"
        )

    os.makedirs(output_dir, exist_ok=True)

    agent_variant_suffix = (
        "_searchfirst_balanced"
        if args.mode in ["react", "react_filter", "ircot", "ircot_filter"]
        else ""
    )

    output_filename = (
        f"results_baseline_{args.mode}"
        f"{agent_variant_suffix}"
        f"_final{args.final_k}"
        f"_filterret{args.filter_retrieve_k}"
        f"_step{args.max_steps}.json"
    )
    output_file = os.path.join(output_dir, output_filename)

    if args.eval_only:
        print(f"\n[Eval Only] Evaluating: {output_file}")
        evaluate_json_results(output_file)
        return

    completed_records: List[Dict[str, Any]] = []
    completed_ids: Set[Any] = set()

    if os.path.exists(output_file) and not args.force_rerun:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                completed_records = json.load(f)

            completed_ids = {rec["id"] for rec in completed_records if "id" in rec}

            print(
                f"Historical record file found, {len(completed_ids)} tasks completed."
            )

        except Exception:
            backup_file = output_file + f".bak_{int(time.time())}"
            os.rename(output_file, backup_file)
            print(f"Historical result file is corrupted, backed up as: {backup_file}")
            completed_records = []
            completed_ids = set()

    elif os.path.exists(output_file) and args.force_rerun:
        backup_file = output_file + f".bak_{int(time.time())}"
        os.rename(output_file, backup_file)
        print(f"force_rerun=True, old file has been backed up as: {backup_file}")

    test_data_file = RAGConfig().get_dataset_test_file_by_name(
        TemporalDatasets.TimelineCronQR.value
    )
    with open(test_data_file, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    test_dataset = [dp for dp in full_dataset if dp.get("split") == "test"]

    full_test_size = len(test_dataset)

    if args.test_size is not None:
        if args.test_size < full_test_size:
            target_size = args.test_size

            level_groups = defaultdict(list)
            for dp in test_dataset:
                lvl = str(dp.get("question_level", "unknown")).lower()
                level_groups[lvl].append(dp)

            sampled_dataset = []
            remaining_to_sample = target_size

            random.seed(42)

            allocations = {}
            for lvl, group in level_groups.items():
                alloc_size = int((len(group) / full_test_size) * target_size)
                alloc_size = min(alloc_size, len(group))
                allocations[lvl] = alloc_size

                sampled_dataset.extend(random.sample(group, alloc_size))
                remaining_to_sample -= alloc_size

            if remaining_to_sample > 0:
                remaining_pool = [
                    dp for dp in test_dataset if dp not in sampled_dataset
                ]
                sampled_dataset.extend(
                    random.sample(
                        remaining_pool, min(remaining_to_sample, len(remaining_pool))
                    )
                )

            random.shuffle(sampled_dataset)
            test_dataset = sampled_dataset

            print(
                f"[Debug Subset] Stratified sampled test_size={target_size} | "
                f"Base Allocations: {allocations} | "
                f"output_dir={output_dir}"
            )
        else:
            print(
                f"[Debug Subset] test_size={args.test_size} is greater than or equal to full_test_size ({full_test_size}). "
                f"Using full test dataset. | output_dir={output_dir}"
            )

    pending_tasks = [dp for dp in test_dataset if dp.get("id") not in completed_ids]

    total_pending = len(pending_tasks)

    if total_pending == 0:
        print(
            "\nAll data has been processed, calculating metrics directly, skipping embedding loading."
        )
        evaluate_json_results(output_file)
        return

    print(
        f"\nStart execution mode: {args.mode} | "
        f"Agent-Variant: {agent_variant_suffix or 'original'} | "
        f"Test-Size: {args.test_size if args.test_size is not None else 'full'} | "
        f"Output-Dir: {output_dir} | "
        f"Final-K: {args.final_k} | "
        f"Filter-Retrieve-K: {args.filter_retrieve_k} | "
        f"Max Steps: {args.max_steps} | "
        f"Pending: {total_pending} | "
        f"Concurrency: {args.max_workers}"
    )

    vector_db = None
    REAL_FILE_PATH = RAGConfig().get_dataset_kg_by_name(
        TemporalDatasets.TimelineCronQR.value
    )
    if args.mode not in ["llm", "cot"]:
        vector_db = VectorDatabase()
        vector_db.build_from_txt(REAL_FILE_PATH, embed_fn)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_single_item,
                dp,
                vector_db,
                args.mode,
                args.filter_retrieve_k,
                args.final_k,
                args.max_steps,
            ): dp
            for dp in pending_tasks
        }

        done_count = 0
        failed_count = 0
        start_time = time.time()

        progress_bar = tqdm(
            as_completed(future_to_task),
            total=total_pending,
            desc=f"Processing-{args.mode}",
            dynamic_ncols=True,
            mininterval=5,
            file=sys.stdout,
        )

        for future in progress_bar:
            task_data = future_to_task[future]

            try:
                result_record = future.result()
                completed_records.append(result_record)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(
                        completed_records,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )

            except Exception as e:
                failed_count += 1
                logger.error(f"Task [{task_data.get('id')}] failed: {e}")

            done_count += 1

            if done_count % 10 == 0 or done_count == total_pending:
                elapsed = time.time() - start_time
                speed = done_count / elapsed if elapsed > 0 else 0.0
                remaining = total_pending - done_count
                eta_min = (remaining / speed / 60) if speed > 0 else -1

                print(
                    f"[PROGRESS] mode={args.mode} "
                    f"done={done_count}/{total_pending} "
                    f"failed={failed_count} "
                    f"completed_total={len(completed_records)} "
                    f"speed={speed:.3f} samples/s "
                    f"eta={eta_min:.1f} min",
                    flush=True,
                )

    evaluate_json_results(output_file)


if __name__ == "__main__":
    main()
