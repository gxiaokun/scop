import numpy as np
from typing import Dict, Any, List, Optional
import json
from typing import Any, List, Tuple, Type, TypeVar
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from instructor.exceptions import InstructorRetryException

import openai

from src.config.base_config import logger
from src.config.llm_config import LLMConfig


# =============== Embedding ===============
def embed_fn(
    texts: List[str],
    *,
    convert_to_numpy: bool = True,
    normalize: bool = False,
    show_progress_bar=False,
) -> np.ndarray:

    global_llm_config = LLMConfig()
    embedder = global_llm_config.EMBED_MODEL
    assert global_llm_config.EMBED_DIM is not None, "EMBED_DIM CAN NOT BE NONE"
    dim: int = int(global_llm_config.EMBED_DIM)

    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    embs = embedder.encode(
        texts,
        batch_size=global_llm_config.EMBED_BATCH_SIZE,
        convert_to_numpy=convert_to_numpy,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(embs, dtype=np.float32)


T = TypeVar("T", bound=BaseModel)


class LLMTaskExecutor:

    def __init__(self):
        self.config = LLMConfig()

    def execute_structured_task(
        self,
        system_prompt: str,
        few_shots: List[Tuple[Any, Any]],
        current_input_data: dict | str,
        response_model: Type[T],
    ) -> T:

        client = self.config.get_client()
        model_name = self.config.get_model_name()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

        for sample_input, sample_output in few_shots:

            in_content = (
                sample_input
                if isinstance(sample_input, str)
                else json.dumps(sample_input, ensure_ascii=False)
            )
            out_content = json.dumps(sample_output, ensure_ascii=False)

            messages.append({"role": "user", "content": in_content})
            messages.append({"role": "assistant", "content": out_content})

        if isinstance(current_input_data, str):
            current_input_str = current_input_data
        else:
            current_input_str = json.dumps(
                current_input_data, ensure_ascii=False, indent=2
            )

        messages.append({"role": "user", "content": current_input_str})

        try:
            response = client.chat.completions.create(
                model=model_name,
                response_model=response_model,
                messages=messages,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                max_retries=self.config.MAX_RETRIES,
            )
            return response
        except InstructorRetryException as e:

            original_exception = e.args[0] if e.args else None

            if isinstance(original_exception, openai.APITimeoutError):
                raise original_exception

            raise e
