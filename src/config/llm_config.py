import os
import instructor
from openai import OpenAI
from typing import Optional
from threading import Lock
from sentence_transformers import SentenceTransformer

# 假设 BaseConfig 和 logger 已经导入
from src.config.base_config import BaseConfig, logger


class LLMConfig(BaseConfig):


    _instance: Optional["LLMConfig"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):

        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMConfig, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        # Chat 模型全局配置
        chat_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        # 共享与基础配置
        embed_model_path: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        embed_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        if getattr(self, "_initialized", False):
            return

        with self.__class__._lock:
            if getattr(self, "_initialized", False):
                return

            super().__init__(**kwargs)

            # ---- 核心改造：统一单模型参数解析 ----
            self.CHAT_MODEL = chat_model or os.getenv("CHAT_MODEL")
            self.BASE_URL = base_url or os.getenv("OPENAI_BASE_URL")
            _api_key = api_key or os.getenv("OPENAI_API_KEY")

            if not _api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required but not set in environment variables or constructor parameters."
                )
            self.API_KEY = _api_key

            self.TEMPERATURE = (
                temperature
                if temperature is not None
                else float(os.getenv("TEMPERATURE", "0.0"))
            )
            self.TIMEOUT = (
                timeout if timeout is not None else int(os.getenv("TIMEOUT", "20"))
            )
            self.MAX_TOKENS = (
                max_tokens
                if max_tokens is not None
                else int(os.getenv("MAX_TOKENS", "8192"))
            )
            self.DEVICE = device or os.getenv("CUDA_DEVICE", "cuda:0")
            self.MAX_RETRIES = (
                max_retries
                if max_retries is not None
                else int(os.getenv("LLM_MAX_RETRIES", "3"))
            )

            _embed_model_path = embed_model_path or os.getenv(
                "EMBED_MODEL_PATH", "BAAI/bge-m3"
            )
            self.EMBED_BATCH_SIZE = (
                embed_batch_size
                if embed_batch_size is not None
                else int(os.getenv("EMBED_BATCH_SIZE", "64"))
            )

            self.logger.info(f"Loading embedding model from: {_embed_model_path}")
            self.EMBED_MODEL = SentenceTransformer(
                _embed_model_path, local_files_only=True
            )
            self.EMBED_DIM = self.EMBED_MODEL.get_sentence_embedding_dimension()
            self.logger.info(f"Embedding model loaded (dim={self.EMBED_DIM})")


            self.logger.info(f"Initializing LLM Client for model: {self.CHAT_MODEL}")
            self.LLM_CLIENT = instructor.from_openai(
                OpenAI(
                    base_url=self.BASE_URL, api_key=self.API_KEY, timeout=self.TIMEOUT
                ),
                mode=instructor.Mode.JSON,
            )
            self._initialized = True

    @classmethod
    def reset(cls):

        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "EMBED_MODEL"):
                    del cls._instance.EMBED_MODEL

                if hasattr(cls._instance, "LLM_CLIENT"):
                    del cls._instance.LLM_CLIENT

                cls._instance._initialized = False
                cls._instance = None


    def get_client(self) -> instructor.Instructor:
        return self.LLM_CLIENT

    def get_model_name(self) -> str:
        return self.CHAT_MODEL or ""

    def get_base_url(self) -> str:
        return self.BASE_URL or ""
