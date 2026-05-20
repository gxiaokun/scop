import os
import faiss
import igraph as ig
import numpy as np
from threading import Lock
from typing import Optional
from enum import Enum

from src.config.base_config import BaseConfig


class TemporalDatasets(str, Enum):
    MULTITQ = "MultiTQ"
    TimelineCronQR = "TimelineCronQR"


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


class RAGConfig(BaseConfig):

    _instance: Optional["RAGConfig"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RAGConfig, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        base_store_dir: Optional[str] = None,
        faiss_type: Optional[str] = None,
        faiss_embed_dtype=np.float32,
        **kwargs,
    ):

        if getattr(self, "_initialized", False):
            return

        with self.__class__._lock:

            if getattr(self, "_initialized", False):
                return
            super().__init__(**kwargs)

            self.DATASET_NAME = dataset_name or os.getenv("DATASET_NAME")
            self.DATASET_DIR = dataset_dir or os.getenv("DATASET_DIR")

            _raw_base_store_dir = base_store_dir or os.getenv("BASE_STORE_DIR")
            self.BASE_STORE_DIR = _raw_base_store_dir
            self.MAIN_BASE_STORE_DIR = f"{_raw_base_store_dir}/{self.DATASET_NAME}"

            self.FAISS_TYPE = faiss_type or os.getenv("FAISS_TYPE", "flat")
            self.FAISS_EMBED_DTYPE = faiss_embed_dtype

            if hasattr(self, "logger"):
                self.logger.info(
                    f"RAGConfig initialized for dataset: {self.DATASET_NAME}"
                )

            self._initialized = True

    def get_dataset_kg(self) -> str:
        dataset_dir = self.DATASET_DIR or ""
        dataset_name = self.DATASET_NAME or ""
        return f"{dataset_dir}/{dataset_name}/kg/full.txt"

    def get_kg_pkl(self) -> str:

        base_store_dir = self.MAIN_BASE_STORE_DIR or ""
        dataset_name = self.DATASET_NAME or ""
        return f"{base_store_dir}/{dataset_name}.pkl"

    def get_dataset_test_file(self) -> str:

        dataset_dir = self.DATASET_DIR or ""
        dataset_name = self.DATASET_NAME or ""
        return f"{dataset_dir}/{dataset_name}/questions/test.json"

    # To baseline
    def get_dataset_kg_pkl_by_name(self, dataset_name: str) -> str:

        base_store_dir = self.MAIN_BASE_STORE_DIR or ""
        return f"{base_store_dir}/{dataset_name}.pkl"

    def get_dataset_kg_by_name(self, dataset_name: str) -> str:

        dataset_dir = RAGConfig().DATASET_DIR or ""
        return f"{dataset_dir}/{dataset_name}/kg/full.txt"

    def get_dataset_test_file_by_name(self, dataset_name: str) -> str:

        dataset_dir = RAGConfig().DATASET_DIR or ""
        return f"{dataset_dir}/{dataset_name}/questions/test.json"
