import os
import re
import time
import logging
from threading import Lock
from typing import List, Dict, Any, Optional
import instructor

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from src.config.base_config import logger
from src.config.rag_config import RAGConfig
from src.config.llm_config import LLMConfig
from src.utils.llm_utils import embed_fn


def get_vectors_file() -> str:
    return f"{RAGConfig().MAIN_BASE_STORE_DIR}/CR_Baselines/vectors.npy"


def get_corpus_file() -> str:
    return f"{RAGConfig().MAIN_BASE_STORE_DIR}/CR_Baselines/corpus.pkl"


# ==========================
# 文本预处理
# ==========================


def clean_and_normalize(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_line_to_sentence(line: str) -> Optional[str]:
    parts = line.strip().split()
    if not parts:
        return None

    cleaned = [clean_and_normalize(p) for p in parts]
    n = len(cleaned)

    if n == 4:
        s, r, o, date = cleaned
        return f"{s} {r} {o} on {date}"
    elif n == 5:
        s, r, o, start, end = cleaned
        return f"{s} {r} {o} from {start} to {end}"
    else:
        return None


class VectorDatabase:

    def __init__(self):
        self.corpus: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.dim: int = 0

    # ---- 简单持久化 ----
    def save(self) -> None:
        if self.vectors is None or not self.corpus:
            logger.warning("no vectors or corpus to save.")
            return

        VECTORS_FILE = get_vectors_file()
        CORPUS_FILE = get_corpus_file()

        os.makedirs(os.path.dirname(VECTORS_FILE), exist_ok=True)
        np.save(VECTORS_FILE, self.vectors)

        import pickle

        with open(CORPUS_FILE, "wb") as f:
            pickle.dump(self.corpus, f)

        logger.info(f"vector and corpus saved to {VECTORS_FILE} and {CORPUS_FILE}")

    def load(self) -> bool:
        VECTORS_FILE = get_vectors_file()
        CORPUS_FILE = get_corpus_file()

        if not (os.path.exists(VECTORS_FILE) and os.path.exists(CORPUS_FILE)):
            return False

        try:
            self.vectors = np.load(VECTORS_FILE)
            self.dim = int(self.vectors.shape[1])

            import pickle

            with open(CORPUS_FILE, "rb") as f:
                self.corpus = pickle.load(f)

            logger.info(f"has {len(self.corpus)} items, dimension {self.dim}")
            return True
        except Exception as e:
            logger.error(f"failed to load cache, will delete old files and rebuild: {e}")
            for p in (VECTORS_FILE, CORPUS_FILE):
                if os.path.exists(p):
                    os.remove(p)
            self.vectors = None
            self.corpus = []
            self.dim = 0
            return False

    def build_from_txt(self, file_path: str, embed_func) -> None:

        if self.load():
            return

        if not os.path.exists(file_path):
            logger.error(f"knowledge base file does not exist: {file_path}")
            return

        logger.info(f"starting to read file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        sentences: List[str] = []
        for line in raw_lines:
            s = process_line_to_sentence(line)
            if s:
                sentences.append(s)

        if not sentences:
            logger.error("file content is empty or all lines have incorrect format.")
            return

        self.corpus = sentences
        logger.info(f"all {len(self.corpus)} sentences, starting to generate vectors...")

        try:
            start = time.time()
            self.vectors = embed_func(
                self.corpus, normalize=True, show_progress_bar=True
            )
            self.dim = int(self.vectors.shape[1])
            logger.info(
                f"vector generation completed: {self.vectors.shape[0]} items, dimension {self.dim}, took {time.time() - start:.2f}s"
            )

            self.save()
        except Exception as e:
            logger.error(f"failed to generate vectors: {e}")
            self.corpus = []
            self.vectors = None
            self.dim = 0

    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.vectors is None or not self.corpus:
            logger.error("vector library is not built.")
            return []

        if self.dim <= 0:
            logger.error("incorrect vector dimension information.")
            return []

        q_vec = embed_fn([query_text], normalize=True).flatten()
        if q_vec.shape[0] != self.dim:
            logger.error(
                f"query vector dimension {q_vec.shape[0]} is inconsistent with library vector dimension {self.dim}."
            )
            return []

        sims = self.vectors @ q_vec
        top_k_idx = np.argsort(sims)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in top_k_idx:
            idx = int(idx)
            results.append(
                {
                    "index": idx,
                    "sentence": self.corpus[idx],
                    "similarity": float(sims[idx]),
                }
            )
        return results
