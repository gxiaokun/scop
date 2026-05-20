import os
import argparse
import numpy as np
from collections import defaultdict


from src.config.base_config import BaseConfig
from src.config.llm_config import LLMConfig
from src.config.rag_config import RAGConfig, TemporalDatasets
from src.utils.dataset_utils import build_all_with_faiss, load_graph_and_indexes
from src.config.base_config import logger
from src.utils.common_utils import FileHelper

from src.eval import eval_corn
from src.eval import eval_multi

from src.scop import run_once


def parse_args():
    parser = argparse.ArgumentParser(description="SCoP Evaluation Launcher")

    parser.add_argument("--dataset", type=str, default="TimelineCronQR")
    parser.add_argument("--test_size", type=int, default=60000)
    parser.add_argument("--runs", type=int, default=2)

    parser.add_argument(
        "--ablation_type",
        type=str,
        default="full",
        choices=["full", "no_triple", "no_align", "no_constraint"],
        help="full / no_triple / no_align / no_constraint",
    )
    parser.add_argument(
        "--ablation_topk",
        type=int,
        default=10,
        help="Top-k facts used by weak retrieval in ablation modes.",
    )

    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_timeouts", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(">>> Step 1: Initializing global basic configuration...")

    if args.dataset not in TemporalDatasets._value2member_map_:
        raise ValueError(
            f"Unsupported dataset: {args.dataset}. "
            f"Supported datasets are: {[d.value for d in TemporalDatasets]}"
        )

    BaseConfig(log_level="INFO", max_workers=args.max_workers)
    LLMConfig()
    RAGConfig(
        dataset_name=args.dataset,
    )

    global_llm_config = LLMConfig()
    global_rag_config = RAGConfig()
    dataset_name = args.dataset
    base_store_path = global_rag_config.MAIN_BASE_STORE_DIR
    llm_type = global_llm_config.get_model_name()

    experiment_output_dir = os.getenv(
        "EXPERIMENT_OUTPUT_DIR",
        "./test_run_results",
    )

    # Build the base path containing all run directories
    ablation_base_root = os.path.join(
        experiment_output_dir,
        "ablation",
        f"{dataset_name}",
        llm_type,
    )

    print(">>> Step 4: Building and loading graph database and vector indexes...")
    build_all_with_faiss(base_store_path)
    g, ent_index, rel_index, ts_index, meta = load_graph_and_indexes(base_store_path)
    print(
        ">>> Environment and model initialization complete, starting business process!\n"
    )

    for i in range(1, args.runs + 1):
        run_once(
            run_id=i,
            ablation_root=ablation_base_root,
            test_size=args.test_size,
            g=g,
            ent_index=ent_index,
            rel_index=rel_index,
            ablation_type=args.ablation_type,
            ablation_topk=args.ablation_topk,
            max_timeouts=args.max_timeouts,
            shard_size=500,
        )
