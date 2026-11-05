"""Shared paths, seeds, and constants for the ClimRetrieve hierarchy pilot."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
RAW = DATA / "raw" / "ClimRetrieve"
PROCESSED = DATA / "processed"
EMB_DIR = PROCESSED / "embeddings"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

REPORT_LEVEL_PATH = RAW / "Report-Level Dataset" / "ClimRetrieve_ReportLevel_V1.csv"
BASE_PATH = RAW / "Expert-Annotated Relevant Sources Dataset" / "ClimRetrieve_base.xlsx"
CORE_PATH = RAW / "Expert-Annotated Relevant Sources Dataset" / "Core_Questions_with_Explanations.xlsx"
CLIMRETRIEVE_URL = "https://github.com/tobischimanski/ClimRetrieve.git"

SEED = 42
GOLD_THRESHOLD = 2.0
K_LIST = (5, 10, 20)
N_BOOTSTRAP = 1000
EMBED_MODEL_PRIMARY = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
# Official BGE query instruction (documents are unprefixed).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LEVELS = ("leaf", "pillar")
LAMBDAS = (1, 2, 3)
HNSW_M = 32
HNSW_EFSEARCH = (32, 128)
SELECTIVITIES = (0.05, 0.1, 0.25, 0.5)
TIMING_N_QUERIES = 1000
TIMING_WARMUP = 20
SYNTHETIC_NS = (100_000, 1_000_000)
LABEL_SAMPLE_N = 150
NOISE_STD = 0.01
