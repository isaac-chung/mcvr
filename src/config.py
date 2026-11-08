"""One YAML file per experiment. The config hash names the result folder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from src.cache import Cache, shared_cache

_LATER_FIELDS = (
    "dataset",
    "n_videos",
    "n_frames_vis",
    "n_frames_desc",
    "describe_llm",
    "query_llm",
    "asr_model",
    "queries_per_video",
    "describe_prompt",
    "facts_prompt",
    "query_prompt",
    "cache",
)


@dataclass(frozen=True)
class Config:
    name: str
    seed: int = 0
    n_queries: int = 100  # test queries
    n_dev: int = 300  # development queries: pick fixed plans and train the learned router
    k: int = 10  # evaluation cutoff
    depth: int = 100  # ranked list length kept per index
    oracle_epsilon: float = 0.01  # the oracle may lose this much nDCG to be cheaper
    relaxed_epsilon: float = 0.05  # sensitivity check
    n_distractors: int = 2000  # English test videos with a transcript added to the corpus
    indexes: tuple[str, ...] = ("ASR", "VIS", "DESC")
    # embedder spec, e.g. "st:infgrad/Jasper-Token-Compression-600M" or "hash:256"
    embedder: str = "st:infgrad/Jasper-Token-Compression-600M"
    embedder_kwargs: dict = field(default_factory=dict)
    siglip: str = "google/siglip-so400m-patch14-384"
    llm: str = ""  # e.g. "anthropic:<model>", "openai:<model>", "" disables S2
    router_prompt: str = "prompts/router_v1.txt"
    planner_prompt: str = "prompts/planner_v1.txt"
    max_searches: int = 3  # S3 search budget
    n_rewordings: int = 10  # K for the stability experiment
    paraphrase_llm: str = ""  # LLM spec that writes the rewordings
    rewording_prompt: str = "prompts/rewording_v1.txt"
    workers: int = 8  # parallel queries during the run stage (LLM calls are the slow part)
    data_dir: str = "data/multivent"
    results_dir: str = "results"
    # Dataset choice and the settings of its `prepare` stage (Shot2Story20K and CaReBench build
    # their own descriptions, transcripts and queries from a slice of n_videos videos).
    dataset: str = "multivent"  # multivent | shot2story | carebench
    n_videos: int = 150
    n_frames_vis: int = 16  # frames per video embedded by SigLIP
    n_frames_desc: int = 8  # frames per video shown to the description model
    describe_llm: str = "openai:gpt-4.1-mini-2025-04-14"  # vision model that writes DESC
    query_llm: str = "openai:gpt-4.1-mini-2025-04-14"  # writes the constructed queries
    asr_model: str = "whisper-1"  # CaReBench transcription
    queries_per_video: int = 3
    describe_prompt: str = "prompts/describe_v1.txt"
    facts_prompt: str = "prompts/facts_v1.txt"
    query_prompt: str = "prompts/query_v1.txt"
    # "run": the routing pilot's caches (llm_cache.sqlite per run folder, gen_cache.sqlite per
    # dataset folder); "shared": results/_cache/{llm,vlm,embed}.sqlite, used by the new study
    cache: str = "run"

    def llm_cache(self, run_file: str | Path) -> Cache:
        """Cache for text LLM calls: the shared one, or run_file (the pilot's own file)."""
        return shared_cache("llm", self.results_dir) if self.cache == "shared" else Cache(run_file)

    def vlm_cache(self, run_file: str | Path) -> Cache:
        """Cache for vision calls: the shared one, or run_file (the pilot's own file)."""
        return shared_cache("vlm", self.results_dir) if self.cache == "shared" else Cache(run_file)

    def hash(self) -> str:
        fields = asdict(self)
        for name in _LATER_FIELDS:  # keeps the folder names of results made before these existed
            if fields[name] == self.__dataclass_fields__[name].default:
                del fields[name]
        blob = json.dumps(fields, sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def index_dir(self) -> Path:
        """Embeddings are shared by every experiment over the same corpus texts and embedder.

        The corpus settings are part of the name: hashing the embedder alone put every pilot
        dataset in one folder, so one run overwrote another's index."""
        corpus = {
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "seed": self.seed,
            "n_distractors": self.n_distractors,
        }
        if self.dataset != "multivent":  # these datasets generate their own corpus texts
            corpus |= {
                "n_videos": self.n_videos,
                "n_frames_desc": self.n_frames_desc,
                "describe_llm": self.describe_llm,
                "describe_prompt": self.describe_prompt,
                "asr_model": self.asr_model,
            }
        blob = json.dumps([self.embedder, self.embedder_kwargs, corpus], sort_keys=True)
        return Path(self.results_dir) / "_indexes" / hashlib.sha256(blob.encode()).hexdigest()[:8]

    def result_dir(self) -> Path:
        return Path(self.results_dir) / f"{self.name}-{self.hash()}"


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if "indexes" in raw:
        raw["indexes"] = tuple(raw["indexes"])
    return Config(**raw)
