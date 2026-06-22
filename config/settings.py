# -*- coding: utf-8 -*-
"""Central configuration loader for InsureRAG.

Priority:
1. Built-in defaults
2. config/config.yaml
3. .env / process environment variables
"""

from __future__ import annotations

import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from observability.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "vllm",
        "base_url": "http://localhost:8002/v1",
        "model": "Qwen/Qwen3-8B-AWQ",
        "api_key": "EMPTY",
        "temperature": 0.2,
        "max_tokens": 1024,
        "timeout_seconds": 60,
        "health_timeout_seconds": 3,
        # 默认关闭 Qwen3 thinking：评测显示对抽取式条款问答质量基本无影响，但延迟约减半。
        "enable_thinking": False,
    },
    "embedding": {
        "provider": "bge_m3",
        "model_path": "BAAI/bge-m3",
        "device": "cpu",
        "use_fp16": False,
        "local_files_only": True,
    },
    "vector_db": {
        "provider": "milvus",
        "host": "localhost",
        "port": 19530,
        "collection": "insure_rag",
    },
    "retrieval": {
        "dense_top_k": 20,
        "sparse_top_k": 20,
        "final_top_k": 5,
        "rrf_k": 60,
    },
    "reranker": {
        "enabled": True,
        "model": "BAAI/bge-reranker-base",
        "device": "cpu",
        "local_files_only": True,
    },
    "ocr": {
        "enabled": True,
        "fallback_threshold": 0.6,
        "good_threshold": 0.8,
        "min_text_density": 100,
        "sample_pages": 3,
        "image_page_ratio_threshold": 0.5,
    },
    "trace": {
        "enabled": True,
        "output_dir": "traces",
        "retention_days": 7,
        "max_files": 1000,
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "auth_enabled": False,
        "api_key": "",
        "max_upload_mb": 50,
        "allowed_upload_extensions": [".pdf"],
        "max_uploaded_documents": 20,
    },
    "dashboard": {
        "host": "0.0.0.0",
        "port": 8001,
    },
    "cache": {
        "enabled": True,
        "backend": "redis",
        "path": "data/cache.json",
        "redis_url": "redis://localhost:6379/0",
        "redis_prefix": "insurerag:cache",
        "fallback_to_json": True,
        "ttl_seconds": 86400,
        "answer_cache": True,
        "search_cache": True,
        "semantic_cache": True,
        "semantic_direct_threshold": 0.98,
        "semantic_answer_threshold": 0.95,
        "semantic_retrieval_threshold": 0.93,
        "evidence_overlap_threshold": 0.6,
        "require_top1_evidence_match": True,
    },
}


ENV_MAPPING: dict[str, tuple[str, ...]] = {
    "INSURERAG_LLM_PROVIDER": ("llm", "provider"),
    "INSURERAG_LLM_BASE_URL": ("llm", "base_url"),
    "INSURERAG_LLM_MODEL": ("llm", "model"),
    "INSURERAG_LLM_API_KEY": ("llm", "api_key"),
    "INSURERAG_LLM_TEMPERATURE": ("llm", "temperature"),
    "INSURERAG_LLM_MAX_TOKENS": ("llm", "max_tokens"),
    "INSURERAG_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "INSURERAG_LLM_HEALTH_TIMEOUT_SECONDS": ("llm", "health_timeout_seconds"),
    "INSURERAG_LLM_ENABLE_THINKING": ("llm", "enable_thinking"),
    "INSURERAG_EMBEDDING_PROVIDER": ("embedding", "provider"),
    "INSURERAG_EMBEDDING_MODEL_PATH": ("embedding", "model_path"),
    "INSURERAG_EMBEDDING_DEVICE": ("embedding", "device"),
    "INSURERAG_EMBEDDING_USE_FP16": ("embedding", "use_fp16"),
    "INSURERAG_EMBEDDING_LOCAL_FILES_ONLY": ("embedding", "local_files_only"),
    "INSURERAG_VECTOR_DB_HOST": ("vector_db", "host"),
    "INSURERAG_VECTOR_DB_PORT": ("vector_db", "port"),
    "INSURERAG_VECTOR_DB_COLLECTION": ("vector_db", "collection"),
    "INSURERAG_RETRIEVAL_DENSE_TOP_K": ("retrieval", "dense_top_k"),
    "INSURERAG_RETRIEVAL_SPARSE_TOP_K": ("retrieval", "sparse_top_k"),
    "INSURERAG_RETRIEVAL_FINAL_TOP_K": ("retrieval", "final_top_k"),
    "INSURERAG_RETRIEVAL_RRF_K": ("retrieval", "rrf_k"),
    "INSURERAG_RERANKER_ENABLED": ("reranker", "enabled"),
    "INSURERAG_RERANKER_MODEL": ("reranker", "model"),
    "INSURERAG_RERANKER_DEVICE": ("reranker", "device"),
    "INSURERAG_RERANKER_LOCAL_FILES_ONLY": ("reranker", "local_files_only"),
    "INSURERAG_OCR_ENABLED": ("ocr", "enabled"),
    "INSURERAG_OCR_FALLBACK_THRESHOLD": ("ocr", "fallback_threshold"),
    "INSURERAG_OCR_GOOD_THRESHOLD": ("ocr", "good_threshold"),
    "INSURERAG_OCR_MIN_TEXT_DENSITY": ("ocr", "min_text_density"),
    "INSURERAG_OCR_SAMPLE_PAGES": ("ocr", "sample_pages"),
    "INSURERAG_OCR_IMAGE_PAGE_RATIO_THRESHOLD": ("ocr", "image_page_ratio_threshold"),
    "INSURERAG_TRACE_ENABLED": ("trace", "enabled"),
    "INSURERAG_TRACE_OUTPUT_DIR": ("trace", "output_dir"),
    "INSURERAG_TRACE_RETENTION_DAYS": ("trace", "retention_days"),
    "INSURERAG_TRACE_MAX_FILES": ("trace", "max_files"),
    "INSURERAG_API_HOST": ("api", "host"),
    "INSURERAG_API_PORT": ("api", "port"),
    "INSURERAG_API_AUTH_ENABLED": ("api", "auth_enabled"),
    "INSURERAG_API_KEY": ("api", "api_key"),
    "INSURERAG_API_MAX_UPLOAD_MB": ("api", "max_upload_mb"),
    "INSURERAG_API_MAX_UPLOADED_DOCS": ("api", "max_uploaded_documents"),
    "INSURERAG_DASHBOARD_HOST": ("dashboard", "host"),
    "INSURERAG_DASHBOARD_PORT": ("dashboard", "port"),
    "INSURERAG_CACHE_ENABLED": ("cache", "enabled"),
    "INSURERAG_CACHE_BACKEND": ("cache", "backend"),
    "INSURERAG_CACHE_PATH": ("cache", "path"),
    "INSURERAG_CACHE_REDIS_URL": ("cache", "redis_url"),
    "INSURERAG_CACHE_REDIS_PREFIX": ("cache", "redis_prefix"),
    "INSURERAG_CACHE_FALLBACK_TO_JSON": ("cache", "fallback_to_json"),
    "INSURERAG_CACHE_TTL_SECONDS": ("cache", "ttl_seconds"),
    "INSURERAG_CACHE_ANSWER_CACHE": ("cache", "answer_cache"),
    "INSURERAG_CACHE_SEARCH_CACHE": ("cache", "search_cache"),
    "INSURERAG_CACHE_SEMANTIC_CACHE": ("cache", "semantic_cache"),
    "INSURERAG_CACHE_SEMANTIC_DIRECT_THRESHOLD": ("cache", "semantic_direct_threshold"),
    "INSURERAG_CACHE_SEMANTIC_ANSWER_THRESHOLD": ("cache", "semantic_answer_threshold"),
    "INSURERAG_CACHE_SEMANTIC_RETRIEVAL_THRESHOLD": ("cache", "semantic_retrieval_threshold"),
    "INSURERAG_CACHE_EVIDENCE_OVERLAP_THRESHOLD": ("cache", "evidence_overlap_threshold"),
    "INSURERAG_CACHE_REQUIRE_TOP1_EVIDENCE_MATCH": ("cache", "require_top1_evidence_match"),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_dotenv(path: Path = DEFAULT_ENV_PATH) -> None:
    if not path.exists():
        logger.info("No .env file found, using YAML/default configuration.")
        return

    loaded_keys = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded_keys += 1
    logger.info("Loaded %s keys from %s.", loaded_keys, path)


def _cast_env_value(value: str, old_value: Any) -> Any:
    if isinstance(old_value, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(value)
    if isinstance(old_value, float):
        return float(value)
    return value


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        current = current[key]
    return current


class Settings:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    @property
    def llm(self) -> dict[str, Any]:
        return self.data["llm"]

    @property
    def embedding(self) -> dict[str, Any]:
        return self.data["embedding"]

    @property
    def vector_db(self) -> dict[str, Any]:
        return self.data["vector_db"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.data["retrieval"]

    @property
    def reranker(self) -> dict[str, Any]:
        return self.data["reranker"]

    @property
    def ocr(self) -> dict[str, Any]:
        return self.data["ocr"]

    @property
    def trace(self) -> dict[str, Any]:
        return self.data["trace"]

    @property
    def api(self) -> dict[str, Any]:
        return self.data["api"]

    @property
    def dashboard(self) -> dict[str, Any]:
        return self.data["dashboard"]

    @property
    def cache(self) -> dict[str, Any]:
        return self.data["cache"]

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.vector_db['host']}:{self.vector_db['port']}"


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    _load_dotenv()

    config = deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to parse config file: {config_path}") from exc
        config = _deep_merge(config, loaded)
        logger.info("Loaded YAML configuration from %s.", config_path)
    else:
        logger.warning("Config file %s not found, using built-in defaults.", config_path)

    # Backward-compatible aliases from earlier project versions.
    legacy_env = {
        "MILVUS_URI": ("vector_db", "_uri"),
        "ENABLE_OCR_FALLBACK": ("ocr", "enabled"),
        "RAGAS_EMBEDDING_MODEL_PATH": ("embedding", "model_path"),
    }
    for env_key, path in legacy_env.items():
        value = os.getenv(env_key)
        if not value:
            continue
        if path == ("vector_db", "_uri"):
            host_port = value.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in host_port:
                host, port = host_port.rsplit(":", 1)
                config["vector_db"]["host"] = host
                config["vector_db"]["port"] = int(port)
            continue
        old_value = _get_nested(config, path)
        _set_nested(config, path, _cast_env_value(value, old_value))

    for env_key, path in ENV_MAPPING.items():
        value = os.getenv(env_key)
        if value is None:
            continue
        old_value = _get_nested(config, path)
        _set_nested(config, path, _cast_env_value(value, old_value))

    resolved = Settings(config)
    logger.info(
        "Configuration ready: llm=%s, milvus=%s/%s, collection=%s",
        resolved.llm["model"],
        resolved.vector_db["host"],
        resolved.vector_db["port"],
        resolved.vector_db["collection"],
    )
    return resolved


settings = load_settings()
