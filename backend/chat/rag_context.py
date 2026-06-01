"""单轮对话内 RAG trace 暂存与检索运行时配置（如思考模式）。"""

from typing import Any, Optional

_LAST_RAG_CONTEXT: Optional[dict] = None
_RAG_RUNTIME_CONFIG: dict[str, Any] = {}


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def record_rag_context(rag_trace: dict) -> None:
    if rag_trace:
        global _LAST_RAG_CONTEXT
        _LAST_RAG_CONTEXT = {"rag_trace": rag_trace}


def set_rag_config(config: dict[str, Any]) -> None:
    """设置当前请求内的检索配置（如 think_mode）。"""
    global _RAG_RUNTIME_CONFIG
    _RAG_RUNTIME_CONFIG = dict(config or {})


def get_rag_config() -> dict[str, Any]:
    return dict(_RAG_RUNTIME_CONFIG)


def clear_rag_config() -> None:
    global _RAG_RUNTIME_CONFIG
    _RAG_RUNTIME_CONFIG = {}
