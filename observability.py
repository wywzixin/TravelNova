"""Optional Langfuse tracing for the travel assistant.

Tracing is deliberately isolated from the agent logic.  If Langfuse is not
installed, disabled, or temporarily unavailable, the application keeps
running with no-op observations.  Inputs and outputs are redacted before they
leave the process so API keys accidentally present in tool errors or old
training data are not copied into traces.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

try:
    from dotenv import load_dotenv

    # Resolve the project-local file instead of depending on the process cwd.
    load_dotenv(Path(__file__).with_name(".env"), override=False)
except ImportError:  # pragma: no cover - dependency is optional at import time
    pass

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    Langfuse = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

_SECRET_NAME = re.compile(
    r"(?:api[_-]?key|secret|token|password|authorization|credential)",
    re.IGNORECASE,
)
_SECRET_QUERY = re.compile(
    r"([?&](?:api[_-]?key|key|secret|token|password)=)[^&\s]+",
    re.IGNORECASE,
)
_SECRET_JSON = re.compile(
    r'("\s*(?:api[_-]?key|secret|token|password)\s*"\s*:\s*")[^"\\]*(")',
    re.IGNORECASE,
)
_MAX_STRING_LENGTH = 12000


def _redact_string(value: str) -> str:
    value = _SECRET_QUERY.sub(r"\1***REDACTED***", value)
    value = _SECRET_JSON.sub(r"\1***REDACTED***\2", value)
    if len(value) > _MAX_STRING_LENGTH:
        value = value[:_MAX_STRING_LENGTH] + "...[truncated]"
    return value


def safe_value(value: Any) -> Any:
    """Return a JSON-safe, size-limited, secret-redacted representation."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if dataclasses.is_dataclass(value):
        return safe_value(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return safe_value(value.model_dump())
        except Exception:
            pass
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_NAME.search(key_text):
                result[key_text] = "***REDACTED***"
            else:
                result[key_text] = safe_value(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [safe_value(item) for item in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return _redact_string(str(value))


def usage_details(response: Any) -> Optional[Dict[str, int]]:
    """Extract OpenAI-compatible token usage when a provider returns it."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, Mapping):
        return None

    result: Dict[str, int] = {}
    for source, target in (
        ("prompt_tokens", "input"),
        ("completion_tokens", "output"),
        ("total_tokens", "total"),
    ):
        value = usage.get(source)
        if isinstance(value, (int, float)):
            result[target] = int(value)
    return result or None


class _NoOpObservation:
    def update(self, **_: Any) -> "_NoOpObservation":
        return self

    def end(self, **_: Any) -> None:
        return None


class LangfuseObservability:
    """Small compatibility wrapper around the Langfuse v4 observation API."""

    def __init__(self) -> None:
        enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        self.client = None
        self.enabled = False

        if not enabled or Langfuse is None or not public_key or not secret_key:
            return

        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=os.getenv(
                    "LANGFUSE_BASE_URL",
                    os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                ).rstrip("/"),
                environment=os.getenv("LANGFUSE_ENVIRONMENT", "development"),
                flush_at=int(os.getenv("LANGFUSE_FLUSH_AT", "1")),
            )
            self.enabled = True
        except Exception:
            logger.exception("Langfuse 初始化失败，将继续无追踪运行")

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any = None,
        metadata: Any = None,
        model: Optional[str] = None,
    ) -> Iterator[Any]:
        """Create a no-op or real observation without breaking application flow."""

        if not self.client:
            yield _NoOpObservation()
            return

        try:
            manager = self.client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=safe_value(input),
                metadata=safe_value(metadata),
                model=model,
            )
            observation = manager.__enter__()
        except Exception:
            logger.exception("Langfuse observation 创建失败: %s", name)
            yield _NoOpObservation()
            return

        try:
            yield observation
        except BaseException:
            try:
                manager.__exit__(*sys.exc_info())
            except Exception:
                logger.exception("Langfuse observation 结束失败: %s", name)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.exception("Langfuse observation 结束失败: %s", name)

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Langfuse flush 失败")

    def shutdown(self) -> None:
        if self.client:
            try:
                self.client.shutdown()
            except Exception:
                logger.exception("Langfuse shutdown 失败")


_observability: Optional[LangfuseObservability] = None


def get_observability() -> LangfuseObservability:
    global _observability
    if _observability is None:
        _observability = LangfuseObservability()
    return _observability

