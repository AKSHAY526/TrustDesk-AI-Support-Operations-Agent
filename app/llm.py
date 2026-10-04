"""The single place where the AI provider is called."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from .config import settings

Task = Literal["triage", "draft", "screen"]


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class LLMTimeoutError(LLMError):
    """The model did not answer within the configured timeout."""


@dataclass
class LLMRequest:
    task: Task
    system: str
    user: str
    # Structured view of the same information, for deterministic stand-ins.
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int

    def json(self) -> dict:
        """Parse the reply as JSON, tolerating code fences and stray prose."""
        raw = self.text.strip()
        fenced = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            brace = re.search(r"\{.*\}", raw, re.DOTALL)
            if brace:
                try:
                    return json.loads(brace.group(0))
                except json.JSONDecodeError as exc:
                    raise LLMError(f"model did not return valid JSON: {exc}") from exc
            raise LLMError("model did not return JSON")


class LLMAdapter(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResult: ...


class OpenAICompatibleLLM:
    """Real provider over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self.name = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout_seconds
        if not self.base_url or not self._api_key:
            raise LLMError("LLM_BASE_URL and LLM_API_KEY must both be set for the real model")

    def complete(self, request: LLMRequest) -> LLMResult:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": 0.0,
        }
        if request.task in ("triage", "screen"):
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"model timed out after {self.timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"provider returned {exc.response.status_code}") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"provider call failed: {exc}") from exc

        return LLMResult(
            text=text,
            model=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


# --- Deterministic stand-in -------------------------------------------------

# Weighted because tickets mix vocabulary: "arrived cracked ... delivered on
# June 24" has shipping words but is a refund case.
_CATEGORY_KEYWORDS: dict[str, dict[str, int]] = {
    "account_security": {
        "system prompt": 4, "api key": 4, "internal note": 4, "hidden instruction": 4,
        "hidden prompt": 4, "identity check": 4, "account email": 4, "change my email": 4,
        "account takeover": 4, "password": 3, "login": 2, "verification": 2,
    },
    "warranty": {
        "warranty": 4, "swelling": 4, "swollen": 4, "malfunction": 3, "stopped working": 3,
        "battery": 3, "months": 1,
    },
    "billing": {
        "double charge": 4, "charged twice": 4, "two charges": 4, "duplicate charge": 4,
        "two payments": 4, "overcharge": 4, "invoice": 3, "billing": 3, "payment failed": 3,
        "charge": 2, "charged": 2,
    },
    "shipping": {
        "tracking": 4, "no movement": 4, "not moved": 4, "hasn't moved": 4, "has not moved": 4,
        "courier": 3, "carrier": 3, "in transit": 3, "lost package": 3, "shipment": 2,
        "delivery": 2, "package": 1, "delivered": 1,
    },
    "refund": {
        "refund": 4, "replacement": 4, "cracked": 4, "damaged": 4, "defective": 4,
        "final sale": 4, "money back": 4, "broken": 3, "return": 3, "faulty": 3,
    },
}

_URGENT_WORDS = ("swelling", "swollen", "fire", "smoke", "burn", "hazard", "overheat", "safety")
_HIGH_WORDS = ("double charge", "charged twice", "two charges", "api key", "system prompt",
               "identity check", "account email", "urgent", "not moved", "no movement",
               "takeover", "internal note")
_LOW_WORDS = ("changed my mind", "no longer need", "just wondering", "question about")
_FRUSTRATED_WORDS = ("cracked", "damaged", "broken", "not moved", "no movement",
                     "double charge", "two charges", "charged twice", "quickly")

_INJECTION_PATTERNS = (
    "ignore all instructions", "ignore previous instructions", "ignore the support policy",
    "ignore the policy", "system override", "disregard the guideline", "disregard policy",
    "you are now allowed", "the policy has changed", "do not mention this",
    "do not tell the human", "hide this from", "reveal your system prompt",
    "print your hidden", "hidden system prompt", "your api key", "internal notes",
    "approve every refund", "approve all refunds", "bypass identity", "skip identity",
    "ignore identity checks", "as an admin", "i authorise", "i authorize",
)

_SECRET_PATTERNS = ("api key", "system prompt", "hidden instruction", "internal note",
                    "credentials", "access token")

_IDENTITY_PATTERNS = ("ignore identity", "skip identity", "bypass identity",
                      "without verification", "no verification", "ignore verification")


def _detect(text: str, patterns: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [p for p in patterns if p in lowered]


class HeuristicLLM:
    """Transparent keyword baseline. Deterministic, offline, no API key."""

    name = "heuristic-stand-in"

    def complete(self, request: LLMRequest) -> LLMResult:
        started = time.perf_counter()
        handler = {
            "triage": self._triage,
            "draft": self._draft,
            "screen": self._screen,
        }[request.task]
        text = json.dumps(handler(request.context))
        return LLMResult(
            text=text,
            model=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # -- individual tasks ---------------------------------------------------

    def _triage(self, context: dict) -> dict:
        ticket = context.get("ticket", {})
        blob = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

        scores = {
            name: sum(weight for word, weight in words.items() if word in blob)
            for name, words in _CATEGORY_KEYWORDS.items()
        }
        best = max(scores, key=lambda name: scores[name])
        category = best if scores[best] > 0 else "general"

        if any(word in blob for word in _URGENT_WORDS):
            priority = "urgent"
        elif any(word in blob for word in _HIGH_WORDS):
            priority = "high"
        elif any(word in blob for word in _LOW_WORDS):
            priority = "low"
        else:
            priority = "medium"

        if any(word in blob for word in _URGENT_WORDS):
            sentiment = "worried"
        elif any(word in blob for word in _FRUSTRATED_WORDS):
            sentiment = "frustrated"
        else:
            sentiment = "neutral"

        unsafe = bool(_detect(blob, _INJECTION_PATTERNS))
        should_escalate = (
            unsafe
            or category == "account_security"
            or priority == "urgent"
        )

        return {
            "category": category,
            "priority": priority,
            "sentiment": sentiment,
            "should_escalate": should_escalate,
            "reason_summary": (
                "Unsafe instruction detected in the customer message; routing to a human."
                if unsafe
                else f"Classified as {category} at {priority} priority from the ticket wording."
            ),
        }

    def _draft(self, context: dict) -> dict:
        ticket = context.get("ticket", {})
        order = context.get("order") or {}
        sections = context.get("sections", [])
        customer = context.get("customer") or {}

        citations = []
        for section in sections:
            if section["doc_id"] not in citations:
                citations.append(section["doc_id"])

        name = (customer.get("name") or "there").split(" ")[0]
        lines = [f"Hi {name},", ""]

        subject = (ticket.get("subject") or "your request").strip()
        if order:
            lines.append(
                f"Thanks for getting in touch about {subject.lower()} on order "
                f"{order.get('order_id')} (placed {order.get('placed_at')}, "
                f"status {order.get('status')})."
            )
        else:
            lines.append(f"Thanks for getting in touch about {subject.lower()}.")
        lines.append("")

        if sections:
            lines.append("Here is what our policy says:")
            for section in sections[:2]:
                summary = " ".join(section["text"].split())
                lines.append(f"- {summary[:260]}{'...' if len(summary) > 260 else ''}")
            lines.append("")

        lines.append("I have noted this on your ticket and a member of the team will confirm the next step.")
        lines.append("")
        lines.append("Best regards,")
        lines.append("Customer Support")

        return {"body": "\n".join(lines), "citations": citations}

    def _screen(self, context: dict) -> dict:
        text = context.get("text", "")
        injection = _detect(text, _INJECTION_PATTERNS)
        secrets = _detect(text, _SECRET_PATTERNS)
        identity = _detect(text, _IDENTITY_PATTERNS)
        kinds = []
        if injection:
            kinds.append("prompt_injection")
        if secrets:
            kinds.append("secret_disclosure")
        if identity:
            kinds.append("identity_bypass")
        return {
            "suspicious": bool(kinds),
            "kinds": kinds,
            "reason": (
                "Matched known unsafe instruction patterns: " + ", ".join(sorted(set(injection + secrets + identity))[:4])
                if kinds
                else "No unsafe instruction detected."
            ),
        }


class ScriptedLLM:
    """Returns queued payloads. Lets tests force specific model behaviour."""

    name = "scripted"

    def __init__(self, responses: list[dict | str] | None = None) -> None:
        self.responses: list[dict | str] = list(responses or [])
        self.calls: list[LLMRequest] = []

    def push(self, response: dict | str) -> None:
        self.responses.append(response)

    def complete(self, request: LLMRequest) -> LLMResult:
        self.calls.append(request)
        if not self.responses:
            raise LLMError("ScriptedLLM has no queued response")
        payload = self.responses.pop(0)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResult(text=text, model=self.name, latency_ms=0)


class FailingLLM:
    """Always fails. Used to prove the ticket escalates instead of erroring."""

    name = "failing"

    def __init__(self, error: LLMError | None = None) -> None:
        self.error = error or LLMTimeoutError("simulated provider timeout")

    def complete(self, request: LLMRequest) -> LLMResult:  # noqa: ARG002
        raise self.error


_override: LLMAdapter | None = None


def set_llm(adapter: LLMAdapter | None) -> None:
    """Install an adapter for the process. Used by tests and the app factory."""
    global _override
    _override = adapter


def get_llm() -> LLMAdapter:
    """The active adapter: an override, the real provider, or the stand-in."""
    if _override is not None:
        return _override
    if settings.use_real_model:
        try:
            return OpenAICompatibleLLM()
        except LLMError:
            return HeuristicLLM()
    return HeuristicLLM()
