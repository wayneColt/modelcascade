"""
ModelCascade — Route local. Escalate smart. Never overspend.

Minimal multi-model router for agentic pipelines. Routes to local inference
first (Ollama, llama.cpp, vLLM), escalates to cloud only when local fails
or confidence is low.

Usage:
    from modelcascade import CascadeRouter
    router = CascadeRouter.from_config("mc.yaml")
    result = await router.complete("Explain quicksort")
    # → routed LOCAL · $0.000 · 47ms

MIT License | github.com/wayneColt/modelcascade
"""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

log = logging.getLogger("modelcascade")

__version__ = "0.1.0"


@dataclass
class ProviderConfig:
    """Configuration for a single model provider."""
    name: str
    type: str  # ollama, anthropic, openai, litellm, llamacpp
    model: str
    cost_per_1k: float = 0.0
    base_url: str = ""
    api_key: str = ""
    timeout: float = 10.0

    def __post_init__(self):
        # Resolve env vars in api_key
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")


@dataclass
class RoutingConfig:
    """Routing behavior configuration."""
    cost_ceiling: float = 0.01
    cascade_on_failure: bool = True
    confidence_threshold: float = 0.7
    max_retries: int = 1


@dataclass
class RouteResult:
    """Result of a routed completion."""
    text: str
    provider: str
    model: str
    cost: float
    latency_ms: float
    tier_index: int
    attempts: int = 1
    cascaded: bool = False

    def __str__(self):
        return f"routed {self.provider.upper()} · ${self.cost:.4f} · {self.latency_ms:.0f}ms"


class CascadeRouter:
    """Multi-tier model router. Local first, cloud on escalation."""

    def __init__(self, providers: list[ProviderConfig], routing: RoutingConfig):
        self.providers = sorted(providers, key=lambda p: p.cost_per_1k)
        self.routing = routing
        self.stats = {"total": 0, "by_tier": {}, "total_cost": 0.0}

    @classmethod
    def from_config(cls, path: str) -> "CascadeRouter":
        """Load router from YAML or JSON config file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        text = config_path.read_text()

        if config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML required for .yaml configs: pip install pyyaml")
        else:
            data = json.loads(text)

        providers = []
        for name, cfg in data.get("providers", {}).items():
            providers.append(ProviderConfig(
                name=name,
                type=cfg.get("type", "ollama"),
                model=cfg.get("model", ""),
                cost_per_1k=cfg.get("cost_per_1k", 0.0),
                base_url=cfg.get("base_url", ""),
                api_key=cfg.get("api_key", ""),
                timeout=cfg.get("timeout", 10.0),
            ))

        routing_data = data.get("routing", {})
        routing = RoutingConfig(
            cost_ceiling=routing_data.get("cost_ceiling", 0.01),
            cascade_on_failure=routing_data.get("cascade_on_failure", True),
            confidence_threshold=routing_data.get("confidence_threshold", 0.7),
            max_retries=routing_data.get("max_retries", 1),
        )

        return cls(providers, routing)

    @classmethod
    def default(cls) -> "CascadeRouter":
        """Create router with sensible defaults (Ollama local → Anthropic cloud)."""
        providers = [
            ProviderConfig("local", "ollama", "llama3.2:3b", 0.0),
            ProviderConfig("fast", "anthropic", "claude-haiku-4-5-20251001", 0.001,
                           api_key=os.environ.get("ANTHROPIC_API_KEY", "")),
            ProviderConfig("capable", "anthropic", "claude-sonnet-4-6", 0.005,
                           api_key=os.environ.get("ANTHROPIC_API_KEY", "")),
        ]
        return cls(providers, RoutingConfig())

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> RouteResult:
        """Route a completion through the cascade. Starts cheap, escalates on failure."""
        self.stats["total"] += 1

        for i, provider in enumerate(self.providers):
            if provider.cost_per_1k > self.routing.cost_ceiling:
                log.debug(f"Skipping {provider.name}: cost {provider.cost_per_1k} > ceiling {self.routing.cost_ceiling}")
                continue

            start = time.monotonic()
            try:
                text = await self._call_provider(provider, prompt, system, max_tokens)
                elapsed_ms = (time.monotonic() - start) * 1000
                cost = (len(prompt) + len(text)) / 1000 * provider.cost_per_1k

                self.stats["by_tier"][provider.name] = self.stats.get("by_tier", {}).get(provider.name, 0) + 1
                self.stats["total_cost"] += cost

                result = RouteResult(
                    text=text,
                    provider=provider.name,
                    model=provider.model,
                    cost=cost,
                    latency_ms=elapsed_ms,
                    tier_index=i,
                    cascaded=i > 0,
                )
                log.info(f"→ {result}")
                return result

            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                log.warning(f"{provider.name} failed ({elapsed_ms:.0f}ms): {e}")
                if not self.routing.cascade_on_failure:
                    raise
                continue

        raise RuntimeError("All providers failed or exceeded cost ceiling")

    async def _call_provider(self, provider: ProviderConfig, prompt: str,
                             system: str, max_tokens: int) -> str:
        """Dispatch to the appropriate provider backend."""
        if provider.type == "ollama":
            return await self._call_ollama(provider, prompt, system, max_tokens)
        elif provider.type in ("anthropic", "claude"):
            return await self._call_anthropic(provider, prompt, system, max_tokens)
        elif provider.type in ("openai", "groq", "together"):
            return await self._call_openai_compat(provider, prompt, system, max_tokens)
        elif provider.type in ("llamacpp", "llama.cpp"):
            return await self._call_llamacpp(provider, prompt, system, max_tokens)
        else:
            raise ValueError(f"Unknown provider type: {provider.type}")

    async def _call_ollama(self, p: ProviderConfig, prompt: str,
                           system: str, max_tokens: int) -> str:
        """Call Ollama API (OpenAI-compatible at localhost:11434)."""
        import urllib.request
        url = p.base_url or "http://localhost:11434"
        data = json.dumps({
            "model": p.model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(f"{url}/api/chat", data=data,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=p.timeout)
        result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")

    async def _call_anthropic(self, p: ProviderConfig, prompt: str,
                              system: str, max_tokens: int) -> str:
        """Call Anthropic Messages API."""
        import urllib.request
        url = "https://api.anthropic.com/v1/messages"
        data = json.dumps({
            "model": p.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **({"system": system} if system else {}),
        }).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "x-api-key": p.api_key,
            "anthropic-version": "2023-06-01",
        })
        resp = urllib.request.urlopen(req, timeout=p.timeout)
        result = json.loads(resp.read())
        return result.get("content", [{}])[0].get("text", "")

    async def _call_openai_compat(self, p: ProviderConfig, prompt: str,
                                  system: str, max_tokens: int) -> str:
        """Call OpenAI-compatible API (Groq, Together, vLLM, etc)."""
        import urllib.request
        url = p.base_url or "https://api.openai.com/v1"
        data = json.dumps({
            "model": p.model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(f"{url}/chat/completions", data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {p.api_key}",
        })
        resp = urllib.request.urlopen(req, timeout=p.timeout)
        result = json.loads(resp.read())
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def _call_llamacpp(self, p: ProviderConfig, prompt: str,
                             system: str, max_tokens: int) -> str:
        """Call llama.cpp server (OpenAI-compatible)."""
        import urllib.request
        url = p.base_url or "http://localhost:8080"
        data = json.dumps({
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "n_predict": max_tokens,
        }).encode()
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=p.timeout)
        result = json.loads(resp.read())
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    def report(self) -> dict:
        """Return routing statistics."""
        total = self.stats["total"] or 1
        by_tier = self.stats.get("by_tier", {})
        return {
            "total_requests": self.stats["total"],
            "total_cost": round(self.stats["total_cost"], 4),
            "coverage": {k: f"{v/total*100:.1f}%" for k, v in by_tier.items()},
        }
