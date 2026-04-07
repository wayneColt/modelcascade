# ModelCascade

**Route local. Escalate smart. Never overspend.**

Open-source intelligent model router for agentic pipelines. 74% of requests handled locally at $0. Escalate to cloud only when stuck.

## Install

```bash
pip install modelcascade
```

```python
from modelcascade import CascadeRouter

router = CascadeRouter.from_config("mc.yaml")
result = await router.complete(prompt)
# → routed LOCAL · $0.000 · 47ms
```

## The Cascade

| Tier | Models | Cost/1K | Coverage |
|------|--------|---------|----------|
| **LOCAL** | Ollama, llama.cpp, vLLM | $0 | 74% |
| **FAST** | claude-haiku-4-5, Groq | $0.001 | +18% |
| **CAPABLE** | claude-sonnet-4-6, GPT-4o | $0.005 | +8% |

## Configure

```yaml
# mc.yaml
providers:
  local:
    type: ollama
    model: llama3.2:3b
    cost_per_1k: 0.0
  fast:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-haiku-4-5-20251001
    cost_per_1k: 0.001
  capable:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-sonnet-4-6
    cost_per_1k: 0.005

routing:
  cost_ceiling: 0.01
  cascade_on_failure: true
  calibration: preset_v1
```

## Principles

1. **Classify first, spend second** — Every request gets a difficulty score before a provider is chosen
2. **Fail cheap, succeed capable** — Lower tiers fail fast, escalation is automatic
3. **Your keys, your data** — BYOK, no telemetry, no vendor lock-in

## Production Numbers

- **$3/night** operating cost across 10K+ daily dispatches
- **74%** local coverage at $0
- **21/21** A/B calibration tests passed
- Works with **LangChain**, **CrewAI**, **Claude Code**, and custom pipelines

## License

MIT

---

*The arbitrage was always going to close. Route responsibly.*

Built by [WayneColt](https://github.com/wayneColt) | Research: [catalytic-computing.ai](https://catalytic-computing.ai) | Enterprise: [wayneia.com](https://wayneia.com)
