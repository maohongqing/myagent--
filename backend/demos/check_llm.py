from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myagent.backend.app.agents.llm import friendly_llm_error, get_chat_model
from myagent.backend.app.config import Settings


def mask_secret(value: str | None) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


async def main() -> int:
    env_file = BACKEND_DIR / ".env"
    settings = Settings(_env_file=env_file) if env_file.exists() else Settings(_env_file=None)
    print("LLM config")
    print(f"- env_file: {env_file}")
    print(f"- model: {settings.openai_model}")
    print(f"- base_url: {settings.openai_base_url or '<default OpenAI>'}")
    print(f"- api_key: {mask_secret(settings.openai_api_key)}")
    print(f"- trust_env: {settings.openai_trust_env}")
    print(f"- timeout_seconds: {settings.openai_timeout_seconds}")

    model = get_chat_model(settings)
    if model is None:
        print("\nFAILED: model is not configured. Check OPENAI_API_KEY and langchain_openai installation.")
        return 1

    try:
        response = await model.ainvoke(
            [
                ("system", "You are a connectivity test. Reply with valid, concise Chinese."),
                ("user", "请只回复：模型连接成功"),
            ]
        )
    except Exception as exc:
        print(f"\nFAILED: {friendly_llm_error(exc)}")
        return 1

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    print("\nSUCCESS: model responded")
    print(str(content).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
