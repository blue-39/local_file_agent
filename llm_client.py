import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")

    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    except ImportError as e:
        if "socksio" in str(e).lower():
            raise RuntimeError(
                "当前环境启用了 SOCKS 代理，但缺少 socksio 依赖。"
                "请运行：pip install socksio，或重新执行：pip install -r requirements.txt"
            ) from e
        raise
