import json
import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


class State(TypedDict):
    user_input: str
    action: str
    action_input: str
    tool_result: str
    answer: str


llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
)


def calculator(expression: str) -> str:
    """
    教学用计算器。
    为了安全，只允许数字和基础数学符号。
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression) <= allowed_chars:
            return "表达式包含非法字符"

        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"计算失败：{e}"


def decide_node(state: State) -> State:
    prompt = """
你是一个简单 Agent。

你只能选择两种动作：

1. calculator：当用户需要数学计算时使用
2. final：当你可以直接回答时使用

你必须只输出 JSON，不要输出 Markdown，不要输出代码块，不要输出解释。

格式如下：
{
  "action": "calculator 或 final",
  "action_input": "如果 action 是 calculator，这里放数学表达式；如果 action 是 final，这里放最终回答"
}

要求：
- 如果用户要求计算，请把纯数学表达式放到 action_input 中。
- 例如用户说“帮我计算 23 * 17 + 8”，你应该输出：
{
  "action": "calculator",
  "action_input": "23 * 17 + 8"
}
"""

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_input"])
    ])

    raw_text = response.content.strip()

    try:
        data = json.loads(raw_text)
        action = data.get("action", "final")
        action_input = data.get("action_input", "我没有正确理解任务。")

        if action not in ["calculator", "final"]:
            action = "final"
            action_input = "我没有正确理解任务。"

    except Exception:
        action = "final"
        action_input = f"模型没有返回合法 JSON，原始输出是：{raw_text}"

    return {
        **state,
        "action": action,
        "action_input": action_input,
    }


def route_after_decide(state: State) -> Literal["calculator", "final"]:
    if state["action"] == "calculator":
        return "calculator"
    return "final"


def calculator_node(state: State) -> State:
    result = calculator(state["action_input"])

    return {
        **state,
        "tool_result": result,
    }


def final_node(state: State) -> State:
    if state["action"] == "calculator":
        answer = f"计算结果是：{state['tool_result']}"
    else:
        answer = state["action_input"]

    return {
        **state,
        "answer": answer,
    }


graph_builder = StateGraph(State)

graph_builder.add_node("decide", decide_node)
graph_builder.add_node("calculator", calculator_node)
graph_builder.add_node("final", final_node)

graph_builder.add_edge(START, "decide")

graph_builder.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "calculator": "calculator",
        "final": "final",
    },
)

graph_builder.add_edge("calculator", "final")
graph_builder.add_edge("final", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke({
        "user_input": "你是谁",
        "action": "",
        "action_input": "",
        "tool_result": "",
        "answer": "",
    })

    print(result["answer"])