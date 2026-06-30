from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    task_type: str
    tool_result: str
    answer: str


def calculator(expression: str) -> str:
    """
    一个非常简单的计算器。
    注意：真实项目不要直接 eval 用户输入。
    这里只是教学用。
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


def classify_node(state: State) -> State:
    user_input = state["user_input"]

    if any(op in user_input for op in ["+", "-", "*", "/"]):
        task_type = "math"
    else:
        task_type = "chat"

    return {
        **state,
        "task_type": task_type
    }


def math_tool_node(state: State) -> State:
    user_input = state["user_input"]

    # 为了简单，假设用户直接输入表达式，比如：3 + 5
    result = calculator(user_input)

    return {
        **state,
        "tool_result": result
    }


def chat_node(state: State) -> State:
    return {
        **state,
        "answer": "这是普通聊天任务，当前版本只支持简单计算。"
    }


def answer_node(state: State) -> State:
    return {
        **state,
        "answer": f"计算结果是：{state['tool_result']}"
    }


def route_after_classify(state: State) -> Literal["math", "chat"]:
    if state["task_type"] == "math":
        return "math"
    return "chat"


graph_builder = StateGraph(State)

graph_builder.add_node("classify", classify_node)
graph_builder.add_node("math_tool", math_tool_node)
graph_builder.add_node("chat", chat_node)
graph_builder.add_node("answer", answer_node)

graph_builder.add_edge(START, "classify")

graph_builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "math": "math_tool",
        "chat": "chat",
    }
)

graph_builder.add_edge("math_tool", "answer")
graph_builder.add_edge("answer", END)
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()

result = graph.invoke({
    "user_input": "3 + 5 / 2",
    "task_type": "",
    "tool_result": "",
    "answer": ""
})

print(result["answer"])