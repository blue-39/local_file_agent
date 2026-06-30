from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    task_type: str
    answer: str


def classify_node(state: State) -> State:
    user_input = state["user_input"]

    if "算" in user_input or "+" in user_input or "*" in user_input:
        task_type = "math"
    else:
        task_type = "chat"

    return {
        **state,
        "task_type": task_type
    }


def math_node(state: State) -> State:
    return {
        **state,
        "answer": "我判断这是一个计算任务，但我现在还不会真正计算。"
    }


def chat_node(state: State) -> State:
    return {
        **state,
        "answer": "我判断这是一个普通聊天任务。"
    }


def route_after_classify(state: State) -> Literal["math", "chat"]:
    if state["task_type"] == "math":
        return "math"
    return "chat"


graph_builder = StateGraph(State)

graph_builder.add_node("classify", classify_node)
graph_builder.add_node("math", math_node)
graph_builder.add_node("chat", chat_node)

graph_builder.add_edge(START, "classify")

graph_builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "math": "math",
        "chat": "chat",
    }
)

graph_builder.add_edge("math", END)
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()

result = graph.invoke({
    "user_input": "帮我算一下 3 + 5",
    "task_type": "",
    "answer": ""
})

print(result["answer"])