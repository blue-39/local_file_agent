from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    answer: str


def greet_node(state: State) -> State:
    user_input = state["user_input"]

    return {
        "user_input": user_input,
        "answer": f"你好，你刚才说的是：{user_input}"
    }


graph_builder = StateGraph(State)

graph_builder.add_node("greet", greet_node)

graph_builder.add_edge(START, "greet")
graph_builder.add_edge("greet", END)

graph = graph_builder.compile()

result = graph.invoke({
    "user_input": "我想学习 Agent",
    "answer": ""
})

print(result["answer"])