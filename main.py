import json
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Literal

CONTEXT_LIMIT = 8192
MEMORY_RECENT_TURNS = 12
ANSWER_MEMORY_CHARS = 1200

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from llm_client import get_llm
from rag import build_or_update_index, semantic_search
from tools import (
    calculator,
    list_dir,
    read_file,
    read_file_chunk,
    read_multiple_files,
    search_file,
    regex_search,
    find_files,
    get_file_info,
    project_overview,
    code_stats,
    todo_report,
    python_symbols,
    dependency_report,
    compare_files,
    preview_replace_in_file,
    write_file,
    replace_in_file,
    append_file,
)


class State(TypedDict):
    root_dir: str
    user_input: str
    conversation_history: list[dict]

    action: str
    action_input: dict
    tool_result: str

    steps: list[dict]
    tool_signatures: list[str]

    step_count: int
    max_steps: int
    should_stop: bool

    pending_approval: bool
    approved: bool

    answer: str


_llm = None


def get_model():
    global _llm

    if _llm is None:
        _llm = get_llm()

    return _llm


READ_ACTIONS = {
    "calculator",
    "list_dir",
    "read_file",
    "read_file_chunk",
    "read_multiple_files",
    "index_directory",
    "semantic_search",
    "search_file",
    "regex_search",
    "find_files",
    "get_file_info",
    "project_overview",
    "code_stats",
    "todo_report",
    "python_symbols",
    "dependency_report",
    "compare_files",
    "preview_replace_in_file",
}

WRITE_ACTIONS = {
    "write_file",
    "replace_in_file",
    "append_file",
}

ALLOWED_ACTIONS = READ_ACTIONS | WRITE_ACTIONS | {"final"}


TOOL_SPECS = {
    "calculator": {
        "description": "安全数学计算，支持 + - * / // % ** 和括号。",
        "example": {"expression": "23 * 17 + 8"},
        "func": lambda root_dir, **kwargs: calculator(**kwargs),
    },
    "list_dir": {
        "description": "列出目录直属内容。",
        "example": {"path": "."},
        "func": list_dir,
    },
    "read_file": {
        "description": "读取单个文件，支持常见文本、PDF、DOCX。",
        "example": {"path": "README.md"},
        "func": read_file,
    },
    "read_file_chunk": {
        "description": "按行号读取大文本文件的一段内容。",
        "example": {"path": "main.py", "start_line": 1, "line_count": 120},
        "func": read_file_chunk,
    },
    "read_multiple_files": {
        "description": "一次读取多个文件后综合分析。",
        "example": {"paths": ["README.md", "pyproject.toml", "src/main.py"]},
        "func": read_multiple_files,
    },
    "index_directory": {
        "description": "建立当前 root_dir 下指定目录的本地知识库向量索引。",
        "example": {"path": ".", "max_files": 200},
        "func": build_or_update_index,
    },
    "semantic_search": {
        "description": "在已经建立的本地知识库索引中做语义检索，适合跨文件、跨文档、总结性问答。",
        "example": {"query": "Agent 和 LangGraph 的关系", "k": 5},
        "func": semantic_search,
    },
    "search_file": {
        "description": "在文件内容中搜索关键词。",
        "example": {"keyword": "FastAPI", "path": "."},
        "func": search_file,
    },
    "regex_search": {
        "description": "使用正则表达式搜索纯文本文件内容。",
        "example": {"pattern": "class\\s+\\w+", "path": ".", "ignore_case": True},
        "func": regex_search,
    },
    "find_files": {
        "description": "按文件名模式查找文件。",
        "example": {"pattern": "*.py", "path": "."},
        "func": find_files,
    },
    "get_file_info": {
        "description": "查看文件或目录的基本信息。",
        "example": {"path": "src"},
        "func": get_file_info,
    },
    "project_overview": {
        "description": "生成项目结构概览、关键文件和扩展名分布。",
        "example": {"path": "."},
        "func": project_overview,
    },
    "code_stats": {
        "description": "统计代码文件数量、总行数、按扩展名行数和最大文件。",
        "example": {"path": "."},
        "func": code_stats,
    },
    "todo_report": {
        "description": "扫描 TODO、FIXME、BUG、HACK、XXX 等待办和风险标记。",
        "example": {"path": "."},
        "func": todo_report,
    },
    "python_symbols": {
        "description": "提取 Python 文件的顶层类、函数和 import，生成轻量代码索引。",
        "example": {"path": "."},
        "func": python_symbols,
    },
    "dependency_report": {
        "description": "解析 requirements.txt、pyproject.toml、package.json，生成依赖和脚本报告。",
        "example": {"path": "."},
        "func": dependency_report,
    },
    "compare_files": {
        "description": "对比两个纯文本文件，输出 unified diff。",
        "example": {"left_path": "old.py", "right_path": "new.py"},
        "func": compare_files,
    },
    "preview_replace_in_file": {
        "description": "预览纯文本替换会产生的 diff，不实际修改文件。",
        "example": {"path": "README.md", "old_text": "旧内容", "new_text": "新内容"},
        "func": preview_replace_in_file,
    },
    "write_file": {
        "description": "写入新文件或覆盖文件，支持纯文本和 DOCX。高风险，需要用户确认。",
        "example": {"path": "summary.md", "content": "文件内容", "overwrite": False},
        "func": write_file,
    },
    "replace_in_file": {
        "description": "替换纯文本或 DOCX 文件中的内容。高风险，需要用户确认。",
        "example": {"path": "README.md", "old_text": "旧内容", "new_text": "新内容"},
        "func": replace_in_file,
    },
    "append_file": {
        "description": "向纯文本或 DOCX 文件追加内容。高风险，需要用户确认。",
        "example": {"path": "notes.md", "content": "\\n追加内容"},
        "func": append_file,
    },
}


def extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise

        return json.loads(text[start:end + 1])


def format_tool_docs() -> str:
    blocks = []

    for index, (name, spec) in enumerate(TOOL_SPECS.items(), start=1):
        risk = "写操作，需要用户确认。" if name in WRITE_ACTIONS else "读操作或计算。"
        blocks.append(
            f"{index}. {name}\n"
            f"用途：{spec['description']}\n"
            f"风险：{risk}\n"
            f"参数示例：{json.dumps(spec['example'], ensure_ascii=False)}"
        )

    blocks.append(
        f"{len(TOOL_SPECS) + 1}. final\n"
        "用途：当你已经获得足够信息，可以回答用户问题时使用。\n"
        "参数示例：{\"answer\": \"你的最终回答\"}"
    )

    return "\n\n".join(blocks)


def trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    return text[-limit:]


def format_conversation_memory(history: list[dict], limit: int = CONTEXT_LIMIT) -> str:
    if not history:
        return "暂无历史对话。"

    blocks = []

    for item in history[-MEMORY_RECENT_TURNS:]:
        user = item.get("user", "").strip()
        assistant = item.get("assistant", "").strip()
        steps = item.get("steps", [])
        tool_names = [step.get("action", "") for step in steps if step.get("action")]
        tool_text = ", ".join(tool_names) if tool_names else "无工具调用"
        blocks.append(
            "用户：\n"
            f"{user}\n"
            "Agent：\n"
            f"{trim_text(assistant, ANSWER_MEMORY_CHARS)}\n"
            f"本轮工具：{tool_text}"
        )

    memory = "\n\n---\n\n".join(blocks)

    if len(memory) > limit:
        memory = memory[-limit:]
        return "[历史对话过长，已保留最近内容]\n" + memory

    return memory


def make_tool_signature(action: str, action_input: dict) -> str:
    """
    用于检测重复工具调用。
    """
    return json.dumps(
        {
            "action": action,
            "action_input": action_input,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def format_steps_for_prompt(steps: list[dict]) -> str:
    if not steps:
        return "暂无工具调用历史。"

    blocks = []

    for i, step in enumerate(steps, start=1):
        blocks.append(f"""
第 {i} 步：
工具：{step.get("action")}
参数：{json.dumps(step.get("action_input", {}), ensure_ascii=False)}
结果：
{step.get("tool_result", "")}
""".strip())

    return "\n\n".join(blocks)


def decide_node(state: State) -> State:
    if state["step_count"] >= state["max_steps"]:
        return {
            **state,
            "action": "final",
            "action_input": {
                "answer": "已达到最大工具调用次数，需要基于已有信息生成最终回答。"
            },
            "should_stop": True,
        }

    steps_text = format_steps_for_prompt(state["steps"])
    memory_text = format_conversation_memory(state["conversation_history"])

    prompt = f"""
你是一个多步本地文件分析 Agent。

用户已经指定了一个你可以访问的本地根目录 root_dir。
你不能访问 root_dir 之外的文件。
所有 path 都必须是相对于 root_dir 的路径，不要输出绝对路径。

你可以选择以下动作：

{format_tool_docs()}

你必须只输出 JSON，不要输出 Markdown，不要输出代码块，不要输出解释。

输出格式：
{{
  "action": "{' | '.join([*TOOL_SPECS.keys(), 'final'])}",
  "action_input": {{
    ...
  }}
}}

关键规则：
- 不要重复调用完全相同的工具和参数。
- 如果用户询问项目整体情况，优先 project_overview，再按需要 code_stats、read_multiple_files。
- index_directory 用于建立当前目录的本地知识库索引。当用户要求“建立索引”“初始化知识库”“索引这个目录”“为这个目录建立知识库”时使用。
- semantic_search 用于在已经建立的知识库索引中做语义检索。适合处理概念性、跨文件、跨文档、总结性、问答类问题。
- 如果用户明确要求建立索引，使用 index_directory。
- 如果用户问跨多个文件的问题，优先考虑 semantic_search。
- 如果用户问“根据整个目录”“根据知识库”“根据这些资料”来回答，优先考虑 semantic_search。
- 如果用户尚未建立索引，而问题明显需要跨文档问答，可以先 index_directory，再 semantic_search。
- 如果 semantic_search 没有结果，可以退回关键词搜索、文件查找或文件读取工具。
- semantic_search 返回的 source 必须在最终回答中保留。
- 如果语义检索结果足够回答，应进入 final，不要继续无意义调用工具。
- 不要重复调用完全相同参数的 index_directory 或 semantic_search。
- 如果用户要求理解 Python 代码结构，优先 python_symbols，再按需要 read_file_chunk。
- 如果用户要求分析依赖、启动方式或技术栈，优先 dependency_report。
- 如果用户要求查找待办、风险点或技术债，优先 todo_report 和 regex_search。
- 如果用户要求修改纯文本文件，通常先 preview_replace_in_file 给出变更预览，再执行 replace_in_file。
- 如果文件很长，优先 read_file_chunk 分段读取，不要一次读取过多无关内容。
- 如果用户要求综合分析多个文件，优先使用 read_multiple_files。
- 如果用户问“这个目录是什么项目”，通常先 list_dir，再 find_files 查找 README、pyproject.toml、package.json、requirements.txt、*.py 等关键文件，然后 read_multiple_files。
- 如果用户要求修改、写入、追加文件，选择对应写工具，但不要伪造已修改结果。
- 如果已经获得足够信息，必须选择 final。
- 如果连续工具结果没有帮助，选择 final 并说明信息不足。
"""

    user_message = f"""
用户问题：
{state["user_input"]}

历史对话记忆：
{memory_text}

当前已调用工具次数：
{state["step_count"]}

最大工具调用次数：
{state["max_steps"]}

已有工具签名，禁止重复：
{json.dumps(state["tool_signatures"], ensure_ascii=False)}

历史工具调用记录：
{steps_text}

请决定下一步动作。
"""

    response = get_model().invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ])

    raw_text = response.content.strip()

    try:
        data = extract_json(raw_text)
        action = data.get("action", "final")
        action_input = data.get("action_input", {})

        if action not in ALLOWED_ACTIONS:
            action = "final"
            action_input = {"answer": f"模型输出了不支持的动作：{action}"}

    except Exception:
        action = "final"
        action_input = {"answer": f"模型没有返回合法 JSON，原始输出：{raw_text}"}

    if action != "final":
        signature = make_tool_signature(action, action_input)

        if signature in state["tool_signatures"]:
            return {
                **state,
                "action": "final",
                "action_input": {
                    "answer": "检测到模型试图重复调用相同工具和参数，已停止继续调用工具。"
                },
                "should_stop": True,
            }

    return {
        **state,
        "action": action,
        "action_input": action_input,
        "should_stop": action == "final",
        "pending_approval": action in WRITE_ACTIONS,
        "approved": False,
    }


def route_after_decide(state: State) -> Literal["approve", "tool", "final"]:
    if state["action"] == "final":
        return "final"

    if state["action"] in WRITE_ACTIONS:
        return "approve"

    return "tool"


def approve_node(state: State) -> State:
    """
    写入/修改类工具必须经过用户确认。
    当前是 CLI 版确认。
    后续可以换成 LangGraph interrupt。
    """
    print("\n[Approval Required]")
    print(f"即将执行高风险写操作：{state['action']}")
    print(f"参数：{json.dumps(state['action_input'], ensure_ascii=False, indent=2)}")
    user_confirm = input("是否确认执行？输入 yes 确认，其它任意输入取消：").strip().lower()

    if user_confirm == "yes":
        return {
            **state,
            "approved": True,
        }

    return {
        **state,
        "approved": False,
        "action": "final",
        "action_input": {
            "answer": "用户取消了写入/修改操作，未对文件进行任何更改。"
        },
        "should_stop": True,
    }


def route_after_approve(state: State) -> Literal["tool", "final"]:
    if state["approved"]:
        return "tool"

    return "final"


def tool_node(state: State) -> State:
    action = state["action"]
    action_input = state["action_input"]
    root_dir = state["root_dir"]

    try:
        if action in TOOL_SPECS:
            result = TOOL_SPECS[action]["func"](root_dir=root_dir, **action_input)
        else:
            result = f"未知工具：{action}"

    except Exception as e:
        result = f"工具执行失败：{e}"

    signature = make_tool_signature(action, action_input)

    new_step = {
        "step": state["step_count"] + 1,
        "action": action,
        "action_input": action_input,
        "tool_result": result,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    return {
        **state,
        "tool_result": result,
        "steps": state["steps"] + [new_step],
        "tool_signatures": state["tool_signatures"] + [signature],
        "step_count": state["step_count"] + 1,
        "pending_approval": False,
        "approved": False,
    }


def route_after_tool(state: State) -> Literal["decide", "final"]:
    if state["step_count"] >= state["max_steps"]:
        return "final"

    return "decide"


def final_node(state: State) -> State:
    steps_text = format_steps_for_prompt(state["steps"])
    memory_text = format_conversation_memory(state["conversation_history"])

    if not state["steps"] and state["action"] == "final":
        direct_answer = state["action_input"].get("answer", "我没有生成回答。")
        return {
            **state,
            "answer": direct_answer,
        }

    prompt = """
你是一个本地文件分析 Agent。

请根据用户问题和所有工具调用历史，生成最终回答。

要求：
- 回答必须基于工具结果，不要编造。
- 如果工具结果包含文件路径、文件名、行号，请保留。
- 如果读取了多个文件，请综合比较和归纳，而不是逐字复述。
- 如果工具结果显示执行了写入、替换或追加，请明确说明变更文件、操作内容、是否创建备份。
- 如果使用了 semantic_search，回答必须基于检索结果。
- 如果使用了 semantic_search，必须引用检索结果中的 source。
- 如果多个 source 支持同一结论，可以合并总结并列出来源。
- 如果检索结果不足以回答，应明确说明信息不足。
- 不要编造 source 中没有的信息。
- 不要把 score 当成事实依据，只作为检索相关性信号。
- 如果信息不足，要明确说明还缺少什么。
- 回答用中文。
"""

    response = get_model().invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"""
用户指定的根目录：
{state["root_dir"]}

用户问题：
{state["user_input"]}

历史对话记忆：
{memory_text}

工具调用历史：
{steps_text}

decide 节点最后给出的信息：
{json.dumps(state["action_input"], ensure_ascii=False)}
""")
    ])

    return {
        **state,
        "answer": response.content,
    }


graph_builder = StateGraph(State)

graph_builder.add_node("decide", decide_node)
graph_builder.add_node("approve", approve_node)
graph_builder.add_node("tool", tool_node)
graph_builder.add_node("final", final_node)

graph_builder.add_edge(START, "decide")

graph_builder.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "approve": "approve",
        "tool": "tool",
        "final": "final",
    },
)

graph_builder.add_conditional_edges(
    "approve",
    route_after_approve,
    {
        "tool": "tool",
        "final": "final",
    },
)

graph_builder.add_conditional_edges(
    "tool",
    route_after_tool,
    {
        "decide": "decide",
        "final": "final",
    },
)

graph_builder.add_edge("final", END)

graph = graph_builder.compile()


def ask_root_dir() -> str:
    while True:
        root_dir = input("请输入允许 Agent 访问的本地目录路径：").strip()

        if not root_dir:
            print("路径不能为空。")
            continue

        path = Path(root_dir).expanduser().resolve()

        if not path.exists():
            print(f"路径不存在：{path}")
            continue

        if not path.is_dir():
            print(f"不是目录：{path}")
            continue

        print(f"Agent 当前只能访问该目录及其子目录：{path}")
        return str(path)


def print_trace(steps: list[dict]) -> None:
    if not steps:
        print("\n[Trace] 本轮没有调用工具。")
        return

    print("\n[Trace] 工具调用轨迹：")
    for step in steps:
        print(
            f"- Step {step['step']}: "
            f"{step['action']}("
            f"{json.dumps(step['action_input'], ensure_ascii=False)}"
            f")"
        )


def print_tools() -> None:
    print("\n[Tools]")
    for name, spec in TOOL_SPECS.items():
        risk = "write/approval" if name in WRITE_ACTIONS else "read"
        print(f"- {name} [{risk}]: {spec['description']}")


def save_session_log(root_dir: str, user_input: str, result: State, conversation_history: list[dict]) -> None:
    log_dir = Path(".agent_logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"session_{datetime.now().strftime('%Y%m%d')}.jsonl"

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": root_dir,
        "user_input": user_input,
        "steps": result.get("steps", []),
        "answer": result.get("answer", ""),
        "conversation_turns": len(conversation_history),
        "memory_snapshot": conversation_history[-MEMORY_RECENT_TURNS:],
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    root_dir = ask_root_dir()
    conversation_history = []
    print("输入 /tools 查看工具列表，输入 /memory 查看会话记忆，输入 exit、quit 或 q 退出。")

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in ["exit", "quit", "q", "/exit"]:
            break

        if user_input == "/tools":
            print_tools()
            continue

        if user_input == "/memory":
            print("\n[Memory]")
            print(format_conversation_memory(conversation_history))
            continue

        result = graph.invoke({
            "root_dir": root_dir,
            "user_input": user_input,
            "conversation_history": conversation_history,
            "action": "",
            "action_input": {},
            "tool_result": "",
            "steps": [],
            "tool_signatures": [],
            "step_count": 0,
            "max_steps": 10,
            "should_stop": False,
            "pending_approval": False,
            "approved": False,
            "answer": "",
        })

        print_trace(result["steps"])

        conversation_history.append({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": user_input,
            "assistant": result["answer"],
            "steps": result["steps"],
        })

        if len(conversation_history) > MEMORY_RECENT_TURNS:
            conversation_history = conversation_history[-MEMORY_RECENT_TURNS:]

        save_session_log(root_dir, user_input, result, conversation_history)

        print("\nAgent：")
        print(result["answer"])
