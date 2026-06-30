# Local File Agent

一个基于 LangGraph 和 DeepSeek 的本地文件分析 Agent。它可以在用户指定的目录沙箱内多步调用工具，完成项目结构分析、文件阅读、内容搜索、代码统计、文档生成和受控文件修改。

## 核心能力

- 多步 Agent 工作流：使用 LangGraph 编排 `decide -> approve -> tool -> final` 状态流。
- 多轮会话记忆：保留最近多轮问答和工具轨迹，支持承接上下文继续提问。
- 本地 RAG：使用 HuggingFace Embedding 和 Chroma，为用户授权目录建立本地知识库索引。
- 本地目录沙箱：所有路径都限制在用户指定的 `root_dir` 内，阻止 `../` 路径逃逸。
- 工具注册表：工具描述、示例参数和执行函数集中管理，新增工具更容易。
- 项目分析：支持项目概览、关键文件识别、代码行数统计、文件树节选。
- 代码理解：支持 Python 顶层类/函数/import 索引、依赖清单解析、技术栈辅助分析。
- 变更审查：支持文件 diff、替换预览，便于在真正写入前检查影响范围。
- 风险扫描：支持 TODO/FIXME/BUG/HACK 标记扫描和正则搜索。
- 大文件处理：支持按行分块读取，避免一次性把长文件塞进上下文。
- 安全写入：写入、替换、追加都需要用户确认；修改前自动备份。
- 合法 DOCX：创建和修改 Word 文档时使用 `python-docx`，避免生成损坏文件。
- 会话审计：每轮问题、工具调用轨迹和最终回答会写入 `.agent_logs/session_YYYYMMDD.jsonl`。

## 工具列表

- `calculator`：安全数学计算。
- `list_dir`：列出目录直属内容。
- `project_overview`：生成项目结构概览。
- `code_stats`：统计代码文件和行数。
- `python_symbols`：提取 Python 类、函数、import。
- `dependency_report`：解析 Python/Node 常见依赖文件。
- `todo_report`：扫描 TODO、FIXME、BUG、HACK 等标记。
- `find_files`：按文件名模式查找文件。
- `search_file`：搜索文件内容。
- `regex_search`：按正则表达式搜索内容。
- `read_file`：读取文本、PDF、DOCX。
- `read_file_chunk`：按行读取长文本文件片段。
- `read_multiple_files`：批量读取多个文件。
- `index_directory`：为当前目录建立本地知识库向量索引。
- `semantic_search`：在本地知识库索引中进行语义检索。
- `compare_files`：对比两个纯文本文件并输出 diff。
- `preview_replace_in_file`：预览替换操作的 diff。
- `write_file`：写入文本或 DOCX 文件。
- `replace_in_file`：替换文本或 DOCX 内容。
- `append_file`：追加文本或 DOCX 内容。

## 运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置 `.env`：

```env
DEEPSEEK_API_KEY=你的 key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# HuggingFace 下载慢时可使用镜像
# HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_SYMLINKS_WARNING=1
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

3. 启动：

```bash
python main.py
```

启动后先输入允许 Agent 访问的本地目录。运行中可以输入 `/tools` 查看工具列表，输入 `/memory` 查看当前会话记忆，输入 `exit` 退出。

## RAG 用法

本项目的 RAG 模块位于 `rag.py`，使用：

- `HuggingFaceEmbeddings`
- 默认模型：`BAAI/bge-small-zh-v1.5`
- 本地向量库：`Chroma`
- 默认持久化目录：`chroma_db`

示例任务：

- “为这个目录建立知识库索引”
- “根据知识库总结这个目录里的文档主要讲了什么”
- “这些文档里有没有提到 Agent 和 LangGraph 的关系？”
- “根据这个目录里的资料，帮我生成一份学习计划”

语义检索结果会保留 `source`，最终回答应引用来源文件。

首次建立索引时会下载 embedding 模型权重。如果 `model.safetensors` 长时间卡住，通常是 HuggingFace 网络连接慢。可以在 `.env` 中开启镜像：

```env
HF_ENDPOINT=https://hf-mirror.com
```

也可以提前把模型下载到本地，然后把 `RAG_EMBEDDING_MODEL` 改成本地模型目录路径。

## 简历描述参考

基于 LangGraph 实现本地文件分析 Agent，设计多步决策、工具调用、短期会话记忆、RAG 语义检索、人工审批和最终总结的状态机工作流；封装项目概览、代码统计、Python 符号索引、依赖解析、正则搜索、diff 预览、分块读取、DOCX 读写、Chroma 本地向量索引等工具，并通过目录沙箱、写操作审批、自动备份和 JSONL 会话日志提升安全性与可追溯性。
