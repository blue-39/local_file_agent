import json
import mimetypes
import queue
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from main import (
    MEMORY_RECENT_TURNS,
    WRITE_ACTIONS,
    State,
    decide_node,
    final_node,
    format_conversation_memory,
    print_tools,
    route_after_tool,
    save_session_log,
    tool_node,
)


HOST = "127.0.0.1"
PORT = 8000
STATIC_DIR = Path(__file__).parent / "web" / "static"
DEFAULT_ROOT = str(Path.cwd().resolve())

jobs: dict[str, "AgentJob"] = {}
conversation_history: list[dict] = []
history_lock = threading.Lock()


class AgentJob:
    def __init__(self, root_dir: str, user_input: str, max_steps: int = 10):
        self.id = uuid.uuid4().hex
        self.root_dir = root_dir
        self.user_input = user_input
        self.max_steps = max_steps
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.started_at = time.perf_counter()
        self.events: list[dict] = []
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.done = False
        self.status = "queued"
        self.answer = ""
        self.steps: list[dict] = []
        self.approval_event = threading.Event()
        self.approval_decision: bool | None = None
        self.pending_approval: dict | None = None

    def emit(self, event_type: str, **payload) -> None:
        event = {
            "id": len(self.events) + 1,
            "type": event_type,
            "job_id": self.id,
            "elapsed_ms": round((time.perf_counter() - self.started_at) * 1000),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        with self.lock:
            self.events.append(event)
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            for event in self.events:
                q.put(event)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def approve(self, approved: bool) -> bool:
        if not self.pending_approval or self.done:
            return False
        self.approval_decision = approved
        self.approval_event.set()
        return True


def make_initial_state(root_dir: str, user_input: str, max_steps: int) -> State:
    with history_lock:
        memory = list(conversation_history)

    return {
        "root_dir": root_dir,
        "user_input": user_input,
        "conversation_history": memory,
        "action": "",
        "action_input": {},
        "tool_result": "",
        "steps": [],
        "tool_signatures": [],
        "step_count": 0,
        "max_steps": max_steps,
        "should_stop": False,
        "pending_approval": False,
        "approved": False,
        "answer": "",
    }


def run_job(job: AgentJob) -> None:
    state = make_initial_state(job.root_dir, job.user_input, job.max_steps)
    job.status = "running"
    job.emit("job_started", root_dir=job.root_dir, user_input=job.user_input)

    try:
        while True:
            node_start = time.perf_counter()
            job.emit("node_started", node="decide", label="决策节点")
            state = decide_node(state)
            job.emit(
                "decision_made",
                node="decide",
                duration_ms=round((time.perf_counter() - node_start) * 1000),
                action=state.get("action"),
                action_input=state.get("action_input", {}),
                pending_approval=state.get("pending_approval", False),
            )

            action = state.get("action")
            if action == "final":
                break

            if action in WRITE_ACTIONS:
                job.pending_approval = {
                    "action": action,
                    "action_input": state.get("action_input", {}),
                }
                job.status = "waiting_approval"
                job.emit("approval_required", **job.pending_approval)
                job.approval_event.wait()
                approved = bool(job.approval_decision)
                job.emit("approval_resolved", approved=approved)
                job.pending_approval = None
                job.status = "running"

                if not approved:
                    state = {
                        **state,
                        "approved": False,
                        "action": "final",
                        "action_input": {
                            "answer": "用户取消了写入/修改操作，未对文件进行任何更改。"
                        },
                        "should_stop": True,
                    }
                    break

                state = {
                    **state,
                    "approved": True,
                }

            node_start = time.perf_counter()
            job.emit(
                "node_started",
                node="tool",
                label="工具执行",
                action=state.get("action"),
            )
            state = tool_node(state)
            latest_step = state["steps"][-1] if state.get("steps") else {}
            job.steps = state.get("steps", [])
            job.emit(
                "tool_finished",
                node="tool",
                duration_ms=round((time.perf_counter() - node_start) * 1000),
                step=latest_step,
                route=route_after_tool(state),
            )

            if route_after_tool(state) == "final":
                break

        node_start = time.perf_counter()
        job.emit("node_started", node="final", label="最终回答")
        state = final_node(state)
        job.answer = state.get("answer", "")
        job.steps = state.get("steps", [])
        job.status = "completed"
        job.done = True

        with history_lock:
            conversation_history.append(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "user": job.user_input,
                    "assistant": job.answer,
                    "steps": job.steps,
                }
            )
            if len(conversation_history) > MEMORY_RECENT_TURNS:
                del conversation_history[:-MEMORY_RECENT_TURNS]
            memory_snapshot = list(conversation_history)

        save_session_log(job.root_dir, job.user_input, state, memory_snapshot)
        job.emit(
            "job_completed",
            node="final",
            duration_ms=round((time.perf_counter() - node_start) * 1000),
            answer=job.answer,
            steps=job.steps,
            total_duration_ms=round((time.perf_counter() - job.started_at) * 1000),
        )

    except Exception as exc:
        job.status = "failed"
        job.done = True
        job.emit("job_failed", error=str(exc))


def json_response(handler: BaseHTTPRequestHandler, status: int, data: dict) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    body = handler.rfile.read(length).decode("utf-8")
    return json.loads(body or "{}")


class AgentWebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/events":
            self.handle_events(parsed)
            return
        if parsed.path == "/api/status":
            self.handle_status(parsed)
            return
        if parsed.path == "/api/memory":
            with history_lock:
                memory = format_conversation_memory(conversation_history)
            json_response(self, 200, {"memory": memory})
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self.handle_chat()
            return
        if parsed.path == "/api/approve":
            self.handle_approve()
            return
        json_response(self, 404, {"error": "Not found"})

    def handle_chat(self) -> None:
        data = read_json(self)
        root_dir = str(data.get("root_dir") or DEFAULT_ROOT).strip()
        user_input = str(data.get("message") or "").strip()
        max_steps = int(data.get("max_steps") or 10)

        if not user_input:
            json_response(self, 400, {"error": "message 不能为空"})
            return

        root = Path(root_dir).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            json_response(self, 400, {"error": f"无效根目录: {root_dir}"})
            return

        job = AgentJob(str(root), user_input, max_steps=max(1, min(max_steps, 20)))
        jobs[job.id] = job
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        json_response(self, 200, {"job_id": job.id, "status": job.status})

    def handle_approve(self) -> None:
        data = read_json(self)
        job = jobs.get(str(data.get("job_id") or ""))
        approved = bool(data.get("approved"))
        if not job:
            json_response(self, 404, {"error": "job 不存在"})
            return
        if not job.approve(approved):
            json_response(self, 409, {"error": "当前没有等待审批的操作"})
            return
        json_response(self, 200, {"ok": True})

    def handle_status(self, parsed) -> None:
        params = parse_qs(parsed.query)
        job = jobs.get((params.get("job_id") or [""])[0])
        if not job:
            json_response(self, 404, {"error": "job 不存在"})
            return
        json_response(
            self,
            200,
            {
                "job_id": job.id,
                "status": job.status,
                "done": job.done,
                "answer": job.answer,
                "steps": job.steps,
                "pending_approval": job.pending_approval,
            },
        )

    def handle_events(self, parsed) -> None:
        params = parse_qs(parsed.query)
        job = jobs.get((params.get("job_id") or [""])[0])
        if not job:
            json_response(self, 404, {"error": "job 不存在"})
            return

        q = job.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    payload = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"event: {event['type']}\n".encode("utf-8"))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if event["type"] in {"job_completed", "job_failed"}:
                        break
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        finally:
            job.unsubscribe(q)

    def serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        target = (STATIC_DIR / path.lstrip("/")).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return

        if not target.exists() or not target.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AgentWebHandler)
    print_tools()
    print(f"\nAgent Web UI: http://{HOST}:{PORT}")
    print(f"默认根目录: {DEFAULT_ROOT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
