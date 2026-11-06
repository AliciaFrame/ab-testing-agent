"""HTTP tool server + endpoint grader for agentic RFT (Starlette ASGI app).

Contract (as used by the Foundry agentic-RFT trainer):

* ``POST /tools/{name}`` receives ``{"type": "function_call", "call_id", "id"?, "name"?, "arguments": "<json>",
  "trace_id"}`` and returns ``{"type": "function_call_output", "call_id", "id", "output": "<json string>"}``. Tool
  problems (unknown tool, invalid arguments) come back with HTTP 200 as the same ``{"error": ...}`` text the agent
  returns, so the model sees what it would in production and the trainer's per-tool failure limit is never tripped.
  Only infrastructure failures return 5xx.
* ``POST /grade/{version}`` (``acc``, ``eff`` or ``turns``; ``/grade`` = ``acc``) receives ``{"sample", "item",
  "trace_id"}`` and returns ``{"score": <float in [0, 1]>}``. Details are kept per rollout at ``GET /grades/{trace_id}``.
  ``turns`` also uses ``reference_gap``: the 25th-percentile gap between consecutive calls of all rollouts recorded in
  the last 15 minutes (the trainer's in-batch call spacing under the current load), cached for a minute.

The archive tools are stateless, so no per-rollout scenario binding is needed. Calls are recorded per ``trace_id``
in SQLite (shared by the worker processes of the single replica) so the grader scores the exact outputs the model
saw. All routes except ``/health`` require ``Authorization: Bearer <RFT_ENDPOINT_TOKEN>`` (fail closed when unset).

    hypercorn rft.tool_server:app --bind 0.0.0.0:8000 --workers 4
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import math
import os
import sqlite3
import threading
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rft import grader
from rft.common import TOOL_NAMES

log = logging.getLogger("rft.tool_server")
MAX_BYTES = 1_000_000
TIMEOUT_SECONDS = float(os.getenv("RFT_TIMEOUT_SECONDS", "60"))
_CONCURRENCY = asyncio.Semaphore(int(os.getenv("RFT_MAX_CONCURRENCY", "32")))
VERSIONS = grader.VERSIONS


class HTTPError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path, self._local = path, threading.local()
        db = self._db()
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS calls (trace_id TEXT NOT NULL, call_id TEXT NOT NULL, seq INTEGER NOT NULL,
                name TEXT NOT NULL, arguments TEXT NOT NULL, output TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY (trace_id, call_id));
            CREATE TABLE IF NOT EXISTS grades (trace_id TEXT NOT NULL, created REAL NOT NULL, case_id TEXT,
                version TEXT, score REAL NOT NULL, details TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS errors (created REAL NOT NULL, route TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shapes (created REAL NOT NULL, detail TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS calls_created ON calls (created);
        """)

    def _db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            db.execute("PRAGMA busy_timeout=30000")
            self._local.db = db
        return db

    def count(self, name: str) -> None:
        self._db().execute("INSERT INTO counters VALUES (?, 1) ON CONFLICT(name) DO UPDATE SET value = value + 1",
                           (name,))

    def record(self, trace_id: str, call_id: str, name: str, arguments, output: str) -> None:
        db = self._db()
        seq = db.execute("SELECT COUNT(*) FROM calls WHERE trace_id = ?", (trace_id,)).fetchone()[0]
        db.execute("INSERT OR IGNORE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (trace_id, call_id, seq, name, json.dumps(arguments), output, time.time()))

    def calls(self, trace_id: str) -> list[dict]:
        rows = self._db().execute("SELECT call_id, name, arguments, output, created FROM calls WHERE trace_id = ? "
                                  "ORDER BY seq, created", (trace_id,)).fetchall()
        return [{"call_id": r[0], "name": r[1], "arguments": json.loads(r[2]), "output": r[3], "created": r[4]}
                for r in rows]

    def reference_gap(self, window: float = 900.0, min_gaps: int = 50) -> float | None:
        """25th-percentile gap between consecutive calls of the same rollout over the last ``window`` seconds."""
        rows = self._db().execute("SELECT trace_id, created FROM calls WHERE created >= ? ORDER BY trace_id, seq",
                                  (time.time() - window,)).fetchall()
        gaps = [b[1] - a[1] for a, b in zip(rows, rows[1:]) if a[0] == b[0]]
        return sorted(gaps)[len(gaps) // 4] if len(gaps) >= min_gaps else None

    def save_grade(self, trace_id: str, case_id: str, version: str, score: float, details: dict) -> None:
        self._db().execute("INSERT INTO grades VALUES (?, ?, ?, ?, ?, ?)",
                           (trace_id, time.time(), case_id, version, score, json.dumps(details, default=str)))

    def grades(self, trace_id: str) -> list[dict]:
        rows = self._db().execute("SELECT created, case_id, version, score, details FROM grades WHERE trace_id = ? "
                                  "ORDER BY created", (trace_id,)).fetchall()
        return [{"created": r[0], "case_id": r[1], "version": r[2], "score": r[3], "details": json.loads(r[4])}
                for r in rows]

    def counters(self) -> dict:
        return dict(self._db().execute("SELECT name, value FROM counters").fetchall())

    def save_error(self, route: str, detail: dict) -> None:
        self._db().execute("INSERT INTO errors VALUES (?, ?, ?)", (time.time(), route, json.dumps(detail, default=str)[:4000]))

    def errors(self) -> list[dict]:
        rows = self._db().execute("SELECT created, route, detail FROM errors ORDER BY created DESC LIMIT 50").fetchall()
        return [{"created": r[0], "route": r[1], "detail": json.loads(r[2])} for r in rows]

    def recent_grades(self, limit: int = 20) -> list[dict]:
        rows = self._db().execute("SELECT created, case_id, version, score, details FROM grades ORDER BY created DESC "
                                  "LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = json.loads(r[4])
            out.append({"created": r[0], "case_id": r[1], "version": r[2], "score": r[3],
                        "calls_source": d.get("calls_source"), "tool_calls": d.get("tool_calls"),
                        "strict_pass": d.get("strict_pass"), "sample_keys": d.get("sample_keys"),
                        "sample_shape": d.get("sample_shape")})
        return out

    def prune(self, ttl: float = 3 * 24 * 3600) -> None:
        cutoff = time.time() - ttl
        self._db().execute("DELETE FROM calls WHERE created < ?", (cutoff,))
        self._db().execute("DELETE FROM grades WHERE created < ?", (cutoff,))


_STORE: Store | None = None


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store(os.getenv("RFT_STATE_DB", str(Path(__file__).resolve().parent / "state" / "traces.db")))
    return _STORE


def _authorized(request: Request) -> None:
    token = os.getenv("RFT_ENDPOINT_TOKEN", "")
    if not token:
        raise HTTPError(503, "RFT_ENDPOINT_TOKEN is not configured")
    if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
        raise HTTPError(401, "Unauthorized")


async def _body(request: Request) -> dict:
    _authorized(request)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPError(413, "Payload exceeds 1 MB")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks) or b"{}")
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPError(400, "Invalid JSON") from error
    if not isinstance(value, dict):
        raise HTTPError(400, "Expected a JSON object")
    return value


def _handle(fn):
    async def wrapper(request: Request):
        try:
            return await fn(request)
        except HTTPError as error:
            return JSONResponse({"detail": error.detail}, status_code=error.status)
        except Exception:  # noqa: BLE001 - unexpected failures are infrastructure errors, never results
            log.exception("unhandled error on %s", request.url.path)
            return JSONResponse({"detail": "internal error"}, status_code=500)
    return wrapper


async def _bounded(fn, *args):
    async with _CONCURRENCY:
        try:
            return await asyncio.wait_for(asyncio.to_thread(fn, *args), TIMEOUT_SECONDS)
        except asyncio.TimeoutError as error:
            raise HTTPError(504, "execution timed out") from error


def run_tool(body: dict, name: str) -> dict:
    call_id = body.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise HTTPError(422, "call_id is required")
    arguments = grader._arguments(body.get("arguments", {}))
    output = grader.execute(name, arguments)  # unknown tools / bad arguments -> {"error": ...} with HTTP 200
    trace_id = body.get("trace_id") or ""
    if trace_id:
        store().record(trace_id, call_id, name, arguments, output)
    store().count("tool_calls")
    store().count(f"tool_{name}" if name in TOOL_NAMES else "tool_unknown")
    response = {"type": "function_call_output", "call_id": call_id, "id": body.get("id") or f"fc_{call_id}",
                "output": output}
    if len(json.dumps(response).encode()) > MAX_BYTES:
        raise HTTPError(413, "Tool output exceeds 1 MB")
    return response


def shape(value, depth: int = 0):
    """Structure of a trainer sample without its contents: types, lengths, keys and item ``type`` fields."""
    if isinstance(value, dict):
        if depth >= 3:
            return {"dict": len(value)}
        return {k: shape(v, depth + 1) for k, v in list(value.items())[:30]}
    if isinstance(value, list):
        types = sorted({str(v.get("type")) for v in value if isinstance(v, dict)})
        return {"list": len(value), "item_types": types[:10],
                "first": shape(value[0], depth + 1) if value and depth < 3 else None}
    if isinstance(value, str):
        return {"str": len(value)}
    return type(value).__name__


_REFERENCE = {"at": 0.0, "value": None}


def reference_gap() -> float | None:
    if time.time() - _REFERENCE["at"] > 60:
        _REFERENCE.update(at=time.time(), value=store().reference_gap())
    return _REFERENCE["value"]


def run_grade(body: dict, version: str) -> float:
    sample, item, trace_id = body.get("sample"), body.get("item"), body.get("trace_id") or ""
    if not isinstance(sample, dict) or not isinstance(item, dict):
        raise HTTPError(422, "sample and item objects are required")
    try:
        result = grader.grade(sample, item, store().calls(trace_id) if trace_id else None, version,
                              reference_gap() if version in grader.TURN_PARAMS else None)
    except grader.GradingError as error:
        store().count("grade_infra_error")
        # Structure only (keys and types), never contents, so the record is safe to inspect.
        store().save_error(f"grade/{version}", {
            "error": str(error), "item_keys": sorted(item), "sample_keys": sorted(sample),
            "mode": item.get("mode"), "case_type": type(item.get("case")).__name__, "has_trace": bool(trace_id)})
        raise HTTPError(500, f"grading infrastructure error: {error}") from error
    except Exception as error:
        store().count("grade_unexpected_error")
        store().save_error(f"grade/{version}", {"error": f"{type(error).__name__}: {error}", "item_keys": sorted(item),
                                                 "sample_keys": sorted(sample), "mode": item.get("mode")})
        raise
    score = result["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise HTTPError(500, "grader produced an invalid score")
    result["sample_keys"] = sorted(sample)
    result["sample_shape"] = shape(sample)
    store().save_grade(trace_id, item.get("case_id", ""), version, float(score), result)
    store().count(f"grades_{version}")
    if grader.output_tokens(sample) is not None:
        store().count("grades_with_usage")
    if result["calls_source"] == "sample_output_tools":
        store().count("grades_from_sample_calls")
    return float(score)


async def health(request: Request):
    return JSONResponse({"status": "ok", "graders": VERSIONS, "tools": TOOL_NAMES})


@_handle
async def tool(request: Request):
    body = await _body(request)
    name = request.path_params.get("name") or body.get("name")
    if not isinstance(name, str) or not name:
        raise HTTPError(422, "tool name is required")
    if body.get("name") not in (None, name):
        raise HTTPError(400, "Tool name does not match route")
    _sample_shape(request, body, name)
    return JSONResponse(await _bounded(run_tool, body, name))


SHAPE_LIMIT = 300
_SECRET_HEADERS = {"authorization", "cookie", "x-api-key", "api-key", "ocp-apim-subscription-key"}


def _sample_shape(request: Request, body: dict, name: str) -> None:
    """Request structure for the first calls (header names and short non-secret values, body keys) to look for
    any per-turn metadata the trainer sends. Tool arguments are not stored."""
    db = store()._db()
    if db.execute("SELECT count(*) FROM shapes").fetchone()[0] >= SHAPE_LIMIT:
        return
    headers = {k: (v[:120] if k.lower() not in _SECRET_HEADERS else "<redacted>") for k, v in request.headers.items()}
    extra = {k: (v if isinstance(v, (str, int, float, bool)) and len(str(v)) < 120 else type(v).__name__)
             for k, v in body.items() if k != "arguments"}
    db.execute("INSERT INTO shapes VALUES (?, ?)", (time.time(), json.dumps({"tool": name, "headers": headers,
                                                                           "body": extra})))


@_handle
async def shapes(request: Request):
    _authorized(request)
    rows = store()._db().execute("SELECT created, detail FROM shapes ORDER BY created").fetchall()
    return JSONResponse({"shapes": [{"created": r[0], **json.loads(r[1])} for r in rows]})


@_handle
async def grade(request: Request):
    body = await _body(request)
    version = request.path_params.get("version", "acc")
    if version not in VERSIONS:
        raise HTTPError(404, f"unknown reward {version!r}; available: {sorted(VERSIONS)}")
    return JSONResponse({"score": await _bounded(run_grade, body, version)})


@_handle
async def trace(request: Request):
    _authorized(request)
    return JSONResponse({"trace_id": request.path_params["trace_id"], "calls": store().calls(request.path_params["trace_id"])})


@_handle
async def grades(request: Request):
    _authorized(request)
    return JSONResponse({"trace_id": request.path_params["trace_id"], "grades": store().grades(request.path_params["trace_id"])})


@_handle
async def stats(request: Request):
    _authorized(request)
    return JSONResponse({"counters": store().counters(), "pid": os.getpid()})


@_handle
async def errors(request: Request):
    _authorized(request)
    return JSONResponse({"errors": store().errors()})


@_handle
async def recent(request: Request):
    _authorized(request)
    return JSONResponse({"grades": store().recent_grades()})


def _startup() -> None:
    from rft.common import domain

    domain.execute_tool("get_test", {"test_id": "warmup"})  # loads the archive before the first request
    store().prune()


async def _lifespan(app):
    await asyncio.to_thread(_startup)
    yield


app = Starlette(lifespan=contextlib.asynccontextmanager(_lifespan), routes=[
    Route("/health", health, methods=["GET"]),
    Route("/tools", tool, methods=["POST"]),
    Route("/tools/{name}", tool, methods=["POST"]),
    Route("/grade", grade, methods=["POST"]),
    Route("/grade/{version}", grade, methods=["POST"]),
    Route("/traces/{trace_id}", trace, methods=["GET"]),
    Route("/grades/{trace_id}", grades, methods=["GET"]),
    Route("/stats", stats, methods=["GET"]),
    Route("/errors", errors, methods=["GET"]),
    Route("/recent-grades", recent, methods=["GET"]),
    Route("/shapes", shapes, methods=["GET"]),
])
