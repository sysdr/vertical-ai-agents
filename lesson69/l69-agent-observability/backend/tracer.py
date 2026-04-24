"""
L69 VAIA: OpenTelemetry TracerProvider with SQLite WAL span exporter.
Exports to L70: metric_snapshots table with threshold/breached columns.
"""
import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SpanExporter, SpanExportResult, BatchSpanProcessor
)

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "traces.db")
_lock = threading.Lock()


def init_db(db_path: str = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id        TEXT PRIMARY KEY,
                start_time      INTEGER,
                end_time        INTEGER,
                duration_ms     REAL DEFAULT 0.0,
                span_count      INTEGER DEFAULT 0,
                total_cost_usd  REAL DEFAULT 0.0,
                input_tokens    INTEGER DEFAULT 0,
                output_tokens   INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'OK',
                root_name       TEXT,
                created_at      INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE TABLE IF NOT EXISTS spans (
                span_id         TEXT PRIMARY KEY,
                trace_id        TEXT NOT NULL,
                parent_span_id  TEXT,
                name            TEXT NOT NULL,
                kind            TEXT DEFAULT 'INTERNAL',
                start_time      INTEGER,
                end_time        INTEGER,
                duration_ms     REAL DEFAULT 0.0,
                status          TEXT DEFAULT 'OK',
                attributes      TEXT DEFAULT '{}',
                events          TEXT DEFAULT '[]',
                created_at      INTEGER DEFAULT (strftime('%s','now')),
                FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
            );

            -- Consumed by L70 alerting engine
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                metric      TEXT NOT NULL,
                value       REAL NOT NULL,
                threshold   REAL,
                breached    INTEGER DEFAULT 0,
                trace_id    TEXT,
                created_at  INTEGER DEFAULT (strftime('%s','now'))
            );

            CREATE INDEX IF NOT EXISTS idx_spans_trace  ON spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_traces_time  ON traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_metrics_time ON metric_snapshots(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_metrics_breached ON metric_snapshots(breached, created_at DESC);
        """)


class SQLiteSpanExporter(SpanExporter):
    """
    Custom OTEL SpanExporter persisting spans to SQLite WAL.
    Thread-safe via a single write lock; uses INSERT OR REPLACE + upsert
    to accumulate per-trace aggregates incrementally.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db(db_path)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sid(n: int) -> str:
        return format(n, '016x')

    @staticmethod
    def _tid(n: int) -> str:
        return format(n, '032x')

    # ── export ───────────────────────────────────────────────────────────────

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with _lock, sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                for span in spans:
                    self._write_span(conn, span)
                    self._upsert_trace(conn, span)
            return SpanExportResult.SUCCESS
        except Exception as exc:
            print(f"[SQLiteSpanExporter] export error: {exc}")
            return SpanExportResult.FAILURE

    def _write_span(self, conn: sqlite3.Connection, span: ReadableSpan) -> None:
        span_id  = self._sid(span.context.span_id)
        trace_id = self._tid(span.context.trace_id)
        parent   = self._sid(span.parent.span_id) if span.parent else None
        start_ns = span.start_time or 0
        end_ns   = span.end_time   or 0
        dur_ms   = (end_ns - start_ns) / 1_000_000

        attrs  = dict(span.attributes) if span.attributes else {}
        events = [
            {"name": e.name, "ts": e.timestamp, "attrs": dict(e.attributes or {})}
            for e in span.events
        ]
        conn.execute("""
            INSERT OR REPLACE INTO spans
              (span_id, trace_id, parent_span_id, name, kind,
               start_time, end_time, duration_ms, status, attributes, events)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            span_id, trace_id, parent,
            span.name, span.kind.name,
            start_ns, end_ns, round(dur_ms, 3),
            span.status.status_code.name,
            json.dumps(attrs), json.dumps(events)
        ))

    def _upsert_trace(self, conn: sqlite3.Connection, span: ReadableSpan) -> None:
        trace_id = self._tid(span.context.trace_id)
        attrs    = dict(span.attributes) if span.attributes else {}
        cost     = float(attrs.get("llm.cost_usd", 0.0))
        in_tok   = int(attrs.get("llm.input_tokens", 0))
        out_tok  = int(attrs.get("llm.output_tokens", 0))
        start_ns = span.start_time or 0
        end_ns   = span.end_time   or 0

        conn.execute("""
            INSERT INTO traces
              (trace_id, start_time, end_time, duration_ms, span_count,
               total_cost_usd, input_tokens, output_tokens, status, root_name)
            VALUES (?,?,?,?,1,?,?,?,?,?)
            ON CONFLICT(trace_id) DO UPDATE SET
                end_time       = MAX(end_time, excluded.end_time),
                duration_ms    = (MAX(end_time, excluded.end_time) - start_time) / 1000000.0,
                span_count     = span_count + 1,
                total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                input_tokens   = input_tokens + excluded.input_tokens,
                output_tokens  = output_tokens + excluded.output_tokens,
                status = CASE WHEN excluded.status='ERROR' THEN 'ERROR' ELSE status END
        """, (
            trace_id, start_ns, end_ns,
            round((end_ns - start_ns) / 1_000_000, 3),
            round(cost, 8), in_tok, out_tok,
            span.status.status_code.name, span.name
        ))

    def shutdown(self) -> None:
        pass


def setup_tracer(service_name: str = "vaia-agent-l69") -> trace.Tracer:
    """Configure global TracerProvider backed by SQLiteSpanExporter."""
    resource = Resource.create({
        "service.name":             service_name,
        "service.version":          "1.0.0",
        "deployment.environment":   "production",
        "telemetry.sdk.language":   "python",
    })
    provider  = TracerProvider(resource=resource)
    exporter  = SQLiteSpanExporter(DB_PATH)
    processor = BatchSpanProcessor(
        exporter,
        max_export_batch_size=32,
        schedule_delay_millis=500,
        export_timeout_millis=5000,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name, "1.0.0")


# Module-level singleton — imported by agent.py and main.py
tracer: trace.Tracer = setup_tracer()
