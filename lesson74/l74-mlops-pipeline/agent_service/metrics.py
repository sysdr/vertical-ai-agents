from prometheus_client import Counter, Histogram, Gauge, Info

REQUESTS_TOTAL = Counter(
    "vaia_requests_total",
    "Total requests",
    ["variant", "status", "endpoint"],
)

LLM_LATENCY = Histogram(
    "vaia_llm_latency_seconds",
    "LLM call latency in seconds",
    ["variant"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

DRIFT_SCORE = Gauge("vaia_drift_score", "Current drift score", ["variant"])

ERROR_RATE = Gauge("vaia_error_rate", "Current error rate (rolling window)", ["variant"])

DEPLOYMENT_STATE = Gauge(
    "vaia_deployment_state",
    "Deployment state enum (0=PENDING 1=DEPLOYING 2=HEALTHY 3=DEGRADED 4=ROLLING_BACK 5=FAILED)",
)

CIRCUIT_BREAKER_OPEN = Gauge(
    "vaia_circuit_breaker_open",
    "1 if circuit breaker is open, 0 otherwise",
)

ACTIVE_CONNECTIONS = Gauge("vaia_active_connections", "Active request connections")

BUILD_INFO = Info("vaia_build", "Build metadata")
BUILD_INFO.info({"version": "74.0.1", "lesson": "L74", "model": "gemini-2.0-flash"})
