"""Custom metrics endpoint for monitoring."""
import time
from collections import defaultdict

# In-memory metrics store
_metrics = defaultdict(list)
_start_time = time.time()


def record_prediction(probability: float, risk_level: str):
    """Record a prediction for monitoring."""
    _metrics["predictions"].append({
        "probability": probability,
        "risk_level": risk_level,
        "timestamp": time.time(),
    })


def record_latency(endpoint: str, duration_ms: float):
    """Record endpoint latency."""
    _metrics["latency"].append({
        "endpoint": endpoint,
        "duration_ms": duration_ms,
        "timestamp": time.time(),
    })


def get_metrics() -> dict:
    """Return current metrics summary."""
    predictions = _metrics["predictions"]
    latencies = _metrics["latency"]

    total_preds = len(predictions)
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for p in predictions:
        risk_counts[p["risk_level"]] = risk_counts.get(p["risk_level"], 0) + 1

    avg_latency = 0
    if latencies:
        avg_latency = sum(l["duration_ms"] for l in latencies) / len(latencies)

    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_predictions": total_preds,
        "risk_distribution": risk_counts,
        "avg_latency_ms": round(avg_latency, 1),
        "total_requests": len(latencies),
    }
