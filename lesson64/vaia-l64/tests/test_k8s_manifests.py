"""
L64 – Kubernetes manifest validation.
Parses YAML without a cluster; catches common misconfigurations.
"""
import pathlib
import yaml
import pytest


def load_manifest(filename: str) -> dict:
    path = pathlib.Path(f"k8s/{filename}")
    return yaml.safe_load(path.read_text())


def test_deployment_rolling_update_zero_unavailable():
    d = load_manifest("deployment.yaml")
    strategy = d["spec"]["strategy"]["rollingUpdate"]
    assert strategy["maxUnavailable"] == 0, "maxUnavailable must be 0 for zero-downtime"


def test_deployment_has_readiness_probe():
    d = load_manifest("deployment.yaml")
    container = d["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in container, "Deployment must declare readinessProbe"


def test_deployment_has_liveness_probe():
    d = load_manifest("deployment.yaml")
    container = d["spec"]["template"]["spec"]["containers"][0]
    assert "livenessProbe" in container


def test_deployment_resource_limits():
    d = load_manifest("deployment.yaml")
    res = d["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert "limits" in res, "Must set resource limits"
    assert "requests" in res, "Must set resource requests"


def test_hpa_min_replicas():
    h = load_manifest("hpa.yaml")
    assert h["spec"]["minReplicas"] >= 2, "HPA minReplicas must be >= 2"


def test_hpa_cpu_target():
    h = load_manifest("hpa.yaml")
    metrics = h["spec"]["metrics"]
    cpu_metric = next((m for m in metrics if m["resource"]["name"] == "cpu"), None)
    assert cpu_metric is not None
    assert cpu_metric["resource"]["target"]["averageUtilization"] == 60


def test_non_root_security_context():
    d = load_manifest("deployment.yaml")
    sc = d["spec"]["template"]["spec"].get("securityContext", {})
    assert sc.get("runAsNonRoot") is True, "Pod must run as non-root"
