"""
L64 – Dockerfile structural tests.
Verify the Dockerfile follows production hardening patterns.
"""
import re
import pathlib
import pytest

DOCKERFILE = pathlib.Path("Dockerfile").read_text()


def test_multi_stage_builder_stage():
    assert "AS builder" in DOCKERFILE, "Must have builder stage"


def test_multi_stage_runtime_stage():
    assert "AS runtime" in DOCKERFILE, "Must have runtime stage"


def test_dev_stage_present():
    assert "AS dev" in DOCKERFILE, "Must have dev stage (assignment)"


def test_non_root_user():
    assert "USER vaia" in DOCKERFILE or re.search(r"USER \d+", DOCKERFILE), \
        "Must run as non-root user"


def test_healthcheck_present():
    assert "HEALTHCHECK" in DOCKERFILE, "Must declare HEALTHCHECK instruction"


def test_copy_from_builder():
    assert "COPY --from=builder" in DOCKERFILE, \
        "Runtime stage must copy only from builder"


def test_no_apt_cache_in_runtime():
    # The runtime stage should not call apt-get
    runtime_section = DOCKERFILE.split("AS runtime")[1] if "AS runtime" in DOCKERFILE else ""
    assert "apt-get install" not in runtime_section, \
        "Runtime stage must not install OS packages"
