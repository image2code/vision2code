from types import SimpleNamespace

import pytest

from vision2code.benchmark.run_benchmark import (
    DEFAULT_LOCAL_RATER_BASE_URL,
    DEFAULT_LOCAL_RATER_MODEL,
    model_ids_from_response,
    resolved_rater_base_url,
    validate_local_vllm_rater,
)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def list(self):
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


def test_resolved_local_rater_base_url_defaults():
    args = SimpleNamespace(rater_provider="local_vllm", rater_base_url="")
    assert resolved_rater_base_url(args) == DEFAULT_LOCAL_RATER_BASE_URL


def test_model_ids_from_object_and_mapping_payloads():
    response = SimpleNamespace(data=[SimpleNamespace(id="b"), {"id": "a"}, {"name": "ignored"}])
    assert model_ids_from_response(response) == ["a", "b"]


def test_validate_local_vllm_rater_accepts_served_model():
    client = FakeClient(response=SimpleNamespace(data=[SimpleNamespace(id=DEFAULT_LOCAL_RATER_MODEL)]))
    assert validate_local_vllm_rater(
        client,
        model=DEFAULT_LOCAL_RATER_MODEL,
        base_url=DEFAULT_LOCAL_RATER_BASE_URL,
    ) == [DEFAULT_LOCAL_RATER_MODEL]


def test_validate_local_vllm_rater_rejects_missing_model():
    client = FakeClient(response=SimpleNamespace(data=[SimpleNamespace(id="other-model")]))
    with pytest.raises(RuntimeError, match="not serving"):
        validate_local_vllm_rater(
            client,
            model=DEFAULT_LOCAL_RATER_MODEL,
            base_url=DEFAULT_LOCAL_RATER_BASE_URL,
        )


def test_validate_local_vllm_rater_rejects_unreachable_server():
    client = FakeClient(error=ConnectionError("down"))
    with pytest.raises(RuntimeError, match="not reachable"):
        validate_local_vllm_rater(
            client,
            model=DEFAULT_LOCAL_RATER_MODEL,
            base_url=DEFAULT_LOCAL_RATER_BASE_URL,
        )

