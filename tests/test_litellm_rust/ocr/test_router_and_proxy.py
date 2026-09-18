from collections.abc import Iterator
from typing import Final

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.proxy import proxy_server
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

MODEL_GROUP: Final = "ocr"
ROUTER_RETRIES: Final = 1


@pytest.fixture
def ocr_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    recording_server.expected_requests = None
    return recording_server


@pytest.fixture
def router(ocr_server: RecordingServer) -> Router:
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {"model": OCR_MODEL, "api_key": "test-key", "api_base": ocr_server.base_url},
            }
        ],
        num_retries=ROUTER_RETRIES,
        retry_after=0,
    )


@pytest.fixture
def proxy(router: Router, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "master_key", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    with TestClient(proxy_server.app, raise_server_exceptions=False) as client:
        yield client


def post_ocr(proxy: TestClient):
    return proxy.post("/v1/ocr", json={"model": MODEL_GROUP, "document": dict(OCR_DOCUMENT)})


@pytest.mark.asyncio
async def test_router_aocr_contract_rejected_credentials_are_not_retried_and_cool_the_deployment_down(
    ocr_server: RecordingServer, router: Router, ocr_backend: bool
) -> None:
    ocr_server.default_response = ResponseSpec(body={"message": "rejected"}, status=401)

    with pytest.raises(litellm.AuthenticationError):
        await router.aocr(model=MODEL_GROUP, document=dict(OCR_DOCUMENT))

    cooled_down: Final = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=router.get_model_ids(), parent_otel_span=None
    )
    assert len(ocr_server.requests) == 1
    assert len(cooled_down) == 1


def test_proxy_ocr_contract_rejected_credentials_return_401_without_retrying(
    ocr_server: RecordingServer, proxy: TestClient, ocr_backend: bool
) -> None:
    ocr_server.default_response = ResponseSpec(body={"message": "rejected"}, status=401)

    response: Final = post_ocr(proxy)

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert len(ocr_server.requests) == 1


def test_proxy_ocr_contract_missing_upstream_resource_returns_404(
    ocr_server: RecordingServer, proxy: TestClient, ocr_backend: bool
) -> None:
    ocr_server.default_response = ResponseSpec(body={"message": "no such model"}, status=404)

    response: Final = post_ocr(proxy)

    assert response.status_code == 404
    assert len(ocr_server.requests) == 1


def test_proxy_ocr_contract_provider_failure_reports_the_request_timeout_header(
    ocr_server: RecordingServer, proxy: TestClient, ocr_backend: bool
) -> None:
    ocr_server.default_response = ResponseSpec(body={"message": "provider unavailable"}, status=500)

    response: Final = post_ocr(proxy)

    assert response.status_code == 500
    assert len(ocr_server.requests) == ROUTER_RETRIES + 1
    assert "x-litellm-timeout" in response.headers


def test_proxy_ocr_contract_success_matches_the_provider_document(
    ocr_server: RecordingServer, proxy: TestClient, ocr_backend: bool
) -> None:
    response: Final = post_ocr(proxy)

    assert response.status_code == 200
    assert response.json()["pages"][0]["markdown"] == OCR_RESPONSE["pages"][0]["markdown"]
    assert response.headers.get("x-litellm-rust") == ("true" if ocr_backend else None)
