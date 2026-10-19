from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types as gx
import pytest

from nanobanana_mcp_server.config.settings import (
    FlashImageConfig,
    NanoBanana2Config,
    ProImageConfig,
    ServerConfig,
)
from nanobanana_mcp_server.services.gemini_client import GeminiClient, ImageResult
from nanobanana_mcp_server.services.pro_image_service import ProImageService

GROUNDING = {
    "search_queries": ["tokyo weather today"],
    "sources": [{"title": "weather.example", "uri": "https://weather.example/tokyo"}],
}


def _build_client(model_config):
    client = GeminiClient(ServerConfig(gemini_api_key="test-key"), model_config)
    client._client = Mock()
    client._client.models = Mock()
    client._client.models.generate_content = Mock(return_value=Mock())
    return client


def _build_service(model_config, grounding=GROUNDING):
    gemini_client = Mock()
    gemini_client.generate_image_result = Mock(
        return_value=ImageResult(images=[b"image-bytes"], grounding=grounding)
    )
    gemini_client.create_image_parts = Mock(return_value=["image-part"])
    return ProImageService(gemini_client, model_config, storage_service=None), gemini_client


@pytest.mark.unit
@pytest.mark.parametrize("model_config", [NanoBanana2Config(), ProImageConfig()])
def test_grounding_sends_google_search_tool(model_config):
    client = _build_client(model_config)

    client.generate_content(contents=["prompt"], config={"enable_grounding": True})

    sent_config = client._client.models.generate_content.call_args.kwargs["config"]
    assert sent_config.tools == [gx.Tool(google_search=gx.GoogleSearch())]


@pytest.mark.unit
def test_no_tools_without_grounding():
    client = _build_client(NanoBanana2Config())

    client.generate_content(contents=["prompt"], config={"resolution": "1k"})

    sent_config = client._client.models.generate_content.call_args.kwargs["config"]
    assert sent_config.tools is None


@pytest.mark.unit
def test_flash_ignores_grounding():
    client = _build_client(FlashImageConfig())

    client.generate_content(contents=["prompt"], config={"enable_grounding": True})

    sent_config = client._client.models.generate_content.call_args.kwargs["config"]
    assert sent_config.tools is None


@pytest.mark.unit
def test_extract_grounding_returns_queries_and_sources():
    client = _build_client(NanoBanana2Config())
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    web_search_queries=["tokyo weather today"],
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(
                                title="weather.example", uri="https://weather.example/tokyo"
                            )
                        ),
                        SimpleNamespace(web=None),
                    ],
                )
            )
        ]
    )

    assert client.extract_grounding(response) == GROUNDING


@pytest.mark.unit
def test_extract_grounding_is_none_without_metadata():
    client = _build_client(NanoBanana2Config())
    response = SimpleNamespace(candidates=[SimpleNamespace(grounding_metadata=None)])

    assert client.extract_grounding(response) is None


@pytest.mark.unit
def test_generate_with_grounding_uses_tool_not_prompt_hint():
    service, gemini_client = _build_service(NanoBanana2Config())

    _, metadata = service.generate_images(
        prompt="today's weather in Tokyo as an infographic", enable_grounding=True
    )

    contents = gemini_client.generate_image_result.call_args.args[0]
    gen_config = gemini_client.generate_image_result.call_args.kwargs["config"]
    assert gen_config["enable_grounding"] is True
    assert not any("real-world knowledge" in str(part) for part in contents)
    assert metadata[0]["grounding"] == GROUNDING


@pytest.mark.unit
def test_generate_without_grounding_skips_search():
    service, gemini_client = _build_service(NanoBanana2Config(), grounding=None)

    _, metadata = service.generate_images(prompt="a banana")

    gen_config = gemini_client.generate_image_result.call_args.kwargs["config"]
    assert "enable_grounding" not in gen_config
    assert metadata[0]["grounding"] is None


@pytest.mark.unit
def test_nb2_default_thinking_level_is_medium():
    service, gemini_client = _build_service(NanoBanana2Config())

    service.generate_images(prompt="a banana")

    gen_config = gemini_client.generate_image_result.call_args.kwargs["config"]
    assert gen_config["thinking_level"] == "medium"


@pytest.mark.unit
def test_edit_forwards_aspect_ratio_size_and_negative_prompt():
    service, gemini_client = _build_service(NanoBanana2Config())

    _, metadata = service.edit_images(
        "make it a poster",
        base_image_b64="aGVsbG8=",
        aspect_ratio="16:9",
        resolution="2k",
        negative_prompt="text",
    )

    contents = gemini_client.generate_image_result.call_args.args[0]
    call_kwargs = gemini_client.generate_image_result.call_args.kwargs
    assert call_kwargs["aspect_ratio"] == "16:9"
    assert call_kwargs["config"]["resolution"] == "2k"
    assert contents[-1].endswith("Avoid: text")
    assert metadata[0]["aspect_ratio"] == "16:9"
    assert metadata[0]["resolution"] == "2k"


@pytest.mark.unit
def test_edit_keeps_source_size_for_default_resolution():
    service, gemini_client = _build_service(NanoBanana2Config())

    service.edit_images("brighten", base_image_b64="aGVsbG8=", resolution="high")

    call_kwargs = gemini_client.generate_image_result.call_args.kwargs
    assert "resolution" not in call_kwargs["config"]
    assert call_kwargs["aspect_ratio"] is None
