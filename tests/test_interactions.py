import base64
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types as gx
import pytest

from nanobanana_mcp_server.config.settings import (
    AuthMethod,
    LiteImageConfig,
    ModelTier,
    NanoBanana2Config,
    ProImageConfig,
    ServerConfig,
)
from nanobanana_mcp_server.core.exceptions import GeminiAPIError, ValidationError
from nanobanana_mcp_server.services.gemini_client import GeminiClient, ImageResult
from nanobanana_mcp_server.services.model_selector import ModelSelector
from nanobanana_mcp_server.services.pro_image_service import ProImageService
from tests.test_edit_routing import (
    _FakeEnhancedImageService,
    _FakeProLikeService,
    _get_generate_image_fn,
)

PNG = b"\x89PNG-fake"
VIDEO_URL = "https://www.youtube.com/watch?v=UTdfxFyOQTI"


def _interaction(*steps, status="completed", interaction_id="ix-1"):
    return SimpleNamespace(id=interaction_id, status=status, steps=list(steps))


def _image_step(data=PNG):
    return SimpleNamespace(
        type="model_output",
        content=[SimpleNamespace(type="image", data=base64.b64encode(data).decode())],
    )


def _client(model_config, interaction=None, **server_kwargs):
    server = ServerConfig(
        gemini_api_key="test-key", auth_method=AuthMethod.API_KEY, **server_kwargs
    )
    client = GeminiClient(server, model_config)
    client._client = Mock()
    client._client.interactions.create = Mock(
        return_value=interaction or _interaction(_image_step())
    )
    client._client.models.generate_content = Mock(return_value=Mock(candidates=[]))
    return client


# --- When the Interactions API is used -------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("server_kwargs", "expected"),
    [
        ({}, True),
        ({"gemini_base_url": "https://proxy.example"}, False),
        ({"use_interactions": False}, False),
    ],
)
def test_uses_interactions_only_for_api_key_on_default_endpoint(server_kwargs, expected):
    assert _client(NanoBanana2Config(), **server_kwargs).uses_interactions is expected


@pytest.mark.unit
def test_vertex_ai_does_not_use_interactions():
    server = ServerConfig(auth_method=AuthMethod.VERTEX_AI, gcp_project_id="p")
    assert GeminiClient(server, NanoBanana2Config()).uses_interactions is False


# --- Request building and response parsing ---------------------------------


@pytest.mark.unit
def test_interaction_request_carries_inputs_format_thinking_and_grounding():
    client = _client(NanoBanana2Config())
    contents = [
        gx.Part(file_data=gx.FileData(file_uri=VIDEO_URL, mime_type="video/mp4")),
        gx.Part.from_bytes(data=b"raw", mime_type="image/png"),
        {"file_data": {"mime_type": "image/png", "uri": "https://files.example/abc"}},
        "make a poster",
    ]

    result = client.generate_image_result(
        contents,
        config={"resolution": "2k", "thinking_level": "low", "enable_grounding": True},
        aspect_ratio="16:9",
        previous_interaction_id="ix-0",
        timeout=240,
    )

    request = client._client.interactions.create.call_args.kwargs
    assert request["model"] == "gemini-nano-banana-2.1"
    assert request["input"] == [
        {"type": "video", "uri": VIDEO_URL, "mime_type": "video/mp4"},
        {"type": "image", "data": base64.b64encode(b"raw").decode(), "mime_type": "image/png"},
        {"type": "image", "uri": "https://files.example/abc", "mime_type": "image/png"},
        {"type": "text", "text": "make a poster"},
    ]
    assert request["response_format"] == {
        "type": "image",
        "aspect_ratio": "16:9",
        "image_size": "2K",
    }
    assert request["generation_config"] == {"thinking_level": "minimal", "tool_choice": "any"}
    assert request["tools"] == [{"type": "google_search"}]
    assert request["previous_interaction_id"] == "ix-0"
    assert request["store"] is True
    assert request["timeout"] == 240
    assert result == ImageResult(images=[PNG], grounding=None, interaction_id="ix-1")
    client._client.models.generate_content.assert_not_called()


@pytest.mark.unit
def test_lite_interaction_sends_no_search_tool():
    client = _client(LiteImageConfig())

    client.generate_image_result(["a banana"], config={"enable_grounding": True})

    assert "tools" not in client._client.interactions.create.call_args.kwargs


@pytest.mark.unit
def test_interaction_without_image_raises_with_status():
    client = _client(NanoBanana2Config(), interaction=_interaction(status="requires_action"))

    with pytest.raises(GeminiAPIError, match="requires_action"):
        client.generate_image_result(["a banana"])


@pytest.mark.unit
def test_interaction_grounding_collects_queries_sources_and_suggestions():
    client = _client(NanoBanana2Config())
    interaction = _interaction(
        SimpleNamespace(
            type="google_search_call",
            search_type="web_search",
            arguments={"queries": ["tokyo weather"]},
        ),
        SimpleNamespace(
            type="google_search_call",
            search_type="image_search",
            arguments={"queries": ["tokyo tower"]},
        ),
        SimpleNamespace(
            type="google_search_result",
            result=[SimpleNamespace(search_suggestions="<div>chips</div>")],
        ),
        SimpleNamespace(
            type="model_output",
            content=[
                SimpleNamespace(
                    type="text",
                    annotations=[
                        SimpleNamespace(
                            type="url_citation", title="Weather", url="https://w.example"
                        )
                    ],
                )
            ],
        ),
    )

    assert client.extract_interaction_grounding(interaction) == {
        "search_queries": ["tokyo weather"],
        "image_search_queries": ["tokyo tower"],
        "sources": [{"title": "Weather", "uri": "https://w.example"}],
        "search_suggestions_html": "<div>chips</div>",
    }


# --- Paths that stay on generate_content -------------------------------------


@pytest.mark.unit
def test_image_search_uses_generate_content_with_image_search_type():
    client = _client(NanoBanana2Config())

    result = client.generate_image_result(["a butterfly"], config={}, image_search=True, timeout=30)

    client._client.interactions.create.assert_not_called()
    sent_config = client._client.models.generate_content.call_args.kwargs["config"]
    search_types = sent_config.tools[0].google_search.search_types
    assert search_types.image_search is not None and search_types.web_search is not None
    assert sent_config.http_options.timeout == 30_000
    assert result.interaction_id is None


@pytest.mark.unit
def test_image_search_is_ignored_by_models_without_it():
    client = _client(ProImageConfig(), use_interactions=False)

    client.generate_image_result(["a butterfly"], config={}, image_search=True)

    assert client._client.models.generate_content.call_args.kwargs["config"].tools is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("server_kwargs", "image_search"),
    [({"use_interactions": False}, False), ({}, True)],
)
def test_previous_interaction_needs_the_interactions_path(server_kwargs, image_search):
    client = _client(NanoBanana2Config(), **server_kwargs)

    with pytest.raises(ValidationError, match="previous_interaction_id"):
        client.generate_image_result(
            ["make it blue"], previous_interaction_id="ix-0", image_search=image_search
        )


@pytest.mark.unit
def test_generate_content_grounding_includes_image_sources_and_suggestions():
    client = _client(NanoBanana2Config())
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    web_search_queries=["timareta"],
                    image_search_queries=["timareta butterfly"],
                    grounding_chunks=[
                        SimpleNamespace(
                            web=None,
                            image=SimpleNamespace(
                                title="Butterflies",
                                source_uri="https://b.example",
                                image_uri="https://b.example/1.jpg",
                            ),
                        )
                    ],
                    search_entry_point=SimpleNamespace(rendered_content="<div>chips</div>"),
                )
            )
        ]
    )

    assert client.extract_grounding(response) == {
        "search_queries": ["timareta"],
        "image_search_queries": ["timareta butterfly"],
        "sources": [
            {
                "title": "Butterflies",
                "uri": "https://b.example",
                "image_uri": "https://b.example/1.jpg",
            }
        ],
        "search_suggestions_html": "<div>chips</div>",
    }


@pytest.mark.unit
def test_unknown_interaction_id_is_a_validation_error():
    client = _client(NanoBanana2Config())
    client._client.interactions.get = Mock(side_effect=RuntimeError("404 not found"))

    with pytest.raises(ValidationError, match="could not be loaded"):
        client.get_interaction_model("ix-missing")


# --- ProImageService ---------------------------------------------------------


def _service(model_config):
    gemini_client = Mock()
    gemini_client.generate_image_result = Mock(
        return_value=ImageResult(images=[b"image-bytes"], interaction_id="ix-2")
    )
    gemini_client.create_image_parts = Mock(return_value=["image-part"])
    return ProImageService(gemini_client, model_config, storage_service=None), gemini_client


@pytest.mark.unit
def test_follow_up_turn_sends_prompt_as_written_and_keeps_size():
    service, gemini_client = _service(NanoBanana2Config())

    _, metadata = service.generate_images(prompt="make it blue", previous_interaction_id="ix-1")

    call = gemini_client.generate_image_result.call_args
    assert call.args[0] == ["make it blue"]
    assert "resolution" not in call.kwargs["config"]
    assert call.kwargs["previous_interaction_id"] == "ix-1"
    assert metadata[0]["interaction_id"] == "ix-2"
    assert metadata[0]["previous_interaction_id"] == "ix-1"


@pytest.mark.unit
def test_follow_up_turn_keeps_an_explicit_size():
    service, gemini_client = _service(NanoBanana2Config())

    service.generate_images(prompt="make it blue", previous_interaction_id="ix-1", resolution="2k")

    assert gemini_client.generate_image_result.call_args.kwargs["config"]["resolution"] == "2k"


@pytest.mark.unit
def test_video_goes_first_with_a_longer_timeout():
    service, gemini_client = _service(LiteImageConfig())

    service.generate_images(prompt="a poster for this video", input_video_url=VIDEO_URL)

    call = gemini_client.generate_image_result.call_args
    video = call.args[0][0]
    assert video.file_data.file_uri == VIDEO_URL
    assert call.kwargs["timeout"] == LiteImageConfig().video_request_timeout


@pytest.mark.unit
def test_image_search_turns_on_grounding():
    service, gemini_client = _service(NanoBanana2Config())

    service.generate_images(prompt="a timareta butterfly", image_search=True)

    call = gemini_client.generate_image_result.call_args
    assert call.kwargs["image_search"] is True
    assert call.kwargs["config"]["enable_grounding"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "kwargs", "message"),
    [
        (ProImageConfig(), {"input_video_url": VIDEO_URL}, "video input"),
        (ProImageConfig(), {"image_search": True}, "image_search"),
        (LiteImageConfig(), {"image_search": True}, "image_search"),
    ],
)
def test_unsupported_inputs_are_rejected(config, kwargs, message):
    service, gemini_client = _service(config)

    with pytest.raises(ValidationError, match=message):
        service.generate_images(prompt="x", **kwargs)
    gemini_client.generate_image_result.assert_not_called()


# --- ModelSelector -------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "tier"),
    [
        ("gemini-nano-banana-2.1", ModelTier.NB2),
        ("gemini-3-pro-image", ModelTier.PRO),
        ("gemini-3.1-flash-lite-image", ModelTier.LITE),
        ("some-other-model", None),
    ],
)
def test_tier_for_interaction_maps_the_stored_model(model_name, tier):
    def service(config):
        return SimpleNamespace(
            config=config, gemini_client=Mock(get_interaction_model=Mock(return_value=model_name))
        )

    selector = ModelSelector(
        object(),
        service(ProImageConfig()),
        service(NanoBanana2Config()),
        Mock(default_tier=ModelTier.AUTO),
        lite_service=service(LiteImageConfig()),
    )

    assert selector.tier_for_interaction("ix-1") == tier


# --- Tool --------------------------------------------------------------------


class _RecordingSelector:
    def __init__(self, service, tier, previous_tier=None):
        self.service, self.tier, self.previous_tier = service, tier, previous_tier
        self.requested = []

    def tier_for_interaction(self, interaction_id):
        return self.previous_tier

    def select_model(self, prompt, requested_tier=None, **kwargs):
        self.requested.append(requested_tier)
        tier = self.tier if requested_tier == ModelTier.AUTO else requested_tier
        return self.service, tier

    def get_model_info(self, tier):
        return {"tier": tier.value, "name": tier.value, "model_id": tier.value, "emoji": "🧪"}


class _ServiceWithInteraction(_FakeProLikeService):
    def generate_images(self, **kwargs):
        thumbs, metadata = super().generate_images(**kwargs)
        metadata[0]["interaction_id"] = "ix-2"
        return thumbs, metadata


def _patch(monkeypatch, selector):
    monkeypatch.setattr("nanobanana_mcp_server.services.get_model_selector", lambda: selector)
    monkeypatch.setattr(
        "nanobanana_mcp_server.tools.generate_image._get_enhanced_image_service",
        lambda: _FakeEnhancedImageService(),
    )


@pytest.mark.unit
def test_follow_up_reuses_previous_model_and_returns_new_interaction_id(monkeypatch, tmp_path):
    service = _ServiceWithInteraction()
    selector = _RecordingSelector(service, ModelTier.NB2, previous_tier=ModelTier.PRO)
    _patch(monkeypatch, selector)

    result = _get_generate_image_fn()(
        prompt="make it blue", previous_interaction_id="ix-1", output_path=str(tmp_path / "o.png")
    )

    assert selector.requested == [ModelTier.PRO]
    assert service.calls[0][0] == "generate"
    assert service.calls[0][1]["previous_interaction_id"] == "ix-1"
    assert result.structured_content["mode"] == "edit"
    assert result.structured_content["interaction_id"] == "ix-2"
    assert result.structured_content["previous_interaction_id"] == "ix-1"
    assert "previous_interaction_id" in result.content[0].text


@pytest.mark.unit
@pytest.mark.parametrize(
    "video_or_search", [{"input_video_url": VIDEO_URL}, {"image_search": True}]
)
def test_video_and_image_search_pick_nb2_in_auto(monkeypatch, tmp_path, video_or_search):
    service = _FakeProLikeService()
    selector = _RecordingSelector(service, ModelTier.PRO)
    _patch(monkeypatch, selector)

    _get_generate_image_fn()(
        prompt="poster", output_path=str(tmp_path / "o.png"), **video_or_search
    )

    assert selector.requested == [ModelTier.NB2]
    ((key, value),) = video_or_search.items()
    assert service.calls[0][1][key] == value


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"previous_interaction_id": "ix-1", "model_tier": "flash"}, "flash tier"),
        ({"previous_interaction_id": "ix-1", "file_id": "files/abc"}, "file_id cannot"),
        ({"previous_interaction_id": "ix-1", "image_search": True}, "image_search cannot"),
        ({"input_video_url": VIDEO_URL, "model_tier": "pro"}, "input_video_url"),
        ({"image_search": True, "model_tier": "lite"}, "image_search"),
    ],
)
def test_invalid_combinations_are_rejected(monkeypatch, tmp_path, kwargs, message):
    service = _FakeProLikeService()
    _patch(monkeypatch, _RecordingSelector(service, ModelTier.NB2, previous_tier=ModelTier.NB2))

    with pytest.raises(ValidationError, match=message):
        _get_generate_image_fn()(prompt="x", output_path=str(tmp_path / "o.png"), **kwargs)
    assert service.calls == []
