from unittest.mock import Mock

from google.genai import types as gx
import pytest

from nanobanana_mcp_server.config.settings import (
    LiteImageConfig,
    ModelSelectionConfig,
    ModelTier,
    NanoBanana2Config,
    ServerConfig,
    ThinkingLevel,
)
from nanobanana_mcp_server.core.exceptions import ValidationError
from nanobanana_mcp_server.services.gemini_client import GeminiClient
from nanobanana_mcp_server.services.model_selector import ModelSelector
from nanobanana_mcp_server.services.pro_image_service import ProImageService
from tests.test_edit_routing import (
    _FakeEnhancedImageService,
    _FakeModelSelector,
    _FakeProLikeService,
    _get_generate_image_fn,
)


def _build_service(model_config):
    gemini_client = Mock()
    gemini_client.generate_content = Mock(return_value=Mock())
    gemini_client.extract_images = Mock(return_value=[b"image-bytes"])
    gemini_client.extract_grounding = Mock(return_value=None)
    gemini_client.create_image_parts = Mock(return_value=["image-part"])
    return ProImageService(gemini_client, model_config, storage_service=None), gemini_client


def _selector(default_tier=ModelTier.AUTO, with_lite=True):
    services = {name: object() for name in ("flash", "pro", "nb2", "lite")}
    selector = ModelSelector(
        services["flash"],
        services["pro"],
        services["nb2"],
        ModelSelectionConfig(default_tier=default_tier),
        lite_service=services["lite"] if with_lite else None,
    )
    return selector, services


def _patch_tier(monkeypatch, tier, model_id, name, service=None):
    service = service or _FakeProLikeService()
    selector = _FakeModelSelector(
        selected_service=service, selected_tier=tier, model_id=model_id, name=name
    )
    monkeypatch.setattr("nanobanana_mcp_server.services.get_model_selector", lambda: selector)
    monkeypatch.setattr(
        "nanobanana_mcp_server.tools.generate_image._get_enhanced_image_service",
        lambda: _FakeEnhancedImageService(),
    )
    return service


def _write_images(tmp_path, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"ref_{i}.png"
        path.write_bytes(b"x")
        paths.append(str(path))
    return paths


# --- Thinking levels -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, ThinkingLevel.MINIMAL),
        (ThinkingLevel.LOW, ThinkingLevel.MINIMAL),
        (ThinkingLevel.MEDIUM, ThinkingLevel.MINIMAL),  # Lite rejects MEDIUM
        (ThinkingLevel.HIGH, ThinkingLevel.HIGH),
    ],
)
def test_lite_resolves_thinking_to_supported_levels(requested, expected):
    assert LiteImageConfig().resolve_thinking_level(requested) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, ThinkingLevel.MEDIUM),
        (ThinkingLevel.LOW, ThinkingLevel.MINIMAL),
        (ThinkingLevel.MEDIUM, ThinkingLevel.MEDIUM),
    ],
)
def test_nb2_resolves_thinking_levels(requested, expected):
    assert NanoBanana2Config().resolve_thinking_level(requested) == expected


@pytest.mark.unit
def test_lite_client_sends_thinking_but_not_grounding():
    client = GeminiClient(ServerConfig(gemini_api_key="test-key"), LiteImageConfig())
    client._client = Mock()
    client._client.models = Mock()
    client._client.models.generate_content = Mock(return_value=Mock())

    client.generate_content(
        contents=["prompt"], config={"thinking_level": "high", "enable_grounding": True}
    )

    sent = client._client.models.generate_content.call_args.kwargs
    assert sent["model"] == "gemini-3.1-flash-lite-image"
    assert sent["config"].thinking_config.thinking_level == gx.ThinkingLevel.HIGH
    assert sent["config"].tools is None


# --- Lite service ----------------------------------------------------------


@pytest.mark.unit
def test_lite_generate_uses_minimal_thinking_and_lite_label():
    service, gemini_client = _build_service(LiteImageConfig())

    _, metadata = service.generate_images(prompt="a banana", thinking_level=ThinkingLevel.MEDIUM)

    gen_config = gemini_client.generate_content.call_args.kwargs["config"]
    assert gen_config["thinking_level"] == "minimal"
    assert metadata[0]["model_tier"] == "lite"
    assert metadata[0]["thinking_level"] == "minimal"


@pytest.mark.unit
@pytest.mark.parametrize("resolution", ["2k", "4k"])
def test_lite_rejects_2k_and_4k(resolution):
    service, gemini_client = _build_service(LiteImageConfig())

    with pytest.raises(ValidationError, match="does not support"):
        service.generate_images(prompt="a banana", resolution=resolution)
    with pytest.raises(ValidationError, match="does not support"):
        service.edit_images("brighten", base_image_b64="aGVsbG8=", resolution=resolution)
    gemini_client.generate_content.assert_not_called()


# --- Model selection -------------------------------------------------------


@pytest.mark.unit
def test_explicit_lite_selects_lite_service():
    selector, services = _selector()

    service, tier = selector.select_model("a banana", requested_tier=ModelTier.LITE)

    assert (service, tier) == (services["lite"], ModelTier.LITE)


@pytest.mark.unit
def test_lite_falls_back_to_nb2_without_lite_service():
    selector, services = _selector(with_lite=False)

    service, tier = selector.select_model("a banana", requested_tier=ModelTier.LITE)

    assert (service, tier) == (services["nb2"], ModelTier.NB2)


@pytest.mark.unit
@pytest.mark.parametrize("default_tier", [ModelTier.LITE, ModelTier.PRO, ModelTier.FLASH])
def test_configured_default_tier_replaces_auto(default_tier):
    selector, services = _selector(default_tier=default_tier)

    service, tier = selector.select_model("a quick draft", requested_tier=ModelTier.AUTO)

    assert (service, tier) == (services[default_tier.value], default_tier)


@pytest.mark.unit
def test_explicit_tier_overrides_configured_default():
    selector, services = _selector(default_tier=ModelTier.LITE)

    service, tier = selector.select_model("a banana", requested_tier=ModelTier.PRO)

    assert (service, tier) == (services["pro"], ModelTier.PRO)


@pytest.mark.unit
def test_auto_without_configured_default_still_picks_nb2():
    selector, services = _selector()

    service, tier = selector.select_model("a banana", requested_tier=ModelTier.AUTO)

    assert (service, tier) == (services["nb2"], ModelTier.NB2)


# --- Reference images through the tool -------------------------------------


@pytest.mark.unit
def test_input_image_paths_list_is_appended_after_legacy_paths(monkeypatch, tmp_path):
    gen_fn = _get_generate_image_fn()
    nb2 = _patch_tier(monkeypatch, ModelTier.NB2, "gemini-nano-banana-2.1", "Nano Banana 2.1")
    first, *rest = _write_images(tmp_path, 14)

    result = gen_fn(
        prompt="arrange all of these products on one shelf",
        model_tier="nb2",
        input_image_path_1=first,
        input_image_paths=rest,
        output_path=str(tmp_path / "out.png"),
    )

    kwargs = nb2.calls[0][1]
    assert nb2.calls[0][0] == "generate"
    assert len(kwargs["input_images"]) == 14
    assert result.structured_content["input_image_count"] == 14
    assert result.structured_content["input_image_paths"][0].endswith("ref_0.png")


@pytest.mark.unit
def test_more_than_14_reference_images_is_rejected(monkeypatch, tmp_path):
    gen_fn = _get_generate_image_fn()
    nb2 = _patch_tier(monkeypatch, ModelTier.NB2, "gemini-nano-banana-2.1", "Nano Banana 2.1")

    with pytest.raises(ValidationError, match="at most 14"):
        gen_fn(
            prompt="collage",
            model_tier="nb2",
            input_image_paths=_write_images(tmp_path, 15),
            output_path=str(tmp_path / "out.png"),
        )
    assert nb2.calls == []


@pytest.mark.unit
def test_flash_accepts_at_most_3_reference_images(monkeypatch, tmp_path):
    gen_fn = _get_generate_image_fn()
    _patch_tier(monkeypatch, ModelTier.FLASH, "gemini-2.5-flash-image", "Gemini 2.5 Flash Image")

    with pytest.raises(ValidationError, match=r"at most 3 .*'nb2' or 'pro'"):
        gen_fn(
            prompt="collage",
            model_tier="flash",
            input_image_paths=_write_images(tmp_path, 4),
            output_path=str(tmp_path / "out.png"),
        )


@pytest.mark.unit
def test_lite_tier_routes_to_lite_service_with_thinking(monkeypatch, tmp_path):
    gen_fn = _get_generate_image_fn()
    lite = _patch_tier(
        monkeypatch, ModelTier.LITE, "gemini-3.1-flash-lite-image", "Nano Banana 2 Lite"
    )

    result = gen_fn(
        prompt="a quick banana sketch",
        mode="generate",
        model_tier="lite",
        thinking_level="high",
        enable_grounding=True,
        output_path=str(tmp_path / "out.png"),
    )

    assert lite.calls[0][0] == "generate"
    assert lite.calls[0][1]["thinking_level"] == ThinkingLevel.HIGH
    assert result.structured_content["model_tier"] == "lite"
    assert result.structured_content["thinking_level"] == "high"
    assert result.structured_content["grounding_enabled"] is False
    assert "Not supported by this model" in result.content[0].text
