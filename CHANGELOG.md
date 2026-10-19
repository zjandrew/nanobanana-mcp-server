# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-10-19

### Added
- **Multi-turn editing**: NB2, Pro and Lite requests go through the Gemini Interactions API, and every result returns an `interaction_id`. Pass it back as `previous_interaction_id` with just the change ("make the umbrella yellow"); the model keeps the conversation context. With `model_tier="auto"` the follow-up reuses the previous result's model. Follow-up prompts are sent as written, and the output keeps the previous size unless `1k`/`2k`/`4k` is given.
- **Google Image Search grounding** (`image_search=True`, NB2 only): grounds on image results as well as web results and returns image search queries, image sources, and the search suggestions Google requires to be shown. Runs on `generate_content`, because the Interactions API returns image search as an unexecuted function call; these results have no `interaction_id`.
- **Video to image** (`input_video_url`, NB2 and Lite): uses a public YouTube video as context, with a 240s request timeout (calls take about 45–75s).
- Grounded results include `search_suggestions_html` when the API returns it.

### Changed
- **google-genai 2.31** (was 1.59).
- **`NANOBANANA_USE_INTERACTIONS`** (default `true`): set to `false` to keep everything on `generate_content`. Vertex AI and `GEMINI_BASE_URL` always use `generate_content`, so multi-turn editing needs API key auth on the default endpoint. Google stores Interactions conversations so they can be continued.
- Grounding over the Interactions API forces the search (`tool_choice: "any"`): without it the image models skipped the search in 6 of 6 test runs. Pro results grounded this way carry search queries but no source links, since the API only attaches citations to text output.

## [0.6.0] - 2026-10-19

### Added
- **Up to 14 reference images**: New `input_image_paths` list parameter, appended after `input_image_path_1..3`. NB2, Pro and Lite accept up to 14 images; Flash stays at 3. Over-limit requests fail with a message naming the limit.
- **`lite` tier — Nano Banana 2 Lite** (`gemini-3.1-flash-lite-image`): fastest and cheapest model. Behavior matches the live API: 1K output only (2K/4K are rejected with a clear error before calling the API), thinking `minimal` (default) or `high` (`medium` falls back to `minimal`), extreme aspect ratios supported, no Google Search grounding. `auto` never selects it.

### Fixed
- **`NANOBANANA_MODEL` had no effect**: it was only logged at startup. A non-`auto` value now replaces automatic selection for requests that use `model_tier="auto"`; an explicit `model_tier` still wins.

### Changed
- Generated files are named after the tier that made them (`nb2_…`, `lite_…`, `pro_…`) instead of always `pro_…`.

## [0.5.0] - 2026-10-19

### Fixed
- **NB2 thinking levels**: `thinking_level="low"` returned `400 Thinking level LOW is not supported` on Nano Banana 2.1. It is now sent as `minimal`, and `minimal`, `medium` and `high` are accepted. An invalid value falls back to the model default instead of `high`.
- **Google Search grounding**: `enable_grounding` only added a sentence to the prompt. It now sends the Google Search tool to NB2 and Pro, and the search queries and sources are returned in the result (`grounding` in structured content and in each image's metadata).
- **NB2/Pro edits**: `aspect_ratio`, `resolution`, `negative_prompt` and `enable_grounding` were dropped in edit mode. They are now applied. The output keeps the source image's size unless `1k`, `2k` or `4k` is given.

### Changed
- **`enable_grounding` defaults to `false`**: real grounding adds a search call and latency per request, so it is opt-in.
- **NB2 default thinking level is `medium`** (the API default), down from `high`.

## [0.4.8] - 2026-10-19

### Fixed
- **Model naming for MCP clients**: Server instructions and the `model_tier` description now name each tier's model (`nb2` = Nano Banana 2.1 / `gemini-nano-banana-2.1`, `pro` = Nano Banana Pro / `gemini-3-pro-image`, `flash` = Nano Banana / `gemini-2.5-flash-image`). Clients previously saw only "Nano Banana 2 (Gemini 3.1 Flash Image)" and reported that no 2.1 option existed.

## [0.4.7] - 2026-10-19

### Changed
- **Nano Banana 2.1**: NB2 tier now uses `gemini-nano-banana-2.1` (was `gemini-3.1-flash-image-preview`); Pro tier uses the stable `gemini-3-pro-image` ID (preview IDs were deprecated 2026-06-25). Cherry-picked from upstream PR #36.

### Fixed
- **Packaging**: Exclude `.claude/` from the sdist so local Claude Code settings are never published.

## [0.4.6] - 2026-09-27

### Fixed
- **Fresh installs**: Cap `fastmcp` at `>=2.11.0,<3`. FastMCP 4 removes `fastmcp.tools.tool`, so an unbounded `>=2.11.0` resolved 4.x and crashed on startup.
- **Startup log**: Report the configured `NANOBANANA_MODEL` instead of always logging `gemini-2.5-flash-image`.
- **Input images**: `input_image_path_*` and path-based edits only read png, jpg, jpeg, webp, or gif. Symlinks are resolved and must point at a supported image too.

### Changed
- Document that a relative `output_path` resolves from the server working directory. `IMAGE_OUTPUT_DIR` is used only when `output_path` is omitted.

## [0.4.5] - 2026-05-18

### Fixed
- **Vertex AI hang**: Configure request timeouts (60s for Flash/NB2, 90s for Pro) so an unresponsive endpoint returns an error instead of waiting forever (#31, #32).

### Changed
- Document `GCP_REGION=global` as the Vertex default for current image models (#28, #30).

## [0.4.4] - 2026-03-27

### Fixed
- **NB2 thinking support**: Send `thinking_level` to Gemini as `thinking_config` for NB2 requests
- **NB2 aspect ratio validation**: Allow extreme aspect ratios (`4:1`, `1:4`, `8:1`, `1:8`) on the NB2 path
- **Runtime/test alignment**: Update tests to cover NB2 thinking and model-specific aspect ratio behavior

## [0.3.3] - 2026-01-26

### Added
- **aspect_ratio support for Pro model**: Pass aspect ratio parameter to Gemini API for Pro service (#10)
- **output_path support for Pro model**: Direct file saving with automatic thumbnail generation (#10)

### Fixed
- **Input validation**: Add aspect_ratio validation before API calls to prevent invalid requests
- **Error handling**: Graceful degradation when thumbnail creation fails (returns full image instead)
- **Code quality**: Fix datetime timezone awareness (DTZ005) and hashlib security parameter (S324)

### Contributors
- @paulrobello (#10)

## [0.3.2] - 2026-01-17

### Fixed
- **API compatibility**: Remove unsupported `output_mime_type` parameter from `ImageConfig` (#11)
- **Dependency version**: Bump `google-genai` requirement from `>=1.41.0` to `>=1.57.0` to ensure `image_size` parameter support (#12)
- **Test coverage**: Update test expectations after `output_mime_type` removal

### Changed
- `ImageConfig` is now only created when there are actual parameters to configure (aspect_ratio or resolution)

### Contributors
- @georgesung (#13)

## [0.3.1] - 2026-01-08

### Added
- **output_path parameter**: Control where generated images are saved (#8)
  - Exact file path mode: `/path/to/image.png`
  - Directory mode: `/path/to/dir/`
  - Default fallback: `IMAGE_OUTPUT_DIR` or `~/nanobanana-images`

### Fixed
- **Pro model routing**: Fix critical bug where Pro model requests were always routed to Flash service (#9)
- **API parameters**: Remove unsupported `thinking_level` parameter from gemini-3-pro-image-preview
- **Response modalities**: Change from `["Image"]` to `["TEXT", "IMAGE"]` for Pro model compatibility
- **Default region**: Change default GCP region to `global` for Pro model support

### Contributors
- @jgeewax
- @akshayvkt

## [0.3.0] - 2026-01-06

### Added
- **Nano Banana Pro Integration**: Support for Gemini 3 Pro Image model
  - 4K resolution support (up to 3840px)
  - Google Search grounding for factual accuracy
  - Configurable thinking levels (LOW/HIGH)
  - Media resolution control
- **Intelligent Model Selection**: Automatic routing between Flash and Pro models
  - `ModelSelector` service for smart model routing
  - Quality/speed keyword detection
  - Resolution-based selection
- **ADC Authentication**: Application Default Credentials support for Vertex AI (#6)

### Changed
- Multi-tier configuration system: `ModelSelectionConfig`, `ProImageConfig`
- `ProImageService` for dedicated Gemini 3 Pro Image operations

## [0.2.0] - 2026-01-03

### Added
- Initial production-ready release
- Gemini 2.5 Flash Image support
- FastMCP framework integration
- Image generation and editing tools
- Aspect ratio validation
- Comprehensive test coverage
- MIT License

[0.4.6]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.4.3...v0.4.4
[0.3.3]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/zhongweili/nanobanana-mcp-server/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/zhongweili/nanobanana-mcp-server/releases/tag/v0.2.0
