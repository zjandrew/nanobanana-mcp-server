import base64
import logging
import mimetypes
import os
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent
from pydantic import Field

from ..config.constants import MAX_INPUT_IMAGES, MAX_INPUT_IMAGES_BY_TIER
from ..config.settings import ModelTier, ThinkingLevel
from ..core.exceptions import ValidationError
from ..utils.validation_utils import validate_input_image_path, validate_output_path

# Tiers whose models accept a thinking level
THINKING_TIERS = (ModelTier.NB2, ModelTier.LITE)


def register_generate_image_tool(server: FastMCP):
    """Register the generate_image tool with the FastMCP server."""

    @server.tool(
        annotations={
            "title": "Generate or edit images (Multi-Model: Flash & Pro)",
            "readOnlyHint": True,
            "openWorldHint": True,
        }
    )
    def generate_image(
        prompt: Annotated[
            str,
            Field(
                description="Clear, detailed image prompt. Include subject, composition, "
                "action, location, style, and any text to render. Use the aspect_ratio "
                "parameter to pin a specific canvas shape when needed.",
                min_length=1,
                max_length=8192,
            ),
        ],
        n: Annotated[
            int, Field(description="Requested image count (model may return fewer).", ge=1, le=4)
        ] = 1,
        negative_prompt: Annotated[
            str | None,
            Field(description="Things to avoid (style, objects, text).", max_length=1024),
        ] = None,
        system_instruction: Annotated[
            str | None, Field(description="Optional system tone/style guidance.", max_length=512)
        ] = None,
        input_image_path_1: Annotated[
            str | None,
            Field(description="Path to the first input image (png, jpg, jpeg, webp, or gif)."),
        ] = None,
        input_image_path_2: Annotated[
            str | None,
            Field(description="Path to the second input image (png, jpg, jpeg, webp, or gif)."),
        ] = None,
        input_image_path_3: Annotated[
            str | None,
            Field(description="Path to the third input image (png, jpg, jpeg, webp, or gif)."),
        ] = None,
        input_image_paths: Annotated[
            list[str] | None,
            Field(
                description="Paths to input/reference images (png, jpg, jpeg, webp, or gif): "
                "up to 14 for 'nb2', 'pro' and 'lite', 3 for 'flash'. Appended after "
                "input_image_path_1..3. One image edits it; several images are references "
                "for a new image. Say each image's role in the prompt (e.g. 'image 1 is the "
                "product, image 2 is the style'). Nano Banana 2.1 keeps up to 10 objects "
                "and 4 characters consistent; Pro up to 6 objects, 5 characters, 3 styles.",
                max_length=MAX_INPUT_IMAGES,
            ),
        ] = None,
        input_video_url: Annotated[
            str | None,
            Field(
                description="Public YouTube video URL to use as visual context, e.g. for a "
                "poster or thumbnail ('nb2' and 'lite' only; auto picks 'nb2'). "
                "Takes about 1 minute or more."
            ),
        ] = None,
        previous_interaction_id: Annotated[
            str | None,
            Field(
                description="Continue editing an earlier result: pass the interaction_id it "
                "returned and describe only the change (e.g. 'make the banana blue'). The "
                "model keeps the full context of that conversation. With model_tier 'auto' "
                "the same model is reused. Not available for 'flash' or with image_search."
            ),
        ] = None,
        file_id: Annotated[
            str | None,
            Field(
                description="Files API file ID to use as input/edit source (e.g., 'files/abc123'). "
                "If provided, this takes precedence over input_image_path_* parameters for the primary input."
            ),
        ] = None,
        mode: Annotated[
            str,
            Field(
                description="Operation mode: 'generate' for new image creation, 'edit' for modifying existing images. "
                "Auto-detected based on input parameters if not specified."
            ),
        ] = "auto",
        model_tier: Annotated[
            str | None,
            Field(
                description="Model tier: 'flash' (legacy Nano Banana, gemini-2.5-flash-image, 1024px), "
                "'nb2' (Nano Banana 2.1, gemini-nano-banana-2.1, 4K at Flash speed, default), "
                "'pro' (Nano Banana Pro, gemini-3-pro-image, max quality, 4K), "
                "'lite' (Nano Banana 2 Lite, gemini-3.1-flash-lite-image, fastest and "
                "cheapest, 1K only, no grounding), "
                "or 'auto' (smart selection). "
                "Default: 'auto' - automatically selects nb2 or pro based on prompt, "
                "unless the server sets NANOBANANA_MODEL."
            ),
        ] = "auto",
        resolution: Annotated[
            str | None,
            Field(
                description="Output resolution: '1k', '2k', '4k', or 'high' (= 1k). "
                "2K and 4K available with 'nb2' and 'pro' ('lite' is 1K only). In edit mode the output "
                "keeps the source image's size unless '1k', '2k' or '4k' is given. Default: 'high'."
            ),
        ] = "high",
        thinking_level: Annotated[
            str | None,
            Field(
                description="Reasoning depth for 'nb2' and 'lite': 'minimal' (fastest), "
                "'medium' (nb2 default), or 'high' (best for complex compositions). "
                "'lite' supports only 'minimal' (default) and 'high'. "
                "'low' is accepted as an alias for 'minimal'. "
                "'high' also biases auto-selection toward Pro. Default: None (model default)."
            ),
        ] = None,
        enable_grounding: Annotated[
            bool,
            Field(
                description="Ground the image in live Google Search results ('nb2' and 'pro' only). "
                "Use for current or factual subjects such as weather, events, charts, or real "
                "places. Adds latency and search cost. Default: false."
            ),
        ] = False,
        image_search: Annotated[
            bool,
            Field(
                description="Also ground on Google Image Search results, for subjects whose "
                "look matters (a specific species, landmark or product). 'nb2' only; implies "
                "enable_grounding. Results cannot be continued with previous_interaction_id."
            ),
        ] = False,
        aspect_ratio: Annotated[
            Literal[
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
                "4:1",
                "1:4",
                "8:1",
                "1:8",
            ]
            | None,
            Field(
                description="Optional output aspect ratio (e.g., '16:9'). "
                "Standard: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9. "
                "Extreme (nb2 and lite only): 4:1, 1:4, 8:1, 1:8."
            ),
        ] = None,
        output_path: Annotated[
            str | None,
            Field(
                description="Output path for generated image(s). "
                "If a file path with extension (e.g., '/path/image.png'), saves directly to that path. "
                "If a directory path (e.g., '/path/to/dir/'), uses default filename in that directory. "
                "If None, uses IMAGE_OUTPUT_DIR environment variable or ~/nanobanana-images."
            ),
        ] = None,
        return_full_image: Annotated[
            bool | None,
            Field(
                description="Return full-resolution images in MCP response instead of thumbnails. "
                "Warning: full images can be large (3-7MB each for 4K). "
                "Default: uses RETURN_FULL_IMAGE env var, or false if not set."
            ),
        ] = None,
        _ctx: Context | None = None,
    ) -> ToolResult:
        """
        Generate new images or edit existing images using natural language instructions.

        Supports multiple input modes:
        1. Pure generation: Just provide a prompt to create new images
        2. Multi-image conditioning: Provide up to 3 input images using input_image_path_1/2/3 parameters
        3. File ID editing: Edit previously uploaded images using Files API ID
        4. File path editing: Edit local images by providing single input image path

        Automatically detects mode based on parameters or can be explicitly controlled.
        Input images are read from the local filesystem to avoid massive token usage.
        Returns both MCP image content blocks and structured JSON with metadata.
        """
        logger = logging.getLogger(__name__)

        try:
            # Combine the individual path parameters with the input_image_paths list
            listed_paths = input_image_paths or []
            input_image_paths = []
            for path in [input_image_path_1, input_image_path_2, input_image_path_3, *listed_paths]:
                if path:
                    input_image_paths.append(path)

            # Convert empty list to None for consistency
            if not input_image_paths:
                input_image_paths = None

            logger.info(
                f"Generate image request: prompt='{prompt[:50]}...', n={n}, "
                f"paths={input_image_paths}, model_tier={model_tier}, aspect_ratio={aspect_ratio}, "
                f"output_path={output_path}"
            )

            # Validate output_path if provided
            validate_output_path(output_path)

            # A follow-up turn edits the previous result through the generation path
            conversational = bool(previous_interaction_id)
            if conversational and file_id:
                raise ValidationError(
                    "file_id cannot be combined with previous_interaction_id; "
                    "pass extra images with input_image_paths"
                )
            if conversational and image_search:
                raise ValidationError(
                    "image_search cannot continue a previous interaction; "
                    "start a new request without previous_interaction_id"
                )

            # Auto-detect mode based on inputs
            detected_mode = mode
            if mode == "auto":
                if conversational:
                    detected_mode = "edit"
                elif input_video_url:
                    detected_mode = "generate"
                elif file_id or (input_image_paths and len(input_image_paths) == 1):
                    detected_mode = "edit"
                else:
                    detected_mode = "generate"
            if input_video_url and detected_mode == "edit" and not conversational:
                raise ValidationError("input_video_url works in generate mode, not edit mode")

            # Get model selector to determine which model to use
            from ..services import get_model_selector

            model_selector = get_model_selector()

            # Parse model tier
            try:
                tier = ModelTier(model_tier) if model_tier else ModelTier.AUTO
            except ValueError:
                logger.warning(f"Invalid model_tier '{model_tier}', defaulting to AUTO")
                tier = ModelTier.AUTO

            if tier == ModelTier.AUTO:
                if conversational:
                    # Keep editing on the model that made the previous result
                    tier = (
                        model_selector.tier_for_interaction(previous_interaction_id)
                        or ModelTier.AUTO
                    )
                elif image_search or input_video_url:
                    tier = ModelTier.NB2  # The tier that supports both

            # Normalize thinking level ('low' is a legacy alias for 'minimal')
            if thinking_level:
                thinking_level = thinking_level.strip().lower()
                if thinking_level == ThinkingLevel.LOW.value:
                    thinking_level = ThinkingLevel.MINIMAL.value
                try:
                    _ = ThinkingLevel(thinking_level)  # Just validate
                except ValueError:
                    logger.warning(
                        f"Invalid thinking_level '{thinking_level}', using the model default"
                    )
                    thinking_level = None

            # Select model based on prompt and parameters
            selected_service, selected_tier = model_selector.select_model(
                prompt=prompt,
                requested_tier=tier,
                n=n,
                resolution=resolution,
                input_images=input_image_paths,
                thinking_level=thinking_level,
                enable_grounding=enable_grounding,
            )

            model_info = model_selector.get_model_info(selected_tier)
            logger.info(
                f"Selected {model_info['emoji']} {model_info['name']} "
                f"({selected_tier.value}) for this request"
            )

            # Validation
            if mode not in ["auto", "generate", "edit"]:
                raise ValidationError("Mode must be 'auto', 'generate', or 'edit'")
            if conversational and selected_tier == ModelTier.FLASH:
                raise ValidationError("previous_interaction_id is not supported by the flash tier")
            if image_search and selected_tier != ModelTier.NB2:
                raise ValidationError("image_search is only supported by model_tier='nb2'")
            if input_video_url and selected_tier not in THINKING_TIERS:
                raise ValidationError(
                    "input_video_url is only supported by model_tier='nb2' or 'lite'"
                )

            if input_image_paths:
                max_images = MAX_INPUT_IMAGES_BY_TIER.get(selected_tier.value, MAX_INPUT_IMAGES)
                if len(input_image_paths) > max_images:
                    raise ValidationError(
                        f"{model_info['name']} accepts at most {max_images} input images "
                        f"(got {len(input_image_paths)})"
                        + (
                            "; use model_tier='nb2' or 'pro' for up to 14"
                            if max_images < 14
                            else ""
                        )
                    )

                resolved_paths = []
                for i, path in enumerate(input_image_paths):
                    resolved_paths.append(str(validate_input_image_path(path, index=i + 1)))
                input_image_paths = resolved_paths

            # Mode-specific validation
            if detected_mode == "edit" and not conversational:
                if not file_id and not input_image_paths:
                    raise ValidationError("Edit mode requires either file_id or input_image_paths")
                if file_id and input_image_paths and len(input_image_paths) > 1:
                    raise ValidationError(
                        "Edit mode with file_id supports only additional input images, not multiple primary inputs"
                    )

            # Get enhanced image service (workflows.md + Files API + DB)
            enhanced_image_service = _get_enhanced_image_service()

            # Execute based on detected mode
            if detected_mode == "edit" and not conversational:
                if selected_tier == ModelTier.FLASH:
                    # Flash edit path uses EnhancedImageService (workflows.md + Files API)
                    if file_id:
                        logger.info(
                            f"Edit mode (FLASH): using file_id {file_id}, output_path={output_path}"
                        )
                        thumbnail_images, metadata = enhanced_image_service.edit_image_by_file_id(
                            file_id=file_id, edit_prompt=prompt, output_path=output_path
                        )
                    else:
                        # Edit by file path
                        logger.info(
                            f"Edit mode (FLASH): using file path {input_image_paths[0]}, output_path={output_path}"
                        )
                        thumbnail_images, metadata = enhanced_image_service.edit_image_by_path(
                            instruction=prompt,
                            file_path=input_image_paths[0],
                            output_path=output_path,
                        )
                else:
                    # PRO / NB2 edit path uses ProImageService (selected_service)
                    # For file_id, prefer file_data parts to avoid downloading / base64.
                    if file_id:
                        from ..services import get_files_api_service

                        files_api_service = get_files_api_service()
                        file_data_part = files_api_service.create_file_data_part(file_id)
                        logger.info(
                            f"Edit mode ({selected_tier.value.upper()}): using file_id {file_id}, output_path={output_path}"
                        )
                        thumbnail_images, metadata = selected_service.edit_images(
                            instruction=prompt,
                            file_data_part=file_data_part,
                            output_path=output_path,
                            thinking_level=(
                                ThinkingLevel(thinking_level)
                                if (thinking_level and selected_tier in THINKING_TIERS)
                                else None
                            ),
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                            negative_prompt=negative_prompt,
                            enable_grounding=enable_grounding,
                            use_storage=True,
                        )
                        for meta in metadata:
                            if isinstance(meta, dict):
                                meta.setdefault("parent_file_id", file_id)
                    else:
                        # Edit by file path (read bytes locally)
                        src_path = input_image_paths[0]
                        logger.info(
                            f"Edit mode ({selected_tier.value.upper()}): using file path {src_path}, output_path={output_path}"
                        )
                        try:
                            with open(src_path, "rb") as f:
                                image_bytes = f.read()
                            mime_type, _ = mimetypes.guess_type(src_path)
                            if not mime_type or not mime_type.startswith("image/"):
                                mime_type = "image/png"
                            base64_data = base64.b64encode(image_bytes).decode("utf-8")
                        except Exception as e:
                            raise ValidationError(
                                f"Failed to load input image {src_path}: {e}"
                            ) from e

                        thumbnail_images, metadata = selected_service.edit_images(
                            instruction=prompt,
                            base_image_b64=base64_data,
                            mime_type=mime_type,
                            output_path=output_path,
                            thinking_level=(
                                ThinkingLevel(thinking_level)
                                if (thinking_level and selected_tier in THINKING_TIERS)
                                else None
                            ),
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                            negative_prompt=negative_prompt,
                            enable_grounding=enable_grounding,
                            use_storage=True,
                        )
                        for meta in metadata:
                            if isinstance(meta, dict):
                                meta.setdefault("source_path", src_path)

            else:
                # Generation mode (with optional input images for conditioning)
                logger.info("Generate mode: creating new images")
                if aspect_ratio:
                    logger.info(f"Using aspect ratio override: {aspect_ratio}")

                # Prepare input images by reading from file paths
                input_images = None
                if input_image_paths:
                    input_images = []

                    for path in input_image_paths:
                        try:
                            # Read image file
                            with open(path, "rb") as f:
                                image_bytes = f.read()

                            # Detect MIME type
                            mime_type, _ = mimetypes.guess_type(path)
                            if not mime_type or not mime_type.startswith("image/"):
                                mime_type = "image/png"  # Fallback

                            # Convert to base64 for internal API use
                            base64_data = base64.b64encode(image_bytes).decode("utf-8")
                            input_images.append((base64_data, mime_type))

                            logger.debug(f"Loaded input image: {path} ({mime_type})")

                        except Exception as e:
                            raise ValidationError(f"Failed to load input image {path}: {e}") from e

                    logger.info(f"Loaded {len(input_images)} input images from file paths")

                # Generate images following workflows.md pattern:
                # M->G->FS->F->D (save full-res, create thumbnail, upload to Files API, track in DB)
                # Route to correct service based on selected model tier
                if selected_tier == ModelTier.PRO:
                    # Use Pro service for high-quality generation
                    logger.info(f"Using PRO model: {model_info['model_id']}")
                    if aspect_ratio:
                        logger.info(f"Using aspect ratio: {aspect_ratio}")
                    if output_path:
                        logger.info(f"Using output path: {output_path}")
                    thumbnail_images, metadata = selected_service.generate_images(
                        prompt=prompt,
                        n=n,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        output_path=output_path,
                        enable_grounding=enable_grounding,
                        negative_prompt=negative_prompt,
                        system_instruction=system_instruction,
                        input_images=input_images,
                        use_storage=True,
                        previous_interaction_id=previous_interaction_id,
                        image_search=image_search,
                        input_video_url=input_video_url,
                    )
                elif selected_tier in THINKING_TIERS:
                    # NB2 / Lite services (Flash speed, support thinking)
                    logger.info(
                        f"Using {selected_tier.value.upper()} model: {model_info['model_id']}"
                    )
                    thumbnail_images, metadata = selected_service.generate_images(
                        prompt=prompt,
                        n=n,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        output_path=output_path,
                        thinking_level=ThinkingLevel(thinking_level) if thinking_level else None,
                        enable_grounding=enable_grounding,
                        negative_prompt=negative_prompt,
                        system_instruction=system_instruction,
                        input_images=input_images,
                        use_storage=True,
                        previous_interaction_id=previous_interaction_id,
                        image_search=image_search,
                        input_video_url=input_video_url,
                    )
                else:
                    # Use Flash service (via enhanced_image_service) for speed
                    logger.info(f"Using FLASH model: {model_info['model_id']}")
                    thumbnail_images, metadata = enhanced_image_service.generate_images(
                        prompt=prompt,
                        n=n,
                        negative_prompt=negative_prompt,
                        system_instruction=system_instruction,
                        input_images=input_images,
                        aspect_ratio=aspect_ratio,
                        output_path=output_path,
                    )

            # Resolve return_full_image: tool param > server config > env var > default (false)
            effective_return_full_image = return_full_image
            if effective_return_full_image is None:
                from ..services import get_server_config

                try:
                    effective_return_full_image = get_server_config().return_full_image
                except RuntimeError:
                    effective_return_full_image = os.getenv(
                        "RETURN_FULL_IMAGE", "false"
                    ).strip().lower() in ("true", "1", "yes")

            # Create response with file paths and thumbnails
            grounding = None
            if metadata:
                # Filter out any None entries from metadata, keeping thumbnail_images aligned
                filtered_pairs = [
                    (m, thumbnail_images[i] if i < len(thumbnail_images) else None)
                    for i, m in enumerate(metadata)
                    if m is not None and isinstance(m, dict)
                ]
                if filtered_pairs:
                    metadata, thumbnail_images = zip(*filtered_pairs, strict=False)
                    metadata = list(metadata)
                    thumbnail_images = [img for img in thumbnail_images if img is not None]
                else:
                    metadata = []

                if not metadata:
                    summary = f"❌ Failed to {detected_mode} image(s): {prompt[:50]}... No valid results returned."
                    content = [TextContent(type="text", text=summary)]
                    structured_content = {
                        "error": "no_valid_metadata",
                        "message": summary,
                        "mode": detected_mode,
                    }
                    return ToolResult(content=content, structured_content=structured_content)

                # Replace thumbnails with full-resolution images if requested
                if effective_return_full_image:
                    from fastmcp.utilities.types import Image as MCPImage

                    full_images = []
                    total_size = 0
                    for i, meta in enumerate(metadata):
                        if not meta or not isinstance(meta, dict):
                            if i < len(thumbnail_images):
                                full_images.append(thumbnail_images[i])
                            continue
                        full_path = meta.get("full_path")
                        if full_path and os.path.isfile(full_path):
                            full_images.append(MCPImage(path=full_path))
                            total_size += meta.get("size_bytes", 0)
                        else:
                            if i < len(thumbnail_images):
                                full_images.append(thumbnail_images[i])
                            logger.warning(
                                f"Full image not found for image {i + 1}, using thumbnail"
                            )

                    total_size_mb = total_size / (1024 * 1024)
                    if total_size_mb > 10:
                        logger.warning(
                            f"Large MCP response: {total_size_mb:.1f}MB across "
                            f"{len(full_images)} full-resolution image(s)"
                        )
                    thumbnail_images = full_images

                # Build summary with mode-specific information
                action_verb = "Edited" if detected_mode == "edit" else "Generated"
                model_name = model_info["name"]
                model_emoji = model_info["emoji"]
                if selected_tier in THINKING_TIERS:
                    # Report the level the service actually used (default or remapped)
                    thinking_level = next(
                        (m.get("thinking_level") for m in metadata if m.get("thinking_level")),
                        thinking_level,
                    )
                grounding_applied = enable_grounding and selected_tier in (
                    ModelTier.PRO,
                    ModelTier.NB2,
                )
                grounding = next((m.get("grounding") for m in metadata if m.get("grounding")), None)
                summary_lines = [
                    f"✅ {action_verb} {len(metadata)} image(s) with {model_emoji} {model_name}.",
                    f"📊 **Model**: {selected_tier.value.upper()} tier",
                ]

                # Add model-specific information
                if selected_tier == ModelTier.PRO:
                    summary_lines.append(f"📏 **Resolution**: {resolution}")
                    if grounding_applied:
                        summary_lines.append("🔍 **Grounding**: Enabled (Google Search)")
                elif selected_tier in THINKING_TIERS:
                    if thinking_level:
                        summary_lines.append(f"🧠 **Thinking Level**: {thinking_level}")
                    summary_lines.append(f"📏 **Resolution**: {resolution}")
                    if grounding_applied:
                        summary_lines.append("🔍 **Grounding**: Enabled (Google Search)")
                    elif enable_grounding:
                        summary_lines.append("🔍 **Grounding**: Not supported by this model")
                    if aspect_ratio in ("4:1", "1:4", "8:1", "1:8"):
                        summary_lines.append(f"📐 **Extreme Aspect Ratio**: {aspect_ratio}")
                if grounding:
                    if grounding.get("search_queries"):
                        queries = ", ".join(grounding["search_queries"])
                        summary_lines.append(f"🔎 **Search queries**: {queries}")
                    if grounding.get("image_search_queries"):
                        image_queries = ", ".join(grounding["image_search_queries"])
                        summary_lines.append(f"🖼️ **Image search queries**: {image_queries}")
                    for source in grounding.get("sources", [])[:5]:
                        summary_lines.append(
                            f"   • {source.get('title') or source['uri']}: {source['uri']}"
                        )
                summary_lines.append("")  # Blank line

                # Add source information based on mode and inputs
                if conversational:
                    summary_lines.append(
                        f"🔁 **Continued from interaction**: {previous_interaction_id}"
                    )
                if input_video_url:
                    summary_lines.append(f"🎬 **Video context**: {input_video_url}")
                if detected_mode == "edit":
                    if file_id:
                        summary_lines.append(f"📎 **Edit Source**: Files API {file_id}")
                    elif input_image_paths and len(input_image_paths) == 1:
                        summary_lines.append(f"📁 **Edit Source**: {input_image_paths[0]}")
                elif input_image_paths:
                    summary_lines.append(
                        f"🖼️ Conditioned on {len(input_image_paths)} input image(s): {', '.join(input_image_paths)}"
                    )
                if aspect_ratio:
                    summary_lines.append(f"📐 Aspect ratio: {aspect_ratio}")

                # Add file information
                result_label = "Edited Images" if detected_mode == "edit" else "Generated Images"
                summary_lines.append(f"\n📁 **{result_label}:**")
                for i, meta in enumerate(metadata, 1):
                    if not meta or not isinstance(meta, dict):
                        summary_lines.append(f"  {i}. ❌ Invalid metadata entry")
                        continue

                    size_bytes = meta.get("size_bytes", 0)
                    size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes else 0
                    full_path = meta.get("full_path", "Unknown path")
                    width = meta.get("width", "?")
                    height = meta.get("height", "?")

                    # Add Files API and parent info for edits
                    extra_info = ""
                    if detected_mode == "edit":
                        files_api_info = meta.get("files_api") or {}
                        if files_api_info.get("name"):
                            extra_info += f" • 🌐 Files API: {files_api_info['name']}"
                        if meta.get("parent_file_id"):
                            extra_info += f" • 👨‍👩‍👧 Parent: {meta.get('parent_file_id')}"
                    if meta.get("interaction_id"):
                        extra_info += f"\n     🔁 interaction_id: `{meta['interaction_id']}`"

                    summary_lines.append(
                        f"  {i}. `{full_path}`\n"
                        f"     📏 {width}x{height} • 💾 {size_mb}MB{extra_info}"
                    )

                if any(m.get("interaction_id") for m in metadata):
                    summary_lines.append(
                        "\n🔁 To keep editing an image, call again with "
                        "previous_interaction_id set to its interaction_id and describe the change."
                    )

                if effective_return_full_image:
                    summary_lines.append(
                        "\n🖼️ **Full-resolution images shown below** (also saved to disk)"
                    )
                else:
                    summary_lines.append(
                        "\n🖼️ **Thumbnail previews shown below** (actual images saved to disk)"
                    )
                full_summary = "\n".join(summary_lines)

                content = [TextContent(type="text", text=full_summary), *thumbnail_images]
            else:
                # Fallback if no images generated
                summary = "❌ No images were generated. Please check the logs for details."
                content = [TextContent(type="text", text=summary)]

            structured_content = {
                "mode": detected_mode,
                "return_full_image": bool(effective_return_full_image),
                "model_tier": selected_tier.value,
                "model_name": model_info["name"],
                "model_id": model_info["model_id"],
                "requested_tier": model_tier,
                "auto_selected": tier == ModelTier.AUTO,
                "thinking_level": thinking_level if selected_tier in THINKING_TIERS else None,
                "resolution": resolution,
                "grounding_enabled": enable_grounding
                if selected_tier in (ModelTier.PRO, ModelTier.NB2)
                else False,
                "grounding": grounding,
                "image_search": image_search,
                "input_video_url": input_video_url,
                "previous_interaction_id": previous_interaction_id,
                "interaction_id": next(
                    (m.get("interaction_id") for m in metadata if m and m.get("interaction_id")),
                    None,
                ),
                "requested": n,
                "returned": len(thumbnail_images),
                "negative_prompt_applied": bool(negative_prompt),
                "used_input_images": bool(input_image_paths) or bool(file_id),
                "input_image_paths": input_image_paths or [],
                "input_image_count": (
                    len(input_image_paths) if input_image_paths else (1 if file_id else 0)
                ),
                "aspect_ratio": aspect_ratio,
                "output_path": output_path,
                "source_file_id": file_id,
                "edit_instruction": prompt if detected_mode == "edit" else None,
                "generation_prompt": prompt if detected_mode == "generate" else None,
                "output_method": "file_system_with_files_api",
                "workflow": f"workflows.md_{detected_mode}_sequence",
                "images": metadata,
                "file_paths": [
                    m.get("full_path")
                    for m in metadata
                    if m and isinstance(m, dict) and m.get("full_path")
                ],
                "files_api_ids": [
                    m.get("files_api", {}).get("name")
                    for m in metadata
                    if m
                    and isinstance(m, dict)
                    and m.get("files_api", {})
                    and m.get("files_api", {}).get("name")
                ],
                "parent_relationships": (
                    [
                        (m.get("parent_file_id"), m.get("files_api", {}).get("name"))
                        for m in metadata
                        if m and isinstance(m, dict)
                    ]
                    if detected_mode == "edit"
                    else []
                ),
                "total_size_mb": round(
                    sum(m.get("size_bytes", 0) for m in metadata if m and isinstance(m, dict))
                    / (1024 * 1024),
                    2,
                ),
            }

            action_verb = "edited" if detected_mode == "edit" else "generated"
            logger.info(
                f"Successfully {action_verb} {len(thumbnail_images)} images in {detected_mode} mode"
            )

            return ToolResult(content=content, structured_content=structured_content)

        except ValidationError as e:
            logger.error(f"Validation error in generate_image: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_image: {e}")
            raise


def _get_enhanced_image_service():
    """Get the enhanced image service instance."""
    from ..services import get_enhanced_image_service

    return get_enhanced_image_service()
