"""Send Streaming Card Tool -- send Feishu interactive cards with real-time text updates.

Provides a 3-step flow:
  1. send_streaming_card  → creates card entity + sends it, returns card_id
  2. update_streaming_text → pushes text chunks (call repeatedly)
  3. disable_streaming_mode → finalizes the card

Supported only on Feishu. Use ``target`` in 'feishu' or 'feishu:chat_id' format.
"""

import logging

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

STREAMING_CARD_SCHEMA = {
    "name": "send_streaming_card",
    "description": (
        "Send a Feishu interactive card with streaming (real-time typewriter) text updates. "
        "Supported only on Feishu. "
        "Step 1 of the streaming card flow: creates the card entity and sends it to the target chat. "
        "Returns card_id and message_id needed for subsequent update_streaming_card_text calls. "
        "After this, repeatedly call update_streaming_card_text to push content, "
        "then call disable_streaming_card to finalize."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Delivery target in 'platform:chat_id' format. "
                    "Examples: 'feishu', 'feishu:oc_xxx'. "
                    "Uses current session's chat if platform only is provided."
                ),
            },
            "header_title": {
                "type": "string",
                "description": "Card header title text.",
            },
            "header_template": {
                "type": "string",
                "description": "Card header color template. One of: 'blue', 'red', 'orange', 'yellow', 'green', 'purple', 'gray', 'default'. Default: 'blue'.",
                "default": "blue",
            },
            "initial_content": {
                "type": "string",
                "description": "Initial markdown text shown in the card body before any updates.",
                "default": "",
            },
            "element_id": {
                "type": "string",
                "description": "Element ID of the markdown text element to update. Default: 'progress_text'.",
                "default": "progress_text",
            },
            "summary": {
                "type": "string",
                "description": "Summary text shown in the Feishu message list before the card loads.",
                "default": "...",
            },
            "print_frequency_ms": {
                "type": "integer",
                "description": "Streaming speed: milliseconds between character batches. Lower = faster. Default: 50.",
                "default": 50,
            },
            "print_step": {
                "type": "integer",
                "description": "How many characters to print per batch. Default: 2.",
                "default": 2,
            },
        },
        "required": ["target", "header_title"],
    },
}

UPDATE_STREAMING_TEXT_SCHEMA = {
    "name": "update_streaming_card_text",
    "description": (
        "Push a text update to a live Feishu streaming card (typewriter effect). "
        "Call this repeatedly after send_streaming_card to show progressive content. "
        "Each call replaces/appends to the card's text element via the streaming API. "
        "Use uuid for ordered delivery (e.g., '1', '2', '3'...)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card_id": {
                "type": "string",
                "description": "The card_id returned by send_streaming_card.",
            },
            "element_id": {
                "type": "string",
                "description": "The element_id of the text element to update. Must match what was passed to send_streaming_card.",
                "default": "progress_text",
            },
            "content": {
                "type": "string",
                "description": "The new text content. For streaming, append to existing content to get typewriter effect.",
            },
            "uuid": {
                "type": "string",
                "description": (
                    "Optional sequence identifier for ordered delivery. "
                    "Use an incrementing string (e.g., '1', '2', '3'). "
                    "Feishu delivers updates in uuid order. Required for reliable ordering."
                ),
            },
        },
        "required": ["card_id", "content"],
    },
}

DISABLE_STREAMING_CARD_SCHEMA = {
    "name": "disable_streaming_card",
    "description": (
        "Close streaming mode on a Feishu card after all update_streaming_card_text calls are done. "
        "This finalizes the card, stops the streaming animation, and optionally sets a summary. "
        "Call this once after all content has been pushed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card_id": {
                "type": "string",
                "description": "The card_id returned by send_streaming_card.",
            },
            "final_summary": {
                "type": "string",
                "description": "Optional summary text to replace the default summary after streaming ends.",
            },
        },
        "required": ["card_id"],
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_feishu_adapter(pconfig):
    """Create and connect a FeishuAdapter instance."""
    from gateway.platforms.feishu import FeishuAdapter, FEISHU_AVAILABLE, FEISHU_DOMAIN, LARK_DOMAIN

    if not FEISHU_AVAILABLE:
        raise ImportError("Feishu dependencies not installed")

    adapter = FeishuAdapter(pconfig)
    domain_name = getattr(adapter, "_domain_name", "feishu")
    domain = FEISHU_DOMAIN if domain_name != "lark" else LARK_DOMAIN
    adapter._client = adapter._build_lark_client(domain)
    return adapter


def _resolve_target(target):
    """Parse 'platform:chat_id' target string. Returns (platform, chat_id, thread_id)."""
    if not target:
        return None, None, None
    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    rest = parts[1].strip() if len(parts) > 1 else None
    chat_id = None
    thread_id = None
    if rest:
        sub_parts = rest.split(":", 1)
        chat_id = sub_parts[0]
        thread_id = sub_parts[1] if len(sub_parts) > 1 else None
    return platform_name, chat_id, thread_id


# -----------------------------------------------------------------------------
# Tool handlers
# -----------------------------------------------------------------------------


def send_streaming_card_tool(args, **kw):
    """Handle send_streaming_card tool calls."""
    target = args.get("target") or "feishu"
    header_title = args.get("header_title", "")
    header_template = args.get("header_template", "blue")
    initial_content = args.get("initial_content", "")
    element_id = args.get("element_id", "progress_text")
    summary = args.get("summary", "...")
    print_frequency_ms = args.get("print_frequency_ms", 50)
    print_step = args.get("print_step", 2)

    try:
        from model_tools import _run_async
    except ImportError:
        return tool_error("model_tools not available — run inside the agent context")

    platform_name, chat_id, thread_id = _resolve_target(target)
    if not platform_name:
        return tool_error("Could not determine platform from target")

    if platform_name != "feishu":
        return tool_error(f"Streaming cards are only supported on Feishu, got: {platform_name}")

    try:
        from gateway.config import load_gateway_config, Platform
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    try:
        config = load_gateway_config()
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    pconfig = config.platforms.get(Platform.FEISHU)
    if not pconfig or not pconfig.enabled:
        return tool_error("Feishu platform is not enabled in gateway config")

    if not chat_id:
        return tool_error("chat_id is required. Provide target as 'feishu:chat_id'")

    try:
        adapter = _get_feishu_adapter(pconfig)
        metadata = {"thread_id": thread_id} if thread_id else None

        result = _run_async(
            adapter.send_streaming_card(
                chat_id=chat_id,
                header_title=header_title,
                header_template=header_template,
                initial_content=initial_content,
                element_id=element_id,
                summary=summary,
                print_frequency_ms=print_frequency_ms,
                print_step=print_step,
                metadata=metadata,
            )
        )
        card_id, message_id, error = result
        if error:
            return tool_error(f"send_streaming_card failed: {error}")
        return {
            "success": True,
            "card_id": card_id,
            "message_id": message_id,
            "element_id": element_id,
            "chat_id": chat_id,
        }
    except Exception as e:
        return tool_error(f"send_streaming_card failed: {e}")


def update_streaming_card_text_tool(args, **kw):
    """Handle update_streaming_card_text tool calls."""
    card_id = args.get("card_id")
    element_id = args.get("element_id", "progress_text")
    content = args.get("content", "")
    uuid = args.get("uuid")

    if not card_id:
        return tool_error("card_id is required")
    if not content:
        return tool_error("content is required")

    try:
        from model_tools import _run_async
    except ImportError:
        return tool_error("model_tools not available")

    try:
        from gateway.config import load_gateway_config, Platform
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    try:
        config = load_gateway_config()
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    pconfig = config.platforms.get(Platform.FEISHU)
    if not pconfig or not pconfig.enabled:
        return tool_error("Feishu platform is not enabled")

    try:
        adapter = _get_feishu_adapter(pconfig)
        error = _run_async(
            adapter.update_streaming_text(
                card_id=card_id,
                element_id=element_id,
                content=content,
                uuid=uuid,
            )
        )
        if error:
            return tool_error(f"update_streaming_card_text failed: {error}")
        return {"success": True, "card_id": card_id, "element_id": element_id}
    except Exception as e:
        return tool_error(f"update_streaming_card_text failed: {e}")


def disable_streaming_card_tool(args, **kw):
    """Handle disable_streaming_card tool calls."""
    card_id = args.get("card_id")
    final_summary = args.get("final_summary")

    if not card_id:
        return tool_error("card_id is required")

    try:
        from model_tools import _run_async
    except ImportError:
        return tool_error("model_tools not available")

    try:
        from gateway.config import load_gateway_config, Platform
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    try:
        config = load_gateway_config()
    except Exception as e:
        return tool_error(f"Failed to load gateway config: {e}")

    pconfig = config.platforms.get(Platform.FEISHU)
    if not pconfig or not pconfig.enabled:
        return tool_error("Feishu platform is not enabled")

    try:
        adapter = _get_feishu_adapter(pconfig)
        error = _run_async(
            adapter.disable_streaming_mode(card_id=card_id, final_summary=final_summary)
        )
        if error:
            return tool_error(f"disable_streaming_card failed: {error}")
        return {"success": True, "card_id": card_id}
    except Exception as e:
        return tool_error(f"disable_streaming_card failed: {e}")


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

registry.register(
    name="send_streaming_card",
    toolset="messaging",
    schema=STREAMING_CARD_SCHEMA,
    handler=send_streaming_card_tool,
    check_fn=None,
    emoji="📦",
)

registry.register(
    name="update_streaming_card_text",
    toolset="messaging",
    schema=UPDATE_STREAMING_TEXT_SCHEMA,
    handler=update_streaming_card_text_tool,
    check_fn=None,
    emoji="✏️",
)

registry.register(
    name="disable_streaming_card",
    toolset="messaging",
    schema=DISABLE_STREAMING_CARD_SCHEMA,
    handler=disable_streaming_card_tool,
    check_fn=None,
    emoji="🔒",
)
