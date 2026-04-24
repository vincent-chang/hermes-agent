"""Send File Tool -- send files (documents, images, audio, video) to messaging platforms.

Provides ``send_file`` for sending local files to any connected messaging platform
(Telegram, Discord, Feishu, etc.) without needing MEDIA tags in a text message.
File type is auto-detected from extension and routed to the appropriate adapter method.

Supported file types:
  - Images: .jpg, .jpeg, .png, .webp, .gif  → send_image_file / send_photo
  - Video:  .mp4, .mov, .avi, .mkv, .3gp   → send_video
  - Audio:  .mp3, .wav, .ogg, .opus, .m4a  → send_voice / send_audio
  - Other:  (all other extensions)           → send_document
"""

import logging
import os

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".opus", ".m4a"}
_VOICE_EXTS = {".ogg", ".opus"}  # Telegram voice format


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

SEND_FILE_SCHEMA = {
    "name": "send_file",
    "description": (
        "Send a local file (document, image, audio, or video) to a connected "
        "messaging platform. File type is auto-detected from the file extension "
        "and routed to the appropriate platform method. "
        "Use this when the user wants to send a specific file without embedding "
        "it in a text message via MEDIA tags."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Delivery target. Format: 'platform' (uses home channel), "
                    "'platform:chat_id', or 'platform:chat_id:thread_id'. "
                    "Examples: 'feishu', 'feishu:oc_xxx', 'telegram:-1001234567890:17585'. "
                    "Omit platform to use the platform of the current session."
                ),
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to send.",
            },
            "caption": {
                "type": "string",
                "description": "Optional caption or message text to send alongside the file.",
            },
        },
        "required": ["file_path"],
    },
}


# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------

def send_file_tool(args, **kw):
    """Handle send_file tool calls (sync wrapper around async impl)."""
    file_path = args.get("file_path")
    if not file_path:
        return tool_error("'file_path' is required")

    if not os.path.exists(file_path):
        return tool_error(f"File not found: {file_path}")

    target = args.get("target") or ""
    caption = args.get("caption") or None

    # Defer async work to a thread pool to keep the tool sync at this level.
    try:
        from model_tools import _run_async
    except ImportError:
        return tool_error("model_tools not available — run inside the agent context")

    try:
        result = _run_async(_send_file_async(target, file_path, caption))
        return result
    except Exception as e:
        return tool_error(f"send_file failed: {e}")


async def _send_file_async(target, file_path, caption):
    """Async implementation: resolve target, load adapter, send file."""
    from gateway.config import load_gateway_config, Platform, PlatformConfig

    platform_name, chat_id, thread_id = _resolve_target(target)

    if not platform_name:
        return {"error": "Could not determine platform. Provide 'target' in 'platform:chat_id' format."}

    # Load gateway config
    try:
        config = load_gateway_config()
    except Exception as e:
        return {"error": f"Failed to load gateway config: {e}"}

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "feishu": Platform.FEISHU,
    }
    platform = platform_map.get(platform_name)
    if not platform:
        avail = ", ".join(platform_map.keys())
        return {"error": f"Unknown or unsupported platform: {platform_name}. Available: {avail}"}

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return {"error": f"Platform '{platform_name}' is not enabled in gateway config."}

    ext = os.path.splitext(file_path)[1].lower()
    metadata = {"thread_id": thread_id} if thread_id else None

    # -------------------------------------------------------------------------
    # Telegram
    # -------------------------------------------------------------------------
    if platform == Platform.TELEGRAM:
        try:
            from gateway.platforms.telegram import TelegramAdapter, TELEGRAM_AVAILABLE
            if not TELEGRAM_AVAILABLE:
                return {"error": "Telegram dependencies not installed. Run: pip install python-telegram-bot"}
        except ImportError:
            return {"error": "Telegram dependencies not installed. Run: pip install python-telegram-bot"}

        try:
            adapter = TelegramAdapter(pconfig)
            connected = await adapter.connect()
            if not connected:
                return {"error": "Telegram: failed to connect"}
            try:
                with open(file_path, "rb") as f:
                    if ext in _IMAGE_EXTS:
                        last_result = await adapter.send_photo(chat_id=int(chat_id), photo=f, caption=caption)
                    elif ext in _VIDEO_EXTS:
                        last_result = await adapter.send_video(chat_id=int(chat_id), video=f, caption=caption)
                    elif ext in _VOICE_EXTS:
                        last_result = await adapter.send_voice(chat_id=int(chat_id), voice=f)
                    elif ext in _AUDIO_EXTS:
                        last_result = await adapter.send_audio(chat_id=int(chat_id), audio=f)
                    else:
                        last_result = await adapter.send_document(chat_id=int(chat_id), document=f, caption=caption)
                if not last_result:
                    return {"error": "Telegram send returned no result"}
                return {
                    "success": True,
                    "platform": "telegram",
                    "chat_id": chat_id,
                    "message_id": str(last_result.message_id),
                }
            finally:
                await adapter.disconnect()
        except Exception as e:
            return {"error": f"Telegram send failed: {e}"}

    # -------------------------------------------------------------------------
    # Feishu
    # -------------------------------------------------------------------------
    if platform == Platform.FEISHU:
        try:
            from gateway.platforms.feishu import FeishuAdapter, FEISHU_AVAILABLE
            if not FEISHU_AVAILABLE:
                return {"error": "Feishu dependencies not installed. Run: pip install 'hermes-agent[feishu]'"}
            from gateway.platforms.feishu import FEISHU_DOMAIN, LARK_DOMAIN
        except ImportError:
            return {"error": "Feishu dependencies not installed. Run: pip install 'hermes-agent[feishu]'"}

        try:
            adapter = FeishuAdapter(pconfig)
            domain_name = getattr(adapter, "_domain_name", "feishu")
            domain = FEISHU_DOMAIN if domain_name != "lark" else LARK_DOMAIN
            adapter._client = adapter._build_lark_client(domain)

            if ext in _IMAGE_EXTS:
                last_result = await adapter.send_image_file(chat_id, file_path, caption=caption, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                last_result = await adapter.send_video(chat_id, file_path, caption=caption, metadata=metadata)
            elif ext in _VOICE_EXTS:
                last_result = await adapter.send_voice(chat_id, file_path, caption=caption, metadata=metadata)
            elif ext in _AUDIO_EXTS:
                last_result = await adapter.send_voice(chat_id, file_path, caption=caption, metadata=metadata)
            else:
                last_result = await adapter.send_document(chat_id, file_path, caption=caption, metadata=metadata)

            if not last_result.success:
                return {"error": f"Feishu send failed: {last_result.error}"}
            return {
                "success": True,
                "platform": "feishu",
                "chat_id": chat_id,
                "message_id": last_result.message_id,
            }
        except Exception as e:
            return {"error": f"Feishu send failed: {e}"}

    # -------------------------------------------------------------------------
    # Discord
    # -------------------------------------------------------------------------
    if platform == Platform.DISCORD:
        try:
            from gateway.platforms.discord import DiscordAdapter, DISCORD_AVAILABLE
            if not DISCORD_AVAILABLE:
                return {"error": "Discord dependencies not installed. Run: pip install discord.py"}
        except ImportError:
            return {"error": "Discord dependencies not installed. Run: pip install discord.py"}

        try:
            adapter = DiscordAdapter(pconfig)
            connected = await adapter.connect()
            if not connected:
                return {"error": "Discord: failed to connect"}
            try:
                if ext in _IMAGE_EXTS:
                    last_result = await adapter.send_image_file(chat_id, file_path, metadata=metadata)
                elif ext in _VIDEO_EXTS:
                    last_result = await adapter.send_video(chat_id, file_path, metadata=metadata)
                elif ext in _AUDIO_EXTS:
                    last_result = await adapter.send_audio(chat_id, file_path, metadata=metadata)
                else:
                    last_result = await adapter.send_document(chat_id, file_path, metadata=metadata)
                if not last_result.success:
                    return {"error": f"Discord send failed: {last_result.error}"}
                return {
                    "success": True,
                    "platform": "discord",
                    "chat_id": chat_id,
                    "message_id": last_result.message_id,
                }
            finally:
                await adapter.disconnect()
        except Exception as e:
            return {"error": f"Discord send failed: {e}"}

    return {"error": f"send_file is not implemented for platform '{platform_name}'"}


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
# Registry
# -----------------------------------------------------------------------------

registry.register(
    name="send_file",
    toolset="messaging",
    schema=SEND_FILE_SCHEMA,
    handler=send_file_tool,
    check_fn=None,
    emoji="📎",
)
