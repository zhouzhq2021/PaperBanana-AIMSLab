# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
API 调用工具函数，支持 Evolink、Gemini、Claude、OpenAI 等多种 Provider。
"""

import json
import asyncio
import base64
import re
import time
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any, Optional

from PIL import Image

import os
import yaml
from pathlib import Path

# ==================== 配置加载 ====================

config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    try:
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        print(f"警告：配置文件 {config_path} 解析失败，将仅使用环境变量和默认值。错误：{e}")

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


def get_first_config_val(*lookups, default=""):
    for section, key, env_var in lookups:
        val = get_config_val(section, key, env_var, "")
        if val:
            return val
    return default


GATEWAY_PROVIDERS = {"aipaibox", "evolink", "gateway"}
AUTO_PROVIDER = "auto"
OPENAI_IMAGE_QUALITIES = {"standard", "hd", "low", "medium", "high", "auto"}
DEBUG_LOGS = os.getenv("PAPERBANANA_DEBUG", "").lower() in {"1", "true", "yes", "on"}
AUTO_PROVIDER_ATTEMPTS = int(os.getenv("PAPERBANANA_AUTO_PROVIDER_ATTEMPTS", "2"))
AUTO_PROVIDER_RETRY_DELAY = float(os.getenv("PAPERBANANA_AUTO_PROVIDER_RETRY_DELAY", "2"))
AUTO_PROVIDER_COOLDOWN_SECONDS = int(os.getenv("PAPERBANANA_AUTO_PROVIDER_COOLDOWN_SECONDS", "300"))
DEFAULT_TEXT_MODEL_FALLBACKS = [
    "gpt-5.5",
    "gpt-5.4",
    "gemini-2.5-flash",
]
TEXT_MODEL_FALLBACKS = [
    item.strip()
    for item in os.getenv(
        "PAPERBANANA_TEXT_MODEL_FALLBACKS",
        ",".join(DEFAULT_TEXT_MODEL_FALLBACKS),
    ).split(",")
    if item.strip()
]
_provider_unhealthy_until: Dict[tuple[str, str, str], float] = {}


def _debug_log(message: str):
    if DEBUG_LOGS:
        print(message)


def _normalize_provider(provider: str = "") -> str:
    return (provider or "").strip().lower()


def is_gateway_provider(provider: str = "") -> bool:
    return _normalize_provider(provider) in GATEWAY_PROVIDERS


def is_gemini_model(model_name: str = "") -> bool:
    name = (model_name or "").lower()
    return "gemini" in name


def is_openai_text_model(model_name: str = "") -> bool:
    name = (model_name or "").lower()
    return name.startswith(("gpt-", "gpt5", "o1", "o3", "o4", "chatgpt-")) or bool(
        re.match(r"^gpt[-_ ]?\d", name)
    ) or name in {
        "gpt4",
        "gpt-4",
    }


def is_openai_image_model(model_name: str = "") -> bool:
    return (model_name or "").lower().startswith("gpt-image")


def _is_auto_provider(provider: str = "") -> bool:
    return _normalize_provider(provider) in {"", AUTO_PROVIDER}


def _is_bad_response(result) -> bool:
    if not result:
        return True
    return all((not item) or str(item).strip() == "Error" for item in result)


def _is_provider_available(provider: str, kind: str, model_name: str = "") -> bool:
    provider_name = _normalize_provider(provider)
    if is_gateway_provider(provider_name):
        return _resolve_gateway_provider(provider_name, model_name) is not None
    if provider_name == "gemini":
        return gemini_client is not None
    if provider_name == "openai":
        return openai_client is not None
    if provider_name == "anthropic":
        return kind == "text" and anthropic_client is not None
    return False


def _is_provider_healthy(provider: str, kind: str, model_name: str = "") -> bool:
    unhealthy_until = _provider_unhealthy_until.get(
        (_normalize_provider(provider), kind, model_name), 0
    )
    return time.monotonic() >= unhealthy_until


def _mark_provider_unhealthy(provider: str, kind: str, model_name: str = "", reason: str = ""):
    provider_name = _normalize_provider(provider)
    _provider_unhealthy_until[(provider_name, kind, model_name)] = (
        time.monotonic() + AUTO_PROVIDER_COOLDOWN_SECONDS
    )
    reason_msg = f": {reason}" if reason else ""
    print(
        f"[自动通道] 暂停 {kind} 组合 {model_name}@{provider_name} "
        f"{AUTO_PROVIDER_COOLDOWN_SECONDS}s{reason_msg}"
    )


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _text_provider_candidates(model_name: str) -> List[str]:
    if is_gemini_model(model_name):
        return _dedupe(["aipaibox", "gemini"])
    if is_openai_text_model(model_name):
        return _dedupe(["aipaibox", "openai"])
    return _dedupe(["aipaibox", "openai", "gemini", "anthropic"])


def _image_provider_candidates(model_name: str) -> List[str]:
    if is_openai_image_model(model_name):
        return _dedupe(["aipaibox", "openai"])
    if is_gemini_model(model_name):
        return _dedupe(["aipaibox", "gemini"])
    return _dedupe(["aipaibox"])


def _text_model_candidates(model_name: str) -> List[str]:
    return _dedupe([model_name, *TEXT_MODEL_FALLBACKS])


def _extract_text_config(config) -> Dict[str, Any]:
    """Extract common generation options from dict or google GenerateContentConfig."""
    if hasattr(config, "system_instruction"):
        return {
            "system_prompt": config.system_instruction or "",
            "temperature": getattr(config, "temperature", 1.0),
            "candidate_num": getattr(config, "candidate_count", 1) or 1,
            "max_output_tokens": getattr(config, "max_output_tokens", 50000) or 50000,
            "max_completion_tokens": getattr(config, "max_output_tokens", 50000) or 50000,
        }
    if isinstance(config, dict):
        max_tokens = (
            config.get("max_completion_tokens")
            or config.get("max_output_tokens")
            or config.get("max_tokens")
            or 50000
        )
        return {
            "system_prompt": config.get("system_prompt", ""),
            "temperature": config.get("temperature", 1.0),
            "candidate_num": config.get("candidate_num", config.get("candidate_count", 1)),
            "max_output_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
        }
    return {
        "system_prompt": "",
        "temperature": 1.0,
        "candidate_num": 1,
        "max_output_tokens": 50000,
        "max_completion_tokens": 50000,
    }

# ==================== OpenAI-compatible Gateway Provider 初始化 ====================

evolink_provider = None
aipaibox_providers: Dict[str, Any] = {}
legacy_gateway_provider = None


def _gateway_family_for_model(model_name: str = "") -> str:
    if is_gemini_model(model_name):
        return "gemini"
    if is_openai_text_model(model_name) or is_openai_image_model(model_name):
        return "openai"
    return "default"


def _first_gateway_provider():
    return (
        aipaibox_providers.get("default")
        or aipaibox_providers.get("gemini")
        or aipaibox_providers.get("openai")
        or legacy_gateway_provider
    )


def _refresh_gateway_alias():
    """Keep the old evolink_provider name usable for compatibility and cleanup."""
    global evolink_provider
    evolink_provider = _first_gateway_provider()


def _resolve_gateway_provider(provider: str = "gateway", model_name: str = ""):
    provider_name = _normalize_provider(provider)
    if provider_name == "aipaibox":
        family = _gateway_family_for_model(model_name)
        if family == "default":
            return (
                aipaibox_providers.get("default")
                or aipaibox_providers.get("openai")
                or aipaibox_providers.get("gemini")
            )
        return aipaibox_providers.get(family) or aipaibox_providers.get("default")
    if provider_name == "evolink":
        return legacy_gateway_provider
    if provider_name == "gateway":
        return _resolve_gateway_provider("aipaibox", model_name) or legacy_gateway_provider
    return None


def _new_gateway_provider(api_key: str, base_url: str):
    from providers.evolink import EvolinkProvider
    return EvolinkProvider(api_key=api_key, base_url=base_url)

aipaibox_base_url = get_config_val(
    "aipaibox", "base_url", "AIPAIBOX_BASE_URL", "https://api.aipaibox.com"
)
aipaibox_api_key = get_config_val("aipaibox", "api_key", "AIPAIBOX_API_KEY", "")
aipaibox_gemini_api_key = get_first_config_val(
    ("aipaibox", "gemini_api_key", "AIPAIBOX_GEMINI_API_KEY"),
    ("aipaibox", "api_key", "AIPAIBOX_API_KEY"),
    default="",
)
aipaibox_openai_api_key = get_first_config_val(
    ("aipaibox", "openai_api_key", "AIPAIBOX_OPENAI_API_KEY"),
    ("aipaibox", "api_key", "AIPAIBOX_API_KEY"),
    default="",
)
legacy_evolink_api_key = get_config_val("evolink", "api_key", "EVOLINK_API_KEY", "")
legacy_gateway_base_url = get_config_val(
    "evolink", "base_url", "EVOLINK_BASE_URL", "https://api.evolink.ai"
)

try:
    if aipaibox_api_key:
        aipaibox_providers["default"] = _new_gateway_provider(aipaibox_api_key, aipaibox_base_url)
    if aipaibox_gemini_api_key and aipaibox_gemini_api_key != aipaibox_api_key:
        aipaibox_providers["gemini"] = _new_gateway_provider(aipaibox_gemini_api_key, aipaibox_base_url)
    if aipaibox_openai_api_key and aipaibox_openai_api_key != aipaibox_api_key:
        aipaibox_providers["openai"] = _new_gateway_provider(aipaibox_openai_api_key, aipaibox_base_url)
    if legacy_evolink_api_key:
        legacy_gateway_provider = _new_gateway_provider(legacy_evolink_api_key, legacy_gateway_base_url)
    _refresh_gateway_alias()
    if aipaibox_providers:
        configured_families = ", ".join(sorted(aipaibox_providers.keys()))
        print(f"已初始化 AIPAIBOX Gateway Provider ({configured_families}, base_url={aipaibox_base_url})")
    if legacy_gateway_provider:
        print(f"已初始化 Evolink Gateway Provider (base_url={legacy_gateway_base_url})")
except ImportError:
    print("警告：未安装 aiohttp，API Gateway Provider 不可用。请运行 pip install aiohttp")

if not _first_gateway_provider():
    print("警告：未配置 AIPAIBOX/EVOLINK API Key，API Gateway Provider 不可用。")


def init_evolink_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 Legacy OpenAI-compatible Gateway Provider。"""
    global legacy_gateway_provider
    if not api_key:
        return
    url = base_url or legacy_gateway_base_url
    try:
        legacy_gateway_provider = _new_gateway_provider(api_key, url)
        _refresh_gateway_alias()
        print(f"已通过界面初始化 Evolink Gateway Provider (base_url={url})")
    except ImportError:
        print("警告：未安装 aiohttp，API Gateway Provider 不可用。请运行 pip install aiohttp")


def init_aipaibox_provider(
    api_key: str = "",
    base_url: str = "",
    gemini_api_key: str = "",
    openai_api_key: str = "",
):
    """初始化或更新 AIPAIBOX Provider，支持 Gemini/OpenAI 模型使用不同 key。"""
    global aipaibox_providers
    url = base_url or aipaibox_base_url
    if not any([api_key, gemini_api_key, openai_api_key]):
        return
    try:
        updated = {}
        if api_key:
            updated["default"] = _new_gateway_provider(api_key, url)
        if gemini_api_key:
            updated["gemini"] = _new_gateway_provider(gemini_api_key, url)
        if openai_api_key:
            updated["openai"] = _new_gateway_provider(openai_api_key, url)
        aipaibox_providers.update(updated)
        _refresh_gateway_alias()
        configured_families = ", ".join(sorted(updated.keys()))
        print(f"已通过界面初始化 AIPAIBOX Gateway Provider ({configured_families}, base_url={url})")
    except ImportError:
        print("警告：未安装 aiohttp，API Gateway Provider 不可用。请运行 pip install aiohttp")


async def close_gateway_providers():
    """Close all initialized gateway sessions once a Streamlit run finishes."""
    seen = set()
    providers = [*aipaibox_providers.values(), legacy_gateway_provider]
    for provider in providers:
        if provider is None or id(provider) in seen or not hasattr(provider, "close"):
            continue
        seen.add(id(provider))
        await provider.close()


google_base_url = get_config_val("google", "base_url", "GOOGLE_BASE_URL", "")
google_api_version = get_config_val("google", "api_version", "GOOGLE_API_VERSION", "")


def _create_gemini_client(api_key: str, base_url: str = "", api_version: str = ""):
    from google import genai
    from google.genai import types

    http_options_kwargs = {}
    if base_url:
        http_options_kwargs["baseUrl"] = base_url
    if api_version:
        http_options_kwargs["apiVersion"] = api_version

    if http_options_kwargs:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_options_kwargs),
        )
    return genai.Client(api_key=api_key)


def init_gemini_client(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 Gemini Client（供界面动态传入）。"""
    global gemini_client
    if not api_key:
        return
    try:
        url = base_url or google_base_url
        gemini_client = _create_gemini_client(
            api_key=api_key,
            base_url=url,
            api_version=google_api_version,
        )
        if url:
            print(f"已通过界面初始化 Gemini Client (base_url={url})")
        else:
            print("已通过界面初始化 Gemini Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")


# ==================== 原始 Provider 初始化（保留兼容性） ====================

gemini_client = None
anthropic_client = None
openai_client = None

api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    try:
        gemini_client = _create_gemini_client(
            api_key=api_key,
            base_url=google_base_url,
            api_version=google_api_version,
        )
        if google_base_url:
            print(f"已初始化 Gemini Client (base_url={google_base_url})")
        else:
            print("已初始化 Gemini Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。")

anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    try:
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        print("已初始化 Anthropic Client")
    except ImportError:
        print("警告：未安装 anthropic，Anthropic Client 不可用。")

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
openai_base_url = get_config_val("openai", "base_url", "OPENAI_BASE_URL", "")
if openai_api_key:
    try:
        from openai import AsyncOpenAI
        openai_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url
        openai_client = AsyncOpenAI(**openai_kwargs)
        if openai_base_url:
            print(f"已初始化 OpenAI Client (base_url={openai_base_url})")
        else:
            print("已初始化 OpenAI Client")
    except ImportError:
        print("警告：未安装 openai，OpenAI Client 不可用。")


def init_openai_client(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 OpenAI Client（供界面动态传入）。"""
    global openai_client
    if not api_key:
        return
    try:
        from openai import AsyncOpenAI
        url = base_url or openai_base_url
        openai_kwargs = {"api_key": api_key}
        if url:
            openai_kwargs["base_url"] = url
        openai_client = AsyncOpenAI(**openai_kwargs)
        if url:
            print(f"已通过界面初始化 OpenAI Client (base_url={url})")
        else:
            print("已通过界面初始化 OpenAI Client")
    except ImportError:
        print("警告：未安装 openai，OpenAI Client 不可用。请运行 pip install openai")


# ==================== API Gateway 调用函数 ====================

async def call_evolink_text_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context="", provider_name="gateway"
):
    """
    通过 OpenAI-compatible API Gateway Provider 进行文本生成。

    Args:
        model_name: 模型名称（如 "gemini-2.5-flash"）
        contents: 通用内容列表
        config: 配置字典或对象，需包含 system_instruction, temperature, max_output_tokens
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    gateway_provider = _resolve_gateway_provider(provider_name, model_name)
    _debug_log(
        f"[DEBUG] call_api_gateway_text: model={model_name}, provider={provider_name}, "
        f"state={'已初始化' if gateway_provider else '未初始化'}"
    )
    if gateway_provider is None:
        raise RuntimeError(
            "API Gateway Provider 未初始化，请检查 AIPAIBOX_GEMINI_API_KEY/"
            "AIPAIBOX_OPENAI_API_KEY/EVOLINK_API_KEY 配置。"
        )

    # 从 config 中提取参数（兼容 types.GenerateContentConfig 和 dict）
    if hasattr(config, 'system_instruction'):
        system_prompt = config.system_instruction or ""
        temperature = config.temperature
        max_output_tokens = config.max_output_tokens
        _debug_log(f"[DEBUG] call_api_gateway_text: 从 GenerateContentConfig 提取参数")
    elif isinstance(config, dict):
        system_prompt = config.get("system_prompt", "")
        temperature = config.get("temperature", 1.0)
        max_output_tokens = config.get("max_output_tokens", 50000)
        _debug_log(f"[DEBUG] call_api_gateway_text: 从 dict 提取参数")
    else:
        system_prompt = ""
        temperature = 1.0
        max_output_tokens = 50000
        _debug_log(f"[DEBUG] call_api_gateway_text: 使用默认参数, config type={type(config)}")

    return await gateway_provider.generate_text(
        model_name=model_name,
        contents=contents,
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def upload_image_to_evolink(
    image_b64: str,
    media_type: str = "image/jpeg",
    model_name: str = "",
    provider_name: str = "gateway",
) -> str:
    """
    将 base64 图片上传到 API Gateway 文件服务，返回可访问的 URL。

    用于 image-to-image 场景（如 Polish Agent），需要先把本地 base64 图片
    上传为 URL，才能传给图像生成 API 的 image_urls 参数。
    """
    gateway_provider = _resolve_gateway_provider(provider_name, model_name)
    if gateway_provider is None:
        raise RuntimeError(
            "API Gateway Provider 未初始化，请检查 AIPAIBOX_GEMINI_API_KEY/"
            "AIPAIBOX_OPENAI_API_KEY/EVOLINK_API_KEY 配置。"
        )
    url = await gateway_provider.upload_image_base64(image_b64, media_type)
    if not url:
        raise RuntimeError("图片上传到 API Gateway 文件服务失败")
    return url


async def call_evolink_image_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context="", provider_name="gateway"
):
    """
    通过 OpenAI-compatible API Gateway Provider 进行图像生成。

    Args:
        model_name: 图像模型名称（如 "nano-banana-2-lite"，通过 /v1/images/generations）
        prompt: 图像描述提示词
        config: 配置字典，需包含 aspect_ratio, quality 等
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    gateway_provider = _resolve_gateway_provider(provider_name, model_name)
    _debug_log(
        f"[DEBUG] call_api_gateway_image: model={model_name}, config={config}, provider={provider_name}, "
        f"state={'已初始化' if gateway_provider else '未初始化'}"
    )
    if gateway_provider is None:
        raise RuntimeError(
            "API Gateway Provider 未初始化，请检查 AIPAIBOX_GEMINI_API_KEY/"
            "AIPAIBOX_OPENAI_API_KEY/EVOLINK_API_KEY 配置。"
        )

    aspect_ratio = config.get("aspect_ratio", "16:9")
    quality = config.get("quality", "2K")
    image_urls = config.get("image_urls", None)

    return await gateway_provider.generate_image(
        model_name=model_name,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        quality=quality,
        image_urls=image_urls,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


# ==================== 原始 Gemini 调用函数（保留兼容性） ====================

def _convert_to_gemini_parts(contents):
    """将通用内容列表转换为 Gemini 的 Part 对象列表"""
    from google.genai import types
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """原始 Gemini API 异步调用（保留兼容性）"""
    from google.genai import types

    if gemini_client is None:
        raise RuntimeError("Gemini Client 未初始化，请检查 GOOGLE_API_KEY 配置。")

    result_list = []
    target_candidate_count = config.candidate_count
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            client = gemini_client
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            if "nanoviz" in model_name or "image" in model_name:
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    _debug_log(f"[Gemini] 图像响应为空，{retry_delay}s 后重试")
                    await asyncio.sleep(retry_delay)
                    continue
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 30)
            _debug_log(
                f"[Gemini] Attempt {attempt + 1}/{max_attempts} for {model_name} failed{context_msg}: {e}"
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"[Gemini] {model_name} 重试 {max_attempts} 次后失败{context_msg}: {e}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


# ==================== 原始 Claude/OpenAI 调用函数（保留兼容性） ====================

def _convert_to_claude_format(contents):
    return contents

def _convert_to_openai_format(contents):
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "image_base64" in item:
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


def _extract_openai_image_files(contents: Optional[List[Dict[str, Any]]]) -> List[BytesIO]:
    image_files = []
    if not contents:
        return image_files

    for idx, item in enumerate(contents):
        if item.get("type") != "image":
            continue

        image_b64 = ""
        media_type = "image/jpeg"
        source = item.get("source", {})
        if source.get("type") == "base64":
            image_b64 = source.get("data", "")
            media_type = source.get("media_type", "image/jpeg")
        elif "image_base64" in item:
            image_b64 = item.get("image_base64", "")

        if not image_b64:
            continue

        suffix = "png" if "png" in media_type else "jpg"
        image_file = BytesIO(base64.b64decode(image_b64))
        image_file.name = f"input_{idx}.{suffix}"
        image_files.append(image_file)

    return image_files


async def _upload_content_images_to_gateway(
    contents: Optional[List[Dict[str, Any]]],
    model_name: str = "",
    provider_name: str = "gateway",
) -> List[str]:
    image_urls = []
    if not contents:
        return image_urls

    for item in contents:
        if item.get("type") != "image":
            continue
        source = item.get("source", {})
        media_type = source.get("media_type", "image/jpeg")
        image_b64 = ""
        if source.get("type") == "base64":
            image_b64 = source.get("data", "")
        elif "image_base64" in item:
            image_b64 = item.get("image_base64", "")
        if image_b64:
            image_urls.append(
                await upload_image_to_evolink(
                    image_b64,
                    media_type,
                    model_name=model_name,
                    provider_name=provider_name,
                )
            )

    return image_urls


def _openai_size_from_aspect_ratio(aspect_ratio: str = "16:9") -> str:
    ratio = (aspect_ratio or "").strip()
    if ratio in {"1:1", "square"}:
        return "1024x1024"
    if ratio in {"2:3", "3:4", "9:16"}:
        return "1024x1536"
    return "1536x1024"


def _gateway_openai_size_from_aspect_ratio(aspect_ratio: str = "16:9") -> str:
    """AIPAIBOX gpt-image-* expects WIDTHxHEIGHT instead of ratio strings."""
    ratio = (aspect_ratio or "").strip().lower()
    if re.match(r"^\d+x\d+$", ratio):
        return ratio

    return {
        "1:1": "1024x1024",
        "square": "1024x1024",
        "21:9": "1824x784",
        "16:9": "1824x1024",
        "3:2": "1536x1024",
        "4:3": "1368x1024",
        "2:3": "1024x1536",
        "3:4": "1024x1368",
        "9:16": "1024x1824",
    }.get(ratio, _openai_size_from_aspect_ratio(ratio))


def _gateway_quality_for_model(model_name: str, cfg: Dict[str, Any]) -> str:
    if is_openai_image_model(model_name):
        quality = cfg.get("openai_quality") or cfg.get("quality") or "high"
        return quality if quality in OPENAI_IMAGE_QUALITIES else "high"
    return cfg.get("gateway_quality") or (
        "2K" if cfg.get("quality", "2K") in OPENAI_IMAGE_QUALITIES else cfg.get("quality", "2K")
    )


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 Claude API 异步调用（保留兼容性）"""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            _debug_log(f"[Claude] Attempt {attempt + 1}/{max_attempts} failed{context_msg}: {error_str}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"[Claude] {model_name} 重试 {max_attempts} 次后失败")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI API 异步调用（保留兼容性）"""
    if openai_client is None:
        raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY 配置。")

    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            _debug_log(f"[OpenAI] Attempt {attempt + 1}/{max_attempts} for {model_name} failed{context_msg}: {error_str}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"[OpenAI] {model_name} 重试 {max_attempts} 次后失败")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI 图像生成 API 异步调用（保留兼容性）"""
    if openai_client is None:
        raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY 配置。")

    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "response_format": config.get("response_format", "b64_json"),
    }

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                _debug_log("[OpenAI 图像] 响应为空")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            _debug_log(f"[OpenAI 图像] Attempt {attempt + 1}/{max_attempts} for {model_name} failed{context_msg}: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"[OpenAI 图像] {model_name} 重试 {max_attempts} 次后失败{context_msg}: {e}")
                return ["Error"]

    return ["Error"]


async def call_openai_image_edit_with_retry_async(
    model_name, prompt, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """OpenAI 图像编辑 API 调用，支持 gpt-image-* 的 image-to-image 场景。"""
    if openai_client is None:
        raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY 配置。")

    image_files = _extract_openai_image_files(contents)
    if not image_files:
        return await call_openai_image_generation_with_retry_async(
            model_name=model_name,
            prompt=prompt,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    edit_params = {
        "model": model_name,
        "prompt": prompt,
        "image": image_files if len(image_files) > 1 else image_files[0],
        "n": 1,
        "size": config.get("size", "1536x1024"),
        "quality": config.get("quality", "high"),
        "background": config.get("background", "opaque"),
        "output_format": config.get("output_format", "png"),
        "response_format": config.get("response_format", "b64_json"),
    }

    if "input_fidelity" in config:
        edit_params["input_fidelity"] = config["input_fidelity"]

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.edit(**edit_params)
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            _debug_log("[OpenAI 图像编辑] 响应为空")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            _debug_log(f"[OpenAI 图像编辑] Attempt {attempt + 1}/{max_attempts} for {model_name} failed{context_msg}: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"[OpenAI 图像编辑] {model_name} 重试 {max_attempts} 次后失败{context_msg}: {e}")
                return ["Error"]

    return ["Error"]


# ==================== 通用模型路由 ====================

async def call_text_model_with_retry_async(
    provider: str,
    model_name: str,
    contents,
    config,
    max_attempts=5,
    retry_delay=5,
    error_context="",
):
    """按 provider/model 自动路由文本模型。"""
    provider_name = _normalize_provider(provider)
    cfg = _extract_text_config(config)

    if _is_auto_provider(provider_name):
        model_candidates = _text_model_candidates(model_name)
        attempted = []
        for current_model in model_candidates:
            if current_model != model_name:
                print(f"[自动通道] 文本模型 {model_name} 失败后，尝试备用模型 {current_model}")
            candidates = _text_provider_candidates(current_model)
            for candidate in candidates:
                if not _is_provider_available(candidate, kind="text", model_name=current_model):
                    _debug_log(f"[自动通道] 跳过文本通道 {candidate}: 未配置或未初始化")
                    continue
                if not _is_provider_healthy(candidate, kind="text", model_name=current_model):
                    _debug_log(f"[自动通道] 跳过文本组合 {current_model}@{candidate}: 近期失败，冷却中")
                    continue
                attempted.append(f"{current_model}@{candidate}")
                try:
                    _debug_log(f"[自动通道] 文本模型 {current_model} 尝试使用 {candidate}")
                    result = await call_text_model_with_retry_async(
                        provider=candidate,
                        model_name=current_model,
                        contents=contents,
                        config=config,
                        max_attempts=AUTO_PROVIDER_ATTEMPTS,
                        retry_delay=AUTO_PROVIDER_RETRY_DELAY,
                        error_context=error_context,
                    )
                    if not _is_bad_response(result):
                        print(f"[自动通道] 文本模型 {current_model} 使用 {candidate} 成功")
                        return result
                    print(f"[自动通道] 文本组合 {current_model}@{candidate} 重试后仍失败")
                    _mark_provider_unhealthy(candidate, "text", current_model, "返回 Error")
                except Exception as e:
                    print(f"[自动通道] 文本组合 {current_model}@{candidate} 异常，尝试下一项")
                    _mark_provider_unhealthy(candidate, "text", current_model, str(e)[:120])

        candidate_num = cfg.get("candidate_num", 1) or 1
        print(f"[自动通道] 文本模型和通道均失败，已尝试: {attempted}")
        return ["Error"] * candidate_num

    if is_gateway_provider(provider_name):
        return await call_evolink_text_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=cfg,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            provider_name=provider_name,
        )

    if provider_name == "openai" or is_openai_text_model(model_name):
        return await call_openai_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": cfg["system_prompt"],
                "temperature": cfg["temperature"],
                "candidate_num": cfg["candidate_num"],
                "max_completion_tokens": cfg["max_completion_tokens"],
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider_name == "anthropic":
        return await call_claude_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": cfg["system_prompt"],
                "temperature": cfg["temperature"],
                "candidate_num": cfg["candidate_num"],
                "max_output_tokens": cfg["max_output_tokens"],
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider_name == "gemini" or is_gemini_model(model_name):
        from google.genai import types
        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=cfg["system_prompt"],
                temperature=cfg["temperature"],
                candidate_count=cfg["candidate_num"],
                max_output_tokens=cfg["max_output_tokens"],
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    raise ValueError(f"Unsupported text provider/model: provider={provider}, model={model_name}")


async def call_image_model_with_retry_async(
    provider: str,
    model_name: str,
    prompt: str,
    config: Optional[Dict[str, Any]] = None,
    contents: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = "",
    temperature: float = 1.0,
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """按 provider/model 自动路由图像生成或图像编辑。"""
    provider_name = _normalize_provider(provider)
    cfg = config or {}

    if _is_auto_provider(provider_name):
        candidates = _image_provider_candidates(model_name)
        attempted = []
        for candidate in candidates:
            if not _is_provider_available(candidate, kind="image", model_name=model_name):
                _debug_log(f"[自动通道] 跳过图像通道 {candidate}: 未配置或未初始化")
                continue
            if not _is_provider_healthy(candidate, kind="image", model_name=model_name):
                _debug_log(f"[自动通道] 跳过图像组合 {model_name}@{candidate}: 近期失败，冷却中")
                continue
            attempted.append(candidate)
            try:
                _debug_log(f"[自动通道] 图像模型 {model_name} 尝试使用 {candidate}")
                result = await call_image_model_with_retry_async(
                    provider=candidate,
                    model_name=model_name,
                    prompt=prompt,
                    config=config,
                    contents=contents,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_attempts=AUTO_PROVIDER_ATTEMPTS,
                    retry_delay=AUTO_PROVIDER_RETRY_DELAY,
                    error_context=error_context,
                )
                if not _is_bad_response(result):
                    print(f"[自动通道] 图像模型 {model_name} 使用 {candidate} 成功")
                    return result
                print(f"[自动通道] 图像组合 {model_name}@{candidate} 重试后仍失败")
                _mark_provider_unhealthy(candidate, "image", model_name, "返回 Error")
            except Exception as e:
                print(f"[自动通道] 图像组合 {model_name}@{candidate} 异常，尝试下一通道")
                _mark_provider_unhealthy(candidate, "image", model_name, str(e)[:120])

        print(f"[自动通道] 图像模型 {model_name} 所有可用通道均失败，已尝试: {attempted}")
        return ["Error"]

    if is_gateway_provider(provider_name):
        image_urls = cfg.get("image_urls")
        if not image_urls and contents:
            image_urls = await _upload_content_images_to_gateway(
                contents,
                model_name=model_name,
                provider_name=provider_name,
            )
        gateway_size = cfg.get("size") or cfg.get("aspect_ratio", "16:9")
        if is_openai_image_model(model_name):
            gateway_size = _gateway_openai_size_from_aspect_ratio(gateway_size)
        return await call_evolink_image_with_retry_async(
            model_name=model_name,
            prompt=prompt,
            config={
                "aspect_ratio": gateway_size,
                "quality": _gateway_quality_for_model(model_name, cfg),
                "image_urls": image_urls,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            provider_name=provider_name,
        )

    if provider_name == "openai" or is_openai_image_model(model_name):
        openai_quality = cfg.get("openai_quality") or cfg.get("quality", "high")
        if openai_quality not in OPENAI_IMAGE_QUALITIES:
            openai_quality = "high"
        image_config = {
            "size": cfg.get("size") or _openai_size_from_aspect_ratio(cfg.get("aspect_ratio", "16:9")),
            "quality": openai_quality,
            "background": cfg.get("background", "opaque"),
            "output_format": cfg.get("output_format", "png"),
            "input_fidelity": cfg.get("input_fidelity", "high"),
        }
        if contents and _extract_openai_image_files(contents):
            return await call_openai_image_edit_with_retry_async(
                model_name=model_name,
                prompt=prompt,
                contents=contents,
                config=image_config,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )
        return await call_openai_image_generation_with_retry_async(
            model_name=model_name,
            prompt=prompt,
            config=image_config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider_name == "gemini" or is_gemini_model(model_name):
        from google.genai import types
        gemini_contents = contents or [{"type": "text", "text": prompt}]
        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                candidate_count=1,
                max_output_tokens=cfg.get("max_output_tokens", 50000),
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=cfg.get("aspect_ratio", "16:9"),
                    image_size=cfg.get("image_size", "1k"),
                ),
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    raise ValueError(f"Unsupported image provider/model: provider={provider}, model={model_name}")
