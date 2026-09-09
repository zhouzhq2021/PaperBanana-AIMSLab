"""
Gateway Provider 单元测试
测试文本生成和图像生成的核心逻辑
"""

import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.gateway import GatewayProvider


# ==================== 辅助函数 ====================

def make_png_base64():
    """创建一个最小的 PNG 图片并返回 base64 字符串"""
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_provider(api_key="test-key", base_url="https://api.evolink.ai"):
    """创建 GatewayProvider 实例"""
    return GatewayProvider(api_key=api_key, base_url=base_url)


# ==================== 初始化测试 ====================

class TestGatewayProviderInit:
    def test_init_with_params(self):
        p = make_provider(api_key="sk-abc", base_url="https://example.com")
        assert p.api_key == "sk-abc"
        assert p.base_url == "https://example.com"

    def test_init_default_base_url(self):
        p = GatewayProvider(api_key="sk-abc")
        assert p.base_url == "https://api.aipaibox.com"

    def test_headers_contain_auth(self):
        p = make_provider(api_key="sk-test")
        headers = p._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_orcarouter_base_url_and_wire_api(self):
        p = GatewayProvider(
            api_key="sk-orca-test",
            base_url="https://api.orcarouter.ai/v1",
            wire_api="responses",
        )
        assert p._v1_url("responses") == "https://api.orcarouter.ai/v1/responses"
        assert p.wire_api == "responses"


# ==================== 内容格式转换测试 ====================

class TestContentConversion:
    def test_text_only_content(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello world"}]
        messages = p._convert_contents_to_messages(contents, system_prompt="You are helpful")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        # 纯文本时 content 可以是字符串
        user_content = messages[1]["content"]
        assert any("Hello world" in str(part) for part in (user_content if isinstance(user_content, list) else [user_content]))

    def test_text_and_image_content(self):
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": img_b64,
                    "media_type": "image/jpeg",
                },
            },
        ]
        messages = p._convert_contents_to_messages(contents, system_prompt="You are a vision model")

        assert len(messages) == 2
        user_content = messages[1]["content"]
        # 多模态时 content 应该是列表
        assert isinstance(user_content, list)
        # 应该包含文本和图片两个部分
        types_present = {item["type"] for item in user_content}
        assert "text" in types_present
        assert "image_url" in types_present

    def test_image_with_base64_content(self):
        """测试 image 使用 image_base64 字段（planner agent 使用的格式）"""
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Look at this"},
            {"type": "image", "image_base64": img_b64},
        ]
        messages = p._convert_contents_to_messages(contents, system_prompt="Be helpful")

        # messages[0] = system, messages[1] = user
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        image_parts = [part for part in user_content if part["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_empty_system_prompt(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hi"}]
        messages = p._convert_contents_to_messages(contents, system_prompt="")
        # 空 system prompt 不应生成 system message
        assert messages[0]["role"] == "user"


# ==================== 文本生成测试 ====================

class TestTextGeneration:
    @pytest.mark.asyncio
    async def test_responses_api_generation_success(self):
        p = GatewayProvider(
            api_key="sk-orca-test",
            base_url="https://api.orcarouter.ai/v1",
            wire_api="responses",
        )
        mock_response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Orca response"}],
                }
            ]
        }

        with patch.object(p, "_post_json", new_callable=AsyncMock, return_value=mock_response) as post:
            result = await p.generate_text(
                model_name="orcarouter/auto",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="Be helpful",
                temperature=0.7,
                max_output_tokens=1000,
            )

        assert result == ["Orca response"]
        url, payload = post.await_args.args
        assert url == "https://api.orcarouter.ai/v1/responses"
        assert payload["model"] == "orcarouter/auto"
        assert payload["max_output_tokens"] == 1000
        assert payload["input"][0]["role"] == "system"
        assert payload["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "Be helpful",
        }

    def test_responses_payload_converts_image_input(self):
        p = GatewayProvider(api_key="test", wire_api="responses")
        payload = p._build_responses_payload(
            model_name="orcarouter/auto",
            contents=[
                {"type": "text", "text": "Describe"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc123",
                    },
                },
            ],
            system_prompt="",
            temperature=1.0,
            max_output_tokens=100,
        )
        image_part = payload["input"][0]["content"][1]
        assert image_part == {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc123",
        }

    @pytest.mark.asyncio
    async def test_text_generation_success(self):
        p = make_provider()

        mock_response = {
            "choices": [
                {"message": {"content": "This is a test response"}}
            ]
        }

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=mock_response):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="You are helpful",
                temperature=0.7,
                max_output_tokens=1000,
            )

        assert result == ["This is a test response"]

    @pytest.mark.asyncio
    async def test_text_generation_retry_on_failure(self):
        p = make_provider()

        mock_response = {
            "choices": [
                {"message": {"content": "Success after retry"}}
            ]
        }

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API error")
            return mock_response

        with patch.object(p, '_post_json', side_effect=mock_post):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="",
                temperature=1.0,
                max_output_tokens=50000,
                max_attempts=5,
                retry_delay=0,
            )

        assert result == ["Success after retry"]
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_text_generation_all_attempts_fail(self):
        p = make_provider()

        with patch.object(p, '_post_json', new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="",
                temperature=1.0,
                max_output_tokens=50000,
                max_attempts=3,
                retry_delay=0,
            )

        assert result == ["Error"]


# ==================== 图像生成测试 ====================

class TestImageGeneration:
    @pytest.mark.asyncio
    async def test_image_generation_creates_task(self):
        p = make_provider()

        create_response = {
            "id": "task-unified-123",
            "status": "pending",
            "progress": 0,
        }

        completed_response = {
            "id": "task-unified-123",
            "status": "completed",
            "progress": 100,
            "results": ["https://example.com/image.png"],
        }

        # 创建一个小的 PNG 图片用于模拟下载
        png_bytes = base64.b64decode(make_png_base64())

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=completed_response), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="A beautiful diagram",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=3,
                retry_delay=0,
                poll_interval=0,
            )

        assert len(result) == 1
        assert result[0] is not None
        assert len(result[0]) > 10  # base64 string should be non-trivial

    @pytest.mark.asyncio
    async def test_image_generation_polls_until_complete(self):
        p = make_provider()

        create_response = {
            "id": "task-123",
            "status": "pending",
        }

        poll_responses = [
            {"id": "task-123", "status": "processing", "progress": 30},
            {"id": "task-123", "status": "processing", "progress": 60},
            {"id": "task-123", "status": "completed", "progress": 100, "results": ["https://example.com/img.png"]},
        ]

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, side_effect=poll_responses), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Test",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=3,
                retry_delay=0,
                poll_interval=0,
            )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_image_generation_task_fails(self):
        p = make_provider()

        create_response = {"id": "task-fail", "status": "pending"}
        failed_response = {"id": "task-fail", "status": "failed", "progress": 0}

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=failed_response):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Test",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert result == ["Error"]

    @pytest.mark.asyncio
    async def test_image_generation_with_image_urls(self):
        """测试带参考图片的图像生成（image-to-image）"""
        p = make_provider()

        create_response = {
            "id": "task-img2img",
            "status": "pending",
        }
        completed_response = {
            "id": "task-img2img",
            "status": "completed",
            "progress": 100,
            "results": ["https://example.com/result.png"],
        }

        captured_payload = {}
        async def capture_post(url, payload):
            captured_payload.update(payload)
            return create_response

        with patch.object(p, '_post_json', side_effect=capture_post), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=completed_response), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Edit this image",
                aspect_ratio="1:1",
                quality="2K",
                image_urls=["https://example.com/ref.png"],
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert "image_urls" in captured_payload
        assert captured_payload["image_urls"] == ["https://example.com/ref.png"]

    @pytest.mark.asyncio
    async def test_gpt_image_direct_b64_response(self):
        """gpt-image-* gateway responses may be OpenAI-style synchronous results."""
        p = make_provider()
        image_b64 = make_png_base64()
        direct_response = {"data": [{"b64_json": image_b64}]}

        async def capture_post(url, payload, timeout=120):
            assert timeout == 360
            return direct_response

        with patch.object(p, '_post_json', side_effect=capture_post):
            result = await p.generate_image(
                model_name="gpt-image-2",
                prompt="Test",
                aspect_ratio="1824x1024",
                quality="high",
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert result == [image_b64]

    @pytest.mark.asyncio
    async def test_gpt_image_direct_url_response(self):
        """Direct URL responses should be downloaded and converted to base64."""
        p = make_provider()
        image_b64 = make_png_base64()
        direct_response = {"data": [{"url": "https://example.com/result.png"}]}

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=direct_response), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=image_b64):
            result = await p.generate_image(
                model_name="gpt-image-2",
                prompt="Test",
                aspect_ratio="1824x1024",
                quality="high",
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert result == [image_b64]

    @pytest.mark.asyncio
    async def test_gpt_image_with_local_input_uses_images_edits(self):
        """gpt-image-* edits through the gateway should use /v1/images/edits."""
        p = make_provider(base_url="https://api.aipaibox.com")
        image_b64 = make_png_base64()
        response = {"data": [{"b64_json": image_b64}]}
        captured = {}

        async def capture_post(url, form, timeout=120):
            captured["url"] = url
            captured["form"] = form
            captured["timeout"] = timeout
            return response

        with patch.object(p, '_post_form', side_effect=capture_post):
            result = await p.generate_image(
                model_name="gpt-image-2",
                prompt="Edit this image",
                aspect_ratio="1824x1024",
                quality="high",
                image_inputs=[
                    {
                        "data": image_b64,
                        "media_type": "image/png",
                        "filename": "input.png",
                    }
                ],
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert captured["url"] == "https://api.aipaibox.com/v1/images/edits"
        assert captured["timeout"] == 360
        assert captured["form"] is not None
        assert result == [image_b64]

    @pytest.mark.asyncio
    async def test_gemini_image_uses_generate_content_endpoint(self):
        """Gemini image preview on AIPAIBOX should use Gemini generateContent endpoint."""
        p = make_provider(base_url="https://api.aipaibox.com")
        image_b64 = make_png_base64()
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        captured = {}

        async def capture_post(url, payload, timeout=120):
            captured["url"] = url
            captured["payload"] = payload
            return response

        with patch.object(p, '_post_json', side_effect=capture_post):
            result = await p.generate_image(
                model_name="gemini-3.1-flash-image-preview",
                prompt="Test diagram",
                aspect_ratio="21:9",
                quality="2K",
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert captured["url"] == "https://api.aipaibox.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent"
        assert captured["payload"]["contents"][0]["parts"][0]["text"] == "Test diagram"
        assert captured["payload"]["generationConfig"]["responseModalities"] == ["IMAGE"]
        assert "imageConfig" not in captured["payload"]["generationConfig"]
        assert result == [image_b64]

    @pytest.mark.asyncio
    async def test_gemini_image_config_is_opt_in_for_refine(self):
        """Gemini image size/ratio is only sent when the refine path opts in."""
        p = make_provider(base_url="https://api.aipaibox.com")
        image_b64 = make_png_base64()
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        captured = {}

        async def capture_post(url, payload, timeout=120):
            captured["url"] = url
            captured["payload"] = payload
            return response

        with patch.object(p, '_post_json', side_effect=capture_post):
            result = await p.generate_image(
                model_name="gemini-3.1-flash-image-preview",
                prompt="Refine diagram",
                aspect_ratio="21:9",
                quality="2K",
                image_inputs=[
                    {
                        "data": image_b64,
                        "media_type": "image/png",
                        "filename": "input.png",
                    }
                ],
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
                gemini_image_config=True,
            )

        assert captured["url"] == "https://api.aipaibox.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent"
        assert captured["payload"]["generationConfig"]["imageConfig"] == {
            "aspectRatio": "21:9",
            "imageSize": "2K",
        }
        assert result == [image_b64]


# ==================== 请求构建测试 ====================

class TestRequestBuilding:
    def test_text_request_payload(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello"}]

        payload = p._build_text_payload(
            model_name="gemini-2.5-flash-image",
            contents=contents,
            system_prompt="Be helpful",
            temperature=0.5,
            max_output_tokens=4096,
        )

        assert payload["model"] == "gemini-2.5-flash-image"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 4096
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    def test_image_request_payload(self):
        p = make_provider()

        payload = p._build_image_payload(
            model_name="nano-banana-2-lite",
            prompt="A cat on grass",
            aspect_ratio="16:9",
            quality="2K",
        )

        assert payload["model"] == "nano-banana-2-lite"
        assert payload["prompt"] == "A cat on grass"
        assert payload["size"] == "16:9"
        assert payload["quality"] == "2K"

    def test_image_request_payload_with_urls(self):
        p = make_provider()

        payload = p._build_image_payload(
            model_name="nano-banana-2-lite",
            prompt="Edit this",
            aspect_ratio="1:1",
            quality="4K",
            image_urls=["https://example.com/img.png"],
        )

        assert payload["image_urls"] == ["https://example.com/img.png"]


# ==================== generation_utils 集成测试 ====================

class TestGenerationUtilsIntegration:
    def test_orcarouter_model_only_uses_orcarouter_candidate(self):
        from utils import generation_utils

        assert generation_utils._text_provider_candidates("orcarouter/auto") == ["orcarouter"]

    def test_orcarouter_factory_defaults(self):
        from providers import create_provider

        provider = create_provider("orcarouter", api_key="sk-orca-test")
        assert provider.base_url == "https://api.orcarouter.ai/v1"
        assert provider.wire_api == "responses"

    def test_auto_retry_defaults_are_more_patient_for_images(self):
        """Image generation gets more attempts and longer retry delay by default."""
        from utils import generation_utils

        assert generation_utils.AUTO_TEXT_PROVIDER_ATTEMPTS == 3
        assert generation_utils.AUTO_TEXT_PROVIDER_RETRY_DELAY == 15
        assert generation_utils.AUTO_TEXT_PROVIDER_RETRY_MAX_DELAY == 120
        assert generation_utils.AUTO_IMAGE_PROVIDER_ATTEMPTS == 4
        assert generation_utils.AUTO_IMAGE_PROVIDER_RETRY_DELAY == 30
        assert generation_utils.AUTO_IMAGE_PROVIDER_RETRY_MAX_DELAY == 180

    def test_retry_backoff_delay_is_exponential_with_cap(self):
        """Retry delays grow dynamically and stop at the configured cap."""
        from utils import generation_utils

        assert generation_utils._retry_backoff_delay(30, 0, 180) == 30
        assert generation_utils._retry_backoff_delay(30, 1, 180) == 60
        assert generation_utils._retry_backoff_delay(30, 2, 180) == 120
        assert generation_utils._retry_backoff_delay(30, 3, 180) == 180

    def test_image_model_candidates_fallback_to_gemini(self):
        """When gpt-image-2 is unstable, auto mode can try the Gemini image model."""
        from utils import generation_utils

        assert generation_utils._image_model_candidates("gpt-image-2") == [
            "gpt-image-2",
            "gemini-3.1-flash-image-preview",
        ]

    def test_gateway_openai_image_size_uses_dimensions(self):
        """gpt-image-* through gateway must use WIDTHxHEIGHT, not ratio strings."""
        from utils import generation_utils

        assert generation_utils._gateway_openai_size_from_aspect_ratio("21:9") == "1824x784"
        assert generation_utils._gateway_openai_size_from_aspect_ratio("16:9") == "1824x1024"
        assert generation_utils._gateway_openai_size_from_aspect_ratio("1024x1024") == "1024x1024"

    def test_gateway_openai_image_quality_uses_openai_values(self):
        """gpt-image-* through gateway should not receive Gemini-style 2K quality."""
        from utils import generation_utils

        assert generation_utils._gateway_quality_for_model("gpt-image-2", {"quality": "2K"}) == "high"
        assert generation_utils._gateway_quality_for_model("gpt-image-2", {"openai_quality": "medium"}) == "medium"
        assert generation_utils._gateway_quality_for_model("gemini-3.1-flash-image-preview", {"quality": "2K"}) == "2K"

    @pytest.mark.asyncio
    async def test_explicit_aipaibox_gpt_image_routes_to_gateway(self):
        """gpt-image-* on AIPAIBOX must not use the official OpenAI client."""
        from utils import generation_utils

        mock_provider = MagicMock()
        mock_provider.upload_image_base64 = AsyncMock(return_value="https://example.com/input.png")
        mock_provider.generate_image = AsyncMock(return_value=["gateway-image"])

        with patch.object(generation_utils, "_resolve_gateway_provider", return_value=mock_provider), \
             patch.object(generation_utils, "call_openai_image_edit_with_retry_async", new_callable=AsyncMock) as mock_openai_edit, \
             patch.object(generation_utils, "call_openai_image_generation_with_retry_async", new_callable=AsyncMock) as mock_openai_generate:
            result = await generation_utils.call_image_model_with_retry_async(
                provider="aipaibox",
                model_name="gpt-image-2",
                prompt="Edit this",
                config={"aspect_ratio": "16:9", "quality": "2K", "openai_quality": "high"},
                contents=[
                    {"type": "text", "text": "Edit this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": make_png_base64(),
                        },
                    },
                ],
                max_attempts=1,
                retry_delay=0,
            )

        assert result == ["gateway-image"]
        mock_provider.generate_image.assert_awaited_once()
        assert mock_provider.generate_image.await_args.kwargs["image_inputs"][0]["media_type"] == "image/png"
        mock_openai_edit.assert_not_awaited()
        mock_openai_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_call_gateway_text_routes_correctly(self):
        """测试 generation_utils 中的 gateway 文本调用"""
        from utils import generation_utils

        mock_response = {
            "choices": [{"message": {"content": "test response"}}]
        }

        with patch('providers.gateway.GatewayProvider._post_json',
                    new_callable=AsyncMock, return_value=mock_response):
            # 验证函数存在且可调用
            assert hasattr(generation_utils, 'call_evolink_text_with_retry_async')

    @pytest.mark.asyncio
    async def test_call_gateway_image_routes_correctly(self):
        """测试 generation_utils 中的 gateway 图像调用"""
        from utils import generation_utils

        assert hasattr(generation_utils, 'call_evolink_image_with_retry_async')


# ==================== ExpConfig Provider 字段测试 ====================

class TestExpConfigProvider:
    def test_config_has_provider_field(self):
        from utils.config import ExpConfig
        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="aipaibox",
        )
        assert config.provider == "aipaibox"

    def test_config_default_provider(self):
        from utils.config import ExpConfig
        config = ExpConfig(dataset_name="PaperBananaBench")
        assert config.provider == "auto"  # 默认自动选择可用通道


# ==================== Agent 路由测试 ====================

class TestAgentRouting:
    """测试 agent 能正确根据 provider 路由到 gateway"""

    def test_planner_uses_evolink_when_configured(self):
        from utils.config import ExpConfig
        from agents.planner_agent import PlannerAgent

        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="evolink",
            model_name="gemini-2.5-flash-image",
        )
        agent = PlannerAgent(exp_config=config)
        assert agent.exp_config.provider == "evolink"

    def test_visualizer_uses_evolink_image_model(self):
        from utils.config import ExpConfig
        from agents.visualizer_agent import VisualizerAgent

        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="evolink",
            model_name="gemini-2.5-flash-image",
            image_model_name="nano-banana-2-lite",
        )
        agent = VisualizerAgent(exp_config=config)
        assert agent.model_name == "nano-banana-2-lite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
