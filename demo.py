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
PaperVizAgent 并行 Streamlit 演示
接受用户文本输入，复制 10 份，并行处理以生成多个图表候选方案供比较。
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

print("调试：正在导入代理模块...")
try:
    from agents.planner_agent import PlannerAgent
    print("调试：已导入 PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("调试：已导入所有代理模块")
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor
    print("调试：已导入工具模块")

    import yaml
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    model_config_data = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"警告：配置文件 {config_path} 解析失败，将仅使用环境变量和默认值。错误：{e}")

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

    PROVIDER_OPTIONS = ["auto", "google", "aipaibox", "evolink", "orcarouter"]

except ImportError as e:
    print(f"调试：导入错误：{e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"调试：导入过程中发生异常：{e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperVizAgent 并行演示",
    page_icon="🍌"
)

def clean_text(text):
    """清理文本，移除无效的 UTF-8 代理字符。"""
    if not text:
        return text
    if isinstance(text, str):
        # 移除导致 UnicodeEncodeError 的代理字符
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """将 base64 字符串转换为 PIL 图像。"""
    if not b64_str:
        return None
    try:
        from utils.image_utils import image_bytes_from_base64
        image_data = image_bytes_from_base64(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def image_to_jpeg_bytes(image):
    """将 PIL 图像转换为适合 API 上传的 JPEG 字节。"""
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )

    if has_alpha:
        rgba_image = image.convert("RGBA")
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()

def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3):
    """创建多份输入数据副本用于并行处理。"""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio
        },
        "max_critic_rounds": max_critic_rounds  # 添加评审轮次控制
    }

    # 创建 num_copies 份相同的输入，每份带有唯一标识符
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)

    return inputs

TEXT_MODEL_OPTIONS = [
    "gpt-5.5",
    "gpt-5.4",
    "gemini-2.5-flash",
    "orcarouter/auto",
    "自定义...",
]

IMAGE_MODEL_OPTIONS = [
    "gpt-image-2",
    "gemini-3.1-flash-image-preview",
    "自定义...",
]


def _select_index(options, value, default=0):
    return options.index(value) if value in options else default


def _model_selectbox_with_custom(label, options, default_value, key, help_text):
    if default_value in options:
        default_index = options.index(default_value)
        custom_default = ""
    else:
        default_index = options.index("自定义...") if "自定义..." in options else 0
        custom_default = default_value

    selected = st.selectbox(
        label,
        options,
        index=default_index,
        key=f"{key}_preset",
        help=help_text,
    )
    if selected != "自定义...":
        return selected

    return st.text_input(
        f"{label}（自定义）",
        value=custom_default,
        key=f"{key}_custom",
        help="输入任意兼容当前可用 API 通道的模型名称。",
    ).strip()


async def process_parallel_candidates(data_list, exp_mode="dev_planner_critic", retrieval_setting="auto", model_name="", image_model_name="", provider="auto", api_keys=None):
    """使用 PaperVizProcessor 并行处理多个候选方案。"""
    api_keys = api_keys or {}
    print(f"\n{'='*60}")
    print(f"[DEBUG] process_parallel_candidates 开始")
    print(f"[DEBUG]   provider={provider}, model={model_name}, image_model={image_model_name}")
    print(f"[DEBUG]   exp_mode={exp_mode}, retrieval={retrieval_setting}, candidates={len(data_list)}")
    available_key_names = [name for name, key in api_keys.items() if key]
    print(f"[DEBUG]   可用 key: {available_key_names or '无'}")
    print(f"{'='*60}")

    from utils import generation_utils

    # 初始化所有可用通道；实际调用时由 provider="auto" 根据模型和失败情况选择。
    aipaibox_key = api_keys.get("aipaibox", "")
    aipaibox_gemini_key = api_keys.get("aipaibox_gemini", "")
    aipaibox_openai_key = api_keys.get("aipaibox_openai", "")
    evolink_key = api_keys.get("evolink", "")
    orcarouter_key = api_keys.get("orcarouter", "")
    google_key = api_keys.get("google", "")
    openai_key = api_keys.get("openai", "")

    if any([aipaibox_key, aipaibox_gemini_key, aipaibox_openai_key]):
        generation_utils.init_aipaibox_provider(
            api_key=aipaibox_key,
            gemini_api_key=aipaibox_gemini_key,
            openai_api_key=aipaibox_openai_key,
            base_url=get_config_val("aipaibox", "base_url", "AIPAIBOX_BASE_URL", "https://api.aipaibox.com"),
        )
    elif evolink_key:
        generation_utils.init_evolink_provider(
            evolink_key,
            base_url=get_config_val("evolink", "base_url", "EVOLINK_BASE_URL", "https://api.evolink.ai"),
        )
    if orcarouter_key:
        generation_utils.init_orcarouter_provider(
            orcarouter_key,
            base_url=get_config_val("orcarouter", "base_url", "ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1"),
            wire_api=get_config_val("orcarouter", "wire_api", "ORCAROUTER_WIRE_API", "responses"),
        )

    if google_key:
        generation_utils.init_gemini_client(google_key)
    if openai_key:
        generation_utils.init_openai_client(openai_key)

    if not any([aipaibox_key, aipaibox_gemini_key, aipaibox_openai_key, evolink_key, orcarouter_key, google_key, openai_key]):
        print("[DEBUG] ⚠️ 未提供 API Key，将仅使用配置文件/环境变量中已初始化的通道")

    # 创建实验配置
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        model_name=model_name,
        image_model_name=image_model_name,
        provider=provider,
        work_dir=Path(__file__).parent,
    )
    print(f"[DEBUG] ExpConfig 已创建: provider={exp_config.provider}, model={exp_config.model_name}, image_model={exp_config.image_model_name}")

    # 初始化处理器及所有代理
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    # 并行处理所有候选方案（并发量由处理器控制）
    results = []
    concurrent_num = 3  # 控制并发量，避免触发 API 限流 (429)

    try:
        async for result_data in processor.process_queries_batch(
            data_list, max_concurrent=concurrent_num, do_eval=False
        ):
            results.append(result_data)
    finally:
        # 关闭 Gateway Provider 的共享 session，避免资源泄漏
        from utils import generation_utils
        if hasattr(generation_utils, "close_gateway_providers"):
            await generation_utils.close_gateway_providers()
        elif generation_utils.evolink_provider and hasattr(generation_utils.evolink_provider, 'close'):
            await generation_utils.evolink_provider.close()

    return results

async def refine_image_with_nanoviz(
    image_bytes,
    edit_prompt,
    aspect_ratio="21:9",
    image_size="2K",
    api_keys=None,
    provider="auto",
    image_model="gpt-image-2",
    error_context="",
):
    """
    使用自动 API 通道精修图像，支持 AIPAIBOX/Evolink、Gemini 和 OpenAI。

    参数：
        image_bytes: 图像字节数据
        edit_prompt: 描述所需修改的文本
        aspect_ratio: 输出宽高比 (21:9, 16:9, 3:2)
        image_size: 输出分辨率 (2K 或 4K)
        api_keys: 各通道 API 密钥
        provider: 优先 API 提供商
        image_model: 图像模型名称
        error_context: 日志上下文

    返回：
        元组 (编辑后的图像字节数据, 成功消息)
    """
    try:
        from utils import generation_utils
        api_keys = api_keys or {}

        aipaibox_key = api_keys.get("aipaibox", "")
        aipaibox_gemini_key = api_keys.get("aipaibox_gemini", "")
        aipaibox_openai_key = api_keys.get("aipaibox_openai", "")
        evolink_key = api_keys.get("evolink", "")
        orcarouter_key = api_keys.get("orcarouter", "")
        google_key = api_keys.get("google", "")
        openai_key = api_keys.get("openai", "")

        if any([aipaibox_key, aipaibox_gemini_key, aipaibox_openai_key]):
            generation_utils.init_aipaibox_provider(
                api_key=aipaibox_key,
                gemini_api_key=aipaibox_gemini_key,
                openai_api_key=aipaibox_openai_key,
                base_url=get_config_val("aipaibox", "base_url", "AIPAIBOX_BASE_URL", "https://api.aipaibox.com"),
            )
        elif evolink_key:
            generation_utils.init_evolink_provider(
                evolink_key,
                base_url=get_config_val("evolink", "base_url", "EVOLINK_BASE_URL", "https://api.evolink.ai"),
            )
        if orcarouter_key:
            generation_utils.init_orcarouter_provider(
                orcarouter_key,
                base_url=get_config_val("orcarouter", "base_url", "ORCAROUTER_BASE_URL", "https://api.orcarouter.ai/v1"),
                wire_api=get_config_val("orcarouter", "wire_api", "ORCAROUTER_WIRE_API", "responses"),
            )
        if google_key:
            generation_utils.init_gemini_client(google_key)
        if openai_key:
            generation_utils.init_openai_client(openai_key)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            result = await generation_utils.call_image_model_with_retry_async(
                provider=provider,
                model_name=image_model,
                prompt=edit_prompt,
                contents=[
                    {"type": "text", "text": edit_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                ],
                config={
                    "aspect_ratio": aspect_ratio,
                    "quality": image_size,
                    "gateway_quality": image_size,
                    "openai_quality": "high",
                    "output_format": "png",
                    "image_size": image_size,
                    "gemini_image_config": True,
                },
                max_attempts=3,
                retry_delay=10,
                error_context=error_context,
            )

            if result and result[0] and result[0] != "Error":
                from utils.image_utils import image_bytes_from_base64
                return image_bytes_from_base64(result[0]), "✅ 图像精修成功！"

            return None, "❌ 图像精修失败，所有可用通道均未返回有效图像数据"
        finally:
            if hasattr(generation_utils, "close_gateway_providers"):
                await generation_utils.close_gateway_providers()
            elif generation_utils.evolink_provider and hasattr(generation_utils.evolink_provider, 'close'):
                await generation_utils.evolink_provider.close()
            # 延迟一下给 asyncio 机会清理抛出的 SSL 没接住的 exception
            await asyncio.sleep(0.1)

    except Exception as e:
        return None, f"❌ 错误：{str(e)}"


def get_evolution_stages(result, exp_mode):
    """从结果中提取所有演化阶段（图像和描述）。"""
    task_name = "diagram"
    stages = []

    # 阶段 1：规划器输出
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 规划器",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "基于方法内容生成的初始图表规划"
        })

    # 阶段 2：风格化器输出（仅限 demo_full 模式）
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ 风格化器",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "经过风格优化的描述"
            })

    # 阶段 3+：评审迭代
    for round_idx in range(4):  # 检查最多 4 轮
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"

        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 评审第 {round_idx} 轮",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"根据评审反馈进行优化（第 {round_idx} 次迭代）"
            })

    return stages

def display_candidate_result(result, candidate_id, exp_mode):
    """展示单个候选方案的结果。"""
    task_name = "diagram"

    # 根据 exp_mode 决定展示哪张图像
    # 对于演示模式，始终尝试查找最后一轮评审结果
    final_image_key = None
    final_desc_key = None

    # 尝试查找最后一轮评审
    for round_idx in range(3, -1, -1):  # 检查第 3、2、1、0 轮
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break

    # 如果没有完成评审轮次则使用备选方案
    if not final_image_key:
        if exp_mode == "demo_full":
            # demo_full 在可视化之前使用风格化器
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            # demo_planner_critic 使用规划器输出
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"

    # 展示最终图像
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, width="stretch", caption=f"候选方案 {candidate_id}（最终版）")

            # 添加下载按钮
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ 下载",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                width="stretch"
            )
        else:
            st.error(f"候选方案 {candidate_id} 的图像解码失败")
    else:
        st.warning(f"候选方案 {candidate_id} 未生成图像")

    # 在折叠面板中展示演化时间线
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 查看演化时间线（{len(stages)} 个阶段）", expanded=False):
            st.caption("查看图表在不同流水线阶段的演化过程")

            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])

                # 展示该阶段的图像
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, width="stretch")

                # 展示描述
                if stage['desc_key'] in result:
                    with st.expander(f"📝 描述", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)

                # 展示评审建议（如有）
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 评审建议", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ 无需修改——迭代已停止。")
                        else:
                            st.write(cleaned_sugg)

                # 在阶段之间添加分隔线（最后一个除外）
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # 如果只有一个阶段，使用更简洁的折叠面板展示描述
        with st.expander(f"📝 查看描述", expanded=False):
            if final_desc_key and final_desc_key in result:
                # 清理文本，移除无效的 UTF-8 字符
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("暂无描述")

def main():
    st.title("🍌 PaperVizAgent 演示")
    st.markdown("AI 驱动的科学图表生成与精修")

    # 创建选项卡
    tab1, tab2 = st.tabs(["📊 生成候选方案", "✨ 精修图像"])

    # ==================== 选项卡 1：生成候选方案 ====================
    with tab1:
        st.markdown("### 从您的方法章节和图注生成多个图表候选方案")

        # 侧边栏配置（选项卡 1）
        with st.sidebar:
            st.title("⚙️ 生成设置")

            exp_mode = st.selectbox(
                "流水线模式",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="选择使用哪种代理流水线"
            )

            mode_info = {
                "demo_planner_critic": "规划器 → 可视化器 → 评审器 → 可视化器",
                "demo_full": "检索器 → 规划器 → 风格化器 → 可视化器 → 评审器 → 可视化器。（风格化器能让图表更具美感，但可能过度简化。建议两种模式都尝试并选择最佳结果）"
            }
            st.info(f"**流水线：** {mode_info[exp_mode]}")

            retrieval_setting = st.selectbox(
                "检索设置",
                ["auto", "auto-full", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="如何检索参考图表",
                format_func=lambda x: {
                    "auto": "auto — LLM 智能选参考，仅 caption（~3万 tokens/候选）",
                    "auto-full": "auto-full — LLM 智能选参考，含完整论文（⚠️ ~80万 tokens/候选）",
                    "random": "random — 随机选 10 个参考（免费）",
                    "none": "none — 不检索参考（免费）",
                }[x],
            )

            _retrieval_cost_info = {
                "auto": "💡 轻量 auto：仅发送图注（caption）给 LLM 做匹配，每个候选约 **3 万 tokens**，性价比最高。",
                "auto-full": "⚠️ **注意**：完整 auto 将 200 篇参考论文的全文发给 LLM，每个候选消耗约 **80 万 tokens**。仅在需要高精度检索时使用。",
                "random": "✅ 随机从 298 篇参考中选 10 个，不调用 API，零费用。",
                "none": "✅ 跳过检索，不使用参考图表，零费用。",
            }
            st.info(_retrieval_cost_info[retrieval_setting])

            num_candidates = st.number_input(
                "候选方案数量",
                min_value=1,
                max_value=20,
                value=5,
                key="tab1_num_candidates",
                help="要并行生成多少个候选方案"
            )

            aspect_ratio = st.selectbox(
                "宽高比",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="生成图表的宽高比"
            )

            max_critic_rounds = st.number_input(
                "最大评审轮次",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="评审优化迭代的最大轮次"
            )

            default_text_model = get_config_val("defaults", "model_name", "PAPERBANANA_MODEL_NAME", "gpt-5.5")
            default_image_model = get_config_val("defaults", "image_model_name", "PAPERBANANA_IMAGE_MODEL_NAME", "gpt-image-2")

            model_name = _model_selectbox_with_custom(
                "文本模型",
                TEXT_MODEL_OPTIONS,
                key="tab1_model_name",
                default_value=default_text_model,
                help_text="用于推理/规划/评审。系统会根据模型和可用 key 自动选择 AIPAIBOX、Google 或 OpenAI 通道，并在失败时切换。"
            )

            image_model_name = _model_selectbox_with_custom(
                "候选生成图像模型",
                IMAGE_MODEL_OPTIONS,
                key="tab1_image_model_name",
                default_value=default_image_model,
                help_text="用于“生成候选方案”标签页的图像生成。系统会自动选择可用通道。"
            )
            st.caption(f"本次候选生成将使用图像模型：`{image_model_name}`")

            st.caption("API 通道自动选择；发生超时、返回 Error 或初始化失败时会尝试下一个可用通道。")

            with st.expander("API Keys", expanded=False):
                aipaibox_legacy_api_key = get_config_val("aipaibox", "api_key", "AIPAIBOX_API_KEY", "")
                aipaibox_gemini_api_key = st.text_input(
                    "AIPAIBOX Gemini API Key",
                    type="password",
                    key="tab1_aipaibox_gemini_api_key",
                    value=get_config_val("aipaibox", "gemini_api_key", "AIPAIBOX_GEMINI_API_KEY", aipaibox_legacy_api_key),
                    help="用于 https://api.aipaibox.com 调用 Gemini 系列模型"
                )
                aipaibox_openai_api_key = st.text_input(
                    "AIPAIBOX OpenAI API Key",
                    type="password",
                    key="tab1_aipaibox_openai_api_key",
                    value=get_config_val("aipaibox", "openai_api_key", "AIPAIBOX_OPENAI_API_KEY", aipaibox_legacy_api_key),
                    help="用于 https://api.aipaibox.com 调用 GPT/OpenAI 系列模型和 gpt-image-2"
                )
                google_api_key = st.text_input(
                    "Google API Key",
                    type="password",
                    key="tab1_google_api_key",
                    value=get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", ""),
                    help="用于 Google Gemini 官方 endpoint"
                )
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    key="tab1_openai_api_key",
                    value=get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", ""),
                    help="用于 OpenAI 文本模型和 gpt-image-2"
                )
                evolink_api_key = st.text_input(
                    "Evolink API Key",
                    type="password",
                    key="tab1_evolink_api_key",
                    value=get_config_val("evolink", "api_key", "EVOLINK_API_KEY", ""),
                    help="兼容旧配置；未填写 AIPAIBOX key 时才会作为网关通道使用"
                )
                orcarouter_api_key = st.text_input(
                    "OrcaRouter API Key",
                    type="password",
                    key="tab1_orcarouter_api_key",
                    value=get_config_val("orcarouter", "api_key", "ORCA_KEY", ""),
                    help="用于 https://api.orcarouter.ai/v1 的 Responses API（模型示例：orcarouter/auto）"
                )

            provider = st.selectbox(
                "优先 API 提供商",
                PROVIDER_OPTIONS,
                index=0,
                key="tab1_provider",
                help="选择优先使用的通道（auto: 自动选择/备用, google: 官方Gemini, aipaibox: 代理网关）",
                format_func=lambda x: {
                    "auto": "Auto (自动回退)",
                    "google": "Google (官方 Gemini)",
                    "aipaibox": "AIPAIBOX (代理网关)",
                    "evolink": "Evolink (兼容网关)",
                    "orcarouter": "OrcaRouter (Responses 网关)",
                }[x]
            )

            api_keys = {
                "aipaibox": aipaibox_legacy_api_key,
                "aipaibox_gemini": aipaibox_gemini_api_key,
                "aipaibox_openai": aipaibox_openai_api_key,
                "google": google_api_key,
                "openai": openai_api_key,
                "evolink": evolink_api_key,
                "orcarouter": orcarouter_api_key,
            }

        st.divider()

        # 输入区域
        st.markdown("## 📝 输入")

        # 示例内容
        example_method = r"""## Methodology: The PaperVizAgent Framework

        In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive "academic style," as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."

        col_input1, col_input2 = st.columns([3, 2])

        with col_input1:
            # 方法内容示例选择器
            method_example = st.selectbox(
                "加载示例（方法章节）",
                ["无", "PaperVizAgent 框架"],
                key="method_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if method_example == "PaperVizAgent 框架":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")

            method_content = st.text_area(
                "方法章节内容（建议使用 Markdown 格式）",
                value=method_value,
                height=250,
                placeholder="在此粘贴方法章节内容...",
                help="论文中描述方法的章节内容。建议使用 Markdown 格式。"
            )

        with col_input2:
            # 图注示例选择器
            caption_example = st.selectbox(
                "加载示例（图注）",
                ["无", "PaperVizAgent 框架"],
                key="caption_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if caption_example == "PaperVizAgent 框架":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")

            caption = st.text_area(
                "图注（建议使用 Markdown 格式）",
                value=caption_value,
                height=250,
                placeholder="输入图注...",
                help="要生成的图表的标题或描述。建议使用 Markdown 格式。"
            )

        # 处理按钮
        if st.button("🚀 生成候选方案", type="primary", width="stretch"):
            if not method_content or not caption:
                st.error("请同时提供方法内容和图注！")
            else:
                print(
                    f"[Demo UI] 生成候选方案：provider={provider}, "
                    f"text_model={model_name}, image_model={image_model_name}"
                )
                st.info(f"本次候选生成使用图像模型：`{image_model_name}`")
                # 保存到会话状态
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption

                with st.spinner(f"正在并行生成 {num_candidates} 个候选方案... 这可能需要几分钟。"):
                    # 创建输入数据列表
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds
                    )

                    # 并行处理
                    try:
                        results = run_async(process_parallel_candidates(
                            input_data_list,
                            exp_mode=exp_mode,
                            retrieval_setting=retrieval_setting,
                            model_name=model_name,
                            image_model_name=image_model_name,
                            provider=provider,
                            api_keys=api_keys
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str

                        # 将结果保存为 JSON 文件
                        try:
                            # 如果结果目录不存在则创建
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)

                            # 生成带时间戳的文件名
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                            # 保存为 JSON 并正确处理编码（与 main.py 一致）
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # 清理无效的 UTF-8 字符
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)

                            st.session_state["json_file"] = str(json_filename)
                            st.success(f"✅ 成功生成 {len(results)} 个候选方案！")
                            st.info(f"💾 结果已保存至：`{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"⚠️ 已生成 {len(results)} 个候选方案，但 JSON 保存失败：{e}")
                    except Exception as e:
                        st.error(f"处理过程中出错：{e}")
                        import traceback
                        st.code(traceback.format_exc())

        # 展示结果
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")

            st.divider()
            st.markdown("## 🎨 已生成的候选方案")
            st.caption(f"生成时间：{timestamp} | 流水线：{mode_info.get(current_mode, current_mode)}")

            # 如果有 JSON 文件则显示下载按钮
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 结果已保存至：`{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ 下载 JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            width="stretch"
                        )

            # 以网格形式展示结果（3 列）
            num_cols = 3
            num_results = len(results)

            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)

            # 添加 ZIP 下载按钮
            st.divider()
            st.markdown("### 💾 批量下载")

            try:
                import zipfile

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"

                    for candidate_id, result in enumerate(results):

                        # 查找最终图像键（逻辑与展示一致）
                        final_image_key = None

                        # 尝试查找最后一轮评审
                        for round_idx in range(3, -1, -1):
                            image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                            if image_key in result and result[image_key]:
                                final_image_key = image_key
                                break

                        # 如果没有完成评审轮次则使用备选方案
                        if not final_image_key:
                            if current_mode == "demo_full":
                                final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                            else:
                                final_image_key = f"target_{task_name}_desc0_base64_jpg"

                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )

                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ 下载 ZIP 压缩包",
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width="stretch"
                )
                st.success("ZIP 压缩包已准备好，可以下载！")
            except Exception as e:
                st.error(f"创建 ZIP 压缩包失败：{e}")

    # ==================== 选项卡 2：精修图像 ====================
    with tab2:
        st.markdown("### 精修并放大您的图表至高分辨率（2K/4K）")
        st.caption("上传候选方案中的图像或任意图表，描述修改需求，生成高分辨率版本")

        # 精修设置侧边栏
        with st.sidebar:
            st.title("✨ 精修设置")

            refine_provider = st.selectbox(
                "优先 API 提供商",
                PROVIDER_OPTIONS,
                index=0,
                key="tab2_provider",
                help="自动：按配置文件选择；Google：官方Gemini；AIPAIBOX：网关直连",
                format_func=lambda x: {
                    "auto": "Auto (自动回退)",
                    "google": "Google (官方 Gemini)",
                    "aipaibox": "AIPAIBOX (代理网关)",
                    "evolink": "Evolink (兼容网关)",
                    "orcarouter": "OrcaRouter (Responses 网关)",
                }[x]
            )

            refine_resolution = st.selectbox(
                "目标分辨率",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="更高的分辨率需要更长时间但能产生更好的质量"
            )

            refine_aspect_ratio = st.selectbox(
                "宽高比",
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help="精修图像的宽高比"
            )

            refine_image_model_name = _model_selectbox_with_custom(
                "精修图像模型",
                IMAGE_MODEL_OPTIONS,
                key="tab2_image_model_name",
                default_value=get_config_val("defaults", "image_model_name", "PAPERBANANA_IMAGE_MODEL_NAME", "gpt-image-2"),
                help_text="用于“精修图像”标签页。系统会自动选择可用通道。"
            )
            st.caption(f"本次精修将使用图像模型：`{refine_image_model_name}`")

        st.divider()

        # 上传区域
        st.markdown("## 📤 上传图像")
        uploaded_file = st.file_uploader(
            "选择一个图像文件",
            type=["png", "jpg", "jpeg"],
            help="上传您想要精修的图表"
        )

        if uploaded_file is not None:
            # 展示上传的图像
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 原始图像")
                st.image(uploaded_image, width="stretch")

            with col2:
                st.markdown("### 编辑指令")
                edit_prompt = st.text_area(
                    "描述您想要的修改",
                    height=200,
                    placeholder="例如：'将配色方案改为学术论文风格' 或 '将文字放大加粗' 或 '保持内容不变但输出更高分辨率'",
                    help="描述您想要的修改，或使用'保持内容不变'仅进行放大",
                    key="edit_prompt"
                )

                if st.button("✨ 精修图像", type="primary", width="stretch"):
                    if not edit_prompt:
                        st.error("请提供编辑指令！")
                    else:
                        with st.spinner(f"正在将图像精修至 {refine_resolution} 分辨率... 这可能需要一分钟。"):
                            try:
                                # 将 PIL 图像转换为字节
                                image_bytes = image_to_jpeg_bytes(uploaded_image)

                                # 获取当前选择的图像模型
                                current_image_model = refine_image_model_name
                                print(
                                    f"[Demo UI] 精修图像：provider={refine_provider}, "
                                    f"image_model={current_image_model}"
                                )
                                st.info(f"本次精修使用图像模型：`{current_image_model}`")

                                # 调用精修 API
                                refined_bytes, message = run_async(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution,
                                        api_keys=api_keys,
                                        provider=refine_provider,
                                        image_model=current_image_model,
                                        error_context=f"refine-{current_image_model}-{datetime.now().strftime('%H%M%S')}",
                                    )
                                )

                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"精修过程中出错：{e}")
                                import traceback
                                st.code(traceback.format_exc())

            # 展示精修结果（如有）
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 精修结果")
                st.caption(f"生成时间：{st.session_state.get('refine_timestamp', 'N/A')} | 分辨率：{refine_resolution}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 精修前")
                    st.image(uploaded_image, width="stretch")

                with col2:
                    st.markdown(f"### 精修后（{refine_resolution}）")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, width="stretch")

                    # 下载按钮
                    st.download_button(
                        label=f"⬇️ 下载 {refine_resolution} 图像",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        width="stretch"
                    )

if __name__ == "__main__":
    main()
