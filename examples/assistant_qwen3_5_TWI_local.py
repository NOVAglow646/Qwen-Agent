# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Qwen3.5 Think-with-Images demo (local GPU deployment)."""

import os

from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import multimodal_typewriter_print


def init_agent_service():
    llm_cfg = {
        # Use your own model service compatible with OpenAI API by vLLM/SGLang:
        'model': '/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3.5-397B-A17B',
        'model_type': 'qwenvl_oai',
        'model_server': os.getenv('LOCAL_OAI_BASE_URL', 'http://localhost:8000/v1'),
        'api_key': os.getenv('LOCAL_OAI_API_KEY', 'EMPTY'),
        'generate_cfg': {
            'use_raw_api': True,
            'extra_body': {
                'enable_thinking': True
            },
        },
    }

    tools = [
        'image_zoom_in_tool',
        'image_search',
        'web_search',
    ]

    system_message = (
        'You are an image reasoning assistant. Think step-by-step with visual evidence. '
        'If needed, call tools to zoom into image regions or search for supporting information before answering.'
    )

    bot = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='Qwen3.5 Think-with-Images Local Demo',
        description='Qwen3.5 local-GPU demo for image reasoning with tool calling.',
        system_message=system_message,
    )

    return bot


def test(pic_url: str, query: str):
    bot = init_agent_service()

    messages = [{
        'role': 'user',
        'content': [
            {
                'image': pic_url
            },
            {
                'text': query
            },
        ]
    }]

    response_plain_text = ''
    for response in bot.run(messages=messages):
        response_plain_text = multimodal_typewriter_print(response, response_plain_text)


if __name__ == '__main__':
    test(
        'https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg',
        '告诉我这只狗的品种，并给出你使用到的关键视觉证据。',
    )
