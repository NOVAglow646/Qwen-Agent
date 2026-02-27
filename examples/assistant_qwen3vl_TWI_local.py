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
"""An agent implemented by FnCallAgent with Qwen3-VL (local TWI)."""

import os

from qwen_agent.agents import FnCallAgent


def init_agent_service():
    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': os.getenv('LOCAL_MODEL_NAME', 'qwen3-vl-235b-a22b-instruct'),
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
        'image_draw_lines_tool',
        'image_draw_bboxes_tool',
    ]

    bot = FnCallAgent(
        llm=llm_cfg,
        function_list=tools,
        name='Qwen3VL TWI Local Demo',
        system_message='',
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

    response = list(bot.run(messages=messages))[-1]
    print(response)

    response_plain_text = response[-1]['content']
    print('\n\nFinal Response:\n', response_plain_text)


if __name__ == '__main__':
    test('https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg',
         '先通过缩放、画线和画框标注关键视觉证据，再判断这只狗的品种。')
