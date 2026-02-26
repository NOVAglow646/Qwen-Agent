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
"""An agent implemented by assistant with Gemini 2.5 Pro"""

import os

from qwen_agent.agents import FnCallAgent


def init_agent_service():
    llm_cfg = {
        'model_type': 'gemini',
        'model': 'gemini-2.5-pro',
        'location': os.getenv('GEMINI_LOCATION', 'global'),
        'credential_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        'seed': int(os.getenv('GEMINI_SEED', '42')),
    }

    tools = [
        'image_zoom_in_tool',
        'image_search',
        'web_search',
    ]
    bot = FnCallAgent(
        llm=llm_cfg,
        function_list=tools,
        name='Gemini2.5-pro Agent Demo',
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
         '告诉我这只狗的品种')
