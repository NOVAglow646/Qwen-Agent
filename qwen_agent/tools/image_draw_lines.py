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

import os
import uuid
from io import BytesIO
from typing import List, Union

import requests
from PIL import Image, ImageDraw

from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_images_from_messages


@register_tool('image_draw_lines')
class ImageDrawLinesTool(BaseToolWithFileAccess):
    description = 'Draw a line segment on an image using two points.'
    parameters = {
        'type': 'object',
        'properties': {
            'point1_2d': {
                'type': 'array',
                'items': {
                    'type': 'number'
                },
                'minItems': 2,
                'maxItems': 2,
                'description': 'The first point [x1, y1] in normalized 0~1000 coordinates'
            },
            'point2_2d': {
                'type': 'array',
                'items': {
                    'type': 'number'
                },
                'minItems': 2,
                'maxItems': 2,
                'description': 'The second point [x2, y2] in normalized 0~1000 coordinates'
            },
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image (starting from 0). Default: 0'
            }
        },
        'required': ['point1_2d', 'point2_2d']
    }

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        params = self._verify_json_format_args(params)

        point1 = params['point1_2d']
        point2 = params['point2_2d']
        img_idx = int(params.get('img_idx', 0))
        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        if not images:
            return [ContentItem(text='Error: no images found in the messages.')]
        if img_idx >= len(images):
            img_idx = len(images) - 1

        try:
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            elif os.path.exists(image_arg):
                image = Image.open(image_arg).convert('RGB')
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg)).convert('RGB')
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image {images}')]

        try:
            width, height = image.size
            x1 = max(0, min(width, point1[0] / 1000.0 * width))
            y1 = max(0, min(height, point1[1] / 1000.0 * height))
            x2 = max(0, min(width, point2[0] / 1000.0 * width))
            y2 = max(0, min(height, point2[1] / 1000.0 * height))

            draw = ImageDraw.Draw(image)
            line_width = max(2, int(min(width, height) * 0.004))
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=line_width)

            output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
            image.save(output_path)
            return [ContentItem(image=output_path)]
        except Exception as e:
            return [ContentItem(text=f'Tool Execution Error {str(e)}')]
