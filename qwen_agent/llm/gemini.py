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

import base64
import copy
import json
import mimetypes
import os
from pprint import pformat
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, ContentItem, Message

try:
    from google import genai
    from google.genai import types  # type: ignore[import-not-found]
except ImportError:
    genai = None
    types = None


GEMINI_MODEL = 'gemini-2.5-pro'
GEMINI_LOCATION = os.environ.get('GEMINI_LOCATION', 'global')
GEMINI_SEED = 42


def _resolve_gemini_credentials(credential_path: Optional[str] = None) -> str:
    """Resolve GOOGLE_APPLICATION_CREDENTIALS like the example code."""
    if credential_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path:
        default_path = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
        if default_path.exists():
            cred_path = str(default_path)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path

    if not cred_path:
        raise RuntimeError(
            'GOOGLE_APPLICATION_CREDENTIALS is not set; set it or pass credential_path in llm cfg to use gemini')
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f'Credential file not found: {cred_path}')
    return cred_path


def _build_gemini_client(location: str, credential_path: Optional[str] = None):
    """Create a Gemini client using the example's auth and location settings."""
    cred_path = _resolve_gemini_credentials(credential_path)
    with open(cred_path, 'r', encoding='utf-8') as f:
        cred = json.load(f)
    project_id = cred.get('project_id')
    if not project_id:
        raise RuntimeError('project_id is missing in credential file')

    return genai.Client(vertexai=True, project=project_id, location=location)


def _decode_data_uri(data_uri: str) -> tuple[str, bytes]:
    header, encoded = data_uri.split(',', 1)
    mime_type = header[5:].split(';', 1)[0] if header.startswith('data:') else 'image/png'
    return mime_type or 'image/png', base64.b64decode(encoded)


def _image_to_part(v: str):
    if v.startswith('file://'):
        v = v[len('file://'):]

    if v.startswith('data:'):
        mime_type, image_bytes = _decode_data_uri(v)
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    if v.startswith(('http://', 'https://')):
        response = requests.get(v, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        mime_type = response.headers.get('Content-Type') or mimetypes.guess_type(v)[0] or 'image/png'
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    if not os.path.exists(v):
        raise ModelServiceError(f'Local file "{v}" does not exist.')
    with open(v, 'rb') as f:
        image_bytes = f.read()
    mime_type = mimetypes.guess_type(v)[0] or 'image/png'
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


@register_llm('gemini')
class GeminiChatAtVertexAI(BaseFnCallModel):

    @property
    def support_multimodal_input(self) -> bool:
        return True

    def __init__(self, cfg: Optional[Dict] = None):
        if genai is None or types is None:
            raise ImportError('google-genai is required for model_type="gemini". Please install it first.')

        super().__init__(cfg)
        cfg = cfg or {}

        self.model = self.model or GEMINI_MODEL
        self.location = cfg.get('location', GEMINI_LOCATION)
        self.seed = cfg.get('seed', GEMINI_SEED)
        self.credential_path = cfg.get('credential_path')
        self.client = _build_gemini_client(location=self.location, credential_path=self.credential_path)

    def _build_generate_config(self, generate_cfg: dict):
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]

        cfg = copy.deepcopy(generate_cfg)
        temperature = cfg.pop('temperature', 0)
        top_p = cfg.pop('top_p', 0.001)
        include_thoughts = cfg.pop('include_thoughts', False)
        thinking_budget = cfg.pop('thinking_budget', -1)
        seed = cfg.pop('seed', self.seed)

        return types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            thinking_config=types.ThinkingConfig(
                include_thoughts=include_thoughts,
                thinking_budget=thinking_budget,
            ),
            safety_settings=safety_settings,
            seed=seed,
        )

    def convert_messages_to_dicts(self, messages: List[Message]) -> List:
        parts = []
        for msg in messages:
            role = msg.role
            content = msg.content
            if isinstance(content, str):
                content = [ContentItem(text=content)] if content else []

            text_segments = []
            for item in content:
                t, v = item.get_type_and_value()
                if t == 'text' and v:
                    text_segments.append(v)
                elif t == 'image' and v:
                    if text_segments:
                        parts.append(types.Part.from_text(text=f'[{role}] ' + '\n'.join(text_segments)))
                        text_segments = []
                    parts.append(_image_to_part(v))
                elif t in ('video', 'audio', 'file') and v:
                    text_segments.append(f'[{t} omitted] {v}')

            if text_segments:
                parts.append(types.Part.from_text(text=f'[{role}] ' + '\n'.join(text_segments)))

            if msg.reasoning_content:
                parts.append(types.Part.from_text(text=f'[reasoning] {msg.reasoning_content}'))

        if not parts:
            parts = [types.Part.from_text(text='')]
        return parts

    @staticmethod
    def _extract_text(response) -> str:
        texts: List[str] = []
        if getattr(response, 'candidates', None):
            for part in response.candidates[0].content.parts:
                text = getattr(part, 'text', None)
                if text:
                    texts.append(text)
        return ''.join(texts)

    @staticmethod
    def _debug_dump_contents(contents: List) -> str:
        debug_items = []
        for part in contents:
            text = getattr(part, 'text', None)
            if text:
                debug_items.append({'type': 'text', 'text': text})
                continue

            inline_data = getattr(part, 'inline_data', None)
            if inline_data is not None:
                mime_type = getattr(inline_data, 'mime_type', '')
                data = getattr(inline_data, 'data', b'')
                byte_length = len(data) if data else 0
                debug_items.append({'type': 'inline_data', 'mime_type': mime_type, 'bytes': byte_length})
                continue

            debug_items.append({'type': 'unknown', 'repr': str(part)})
        return pformat(debug_items, width=120)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        response = self._chat_no_stream(messages=messages, generate_cfg=generate_cfg)
        if delta_stream:
            text = response[0].content if response else ''
            yield [Message(role=ASSISTANT, content=text)]
        else:
            yield response

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        try:
            contents = self.convert_messages_to_dicts(messages)
            print('\n[Gemini Raw Prompt / contents]\n' + self._debug_dump_contents(contents))
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._build_generate_config(generate_cfg),
            )
            text = self._extract_text(response)
            return [Message(role=ASSISTANT, content=text)]
        except Exception as ex:
            raise ModelServiceError(exception=ex)
