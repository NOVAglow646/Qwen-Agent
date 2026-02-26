import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/mmu_vcg_ssd/shiyang06/Tool/API/mmu-gemini-caption-1-5pro-86ec97219196.json"

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "global")
GEMINI_SEED = 42

# Match gemini2.5-pro_example.py defaults
_GEMINI_SAFETY_SETTINGS = [
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

_GEMINI_CONFIG = types.GenerateContentConfig(
    temperature=0,
    top_p=0.001,
    thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=-1),
    safety_settings=_GEMINI_SAFETY_SETTINGS,
    seed=GEMINI_SEED,
)


def build_deepseek_client() -> OpenAI:
    """Create a DeepSeek client using environment configuration."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def deepseek_chat(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
    model: str = DEEPSEEK_MODEL,
) -> str:
    """Send a chat completion request to DeepSeek and return the text reply."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content


def _resolve_gemini_credentials(credential_path: str | None = None) -> str:
    """Resolve GOOGLE_APPLICATION_CREDENTIALS like the example code."""
    if credential_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        default_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if default_path.exists():
            cred_path = str(default_path)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    if not cred_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set; set it or pass credential_path to build_gemini_client"
        )
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Credential file not found: {cred_path}")
    return cred_path


def build_gemini_client(credential_path: str | None = None) -> genai.Client:
    """Create a Gemini client using the example's auth and location settings."""
    cred_path = _resolve_gemini_credentials(credential_path)
    with open(cred_path, "r", encoding="utf-8") as f:
        cred = json.load(f)
    project_id = cred.get("project_id")
    if not project_id:
        raise RuntimeError("project_id is missing in credential file")

    return genai.Client(vertexai=True, project=project_id, location=GEMINI_LOCATION)


def gemini_generate(
    client: genai.Client,
    user_prompt: str,
    *,
    system_prompt: str = "",
    temperature: float | None = None,
    model: str = GEMINI_MODEL,
) -> str:
    """Generate a text response from Gemini using the example config."""
    prompt_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    config = _GEMINI_CONFIG
    if temperature is not None and temperature != _GEMINI_CONFIG.temperature:
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=_GEMINI_CONFIG.top_p,
            thinking_config=_GEMINI_CONFIG.thinking_config,
            safety_settings=_GEMINI_CONFIG.safety_settings,
            seed=_GEMINI_CONFIG.seed,
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )
    for part in response.candidates[0].content.parts:
        if part.text:
            return part.text
    return ""
