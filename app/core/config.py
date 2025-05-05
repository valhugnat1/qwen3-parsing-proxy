import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

class Settings(BaseModel):
    openai_base_url: str = Field(default="https://api.openai.com/v1") # Default to OpenAI
    openai_api_key: Optional[str] = None
    port: int = Field(default=8000)
    host: str = Field(default="0.0.0.0")

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

def load_config() -> Settings:
    load_dotenv()  # Load environment variables from .env file

    # Prioritize Fireworks API key if set, otherwise use OpenAI key
    api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")
    # Use Fireworks base URL if its key is set and no base URL is explicitly defined
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("FIREWORKS_API_KEY"):
         base_url = "https://api.fireworks.ai/inference/v1"
    elif not base_url:
        base_url = "https://api.openai.com/v1" # Default if nothing else specified

    # Note: We initialize Settings directly here which might seem redundant
    # with Pydantic's built-in .env loading, but it allows prioritizing
    # FIREWORKS_API_KEY and its default URL correctly.
    return Settings(
        openai_base_url=base_url,
        openai_api_key=api_key,
        port=int(os.getenv("PORT", 8000)),
        host=os.getenv("HOST", "0.0.0.0")
    )

settings = load_config()