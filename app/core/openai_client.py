from openai import OpenAI
from typing import Optional
import sys

from app.core.config import settings # Use the loaded settings

# Global variable to hold the client instance
_openai_client: Optional[OpenAI] = None

def initialize_openai_client() -> Optional[OpenAI]:
    """
    Initializes and returns the OpenAI client based on settings.
    Exits if the API key is missing.
    """
    global _openai_client
    if _openai_client is None:
        if not settings.openai_api_key:
            print("ERROR: FIREWORKS_API_KEY or OPENAI_API_KEY environment variable not set.")
            print("Please create a .env file or set the variable.")
            sys.exit(1) # Exit if no key is found

        try:
            _openai_client = OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key
            )
            print(f"OpenAI client initialized pointing to: {settings.openai_base_url}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            _openai_client = None # Ensure it's None on failure
            # Decide if you want to exit here too, or let the API endpoint fail
            # sys.exit(1)
    return _openai_client

def get_openai_client() -> OpenAI:
    """
    Returns the initialized OpenAI client.
    Raises an exception if the client is not initialized.
    """
    client = initialize_openai_client() # Ensure initialization attempt
    if client is None:
        # This should ideally not happen if initialize checks properly,
        # but serves as a safeguard.
        raise RuntimeError("OpenAI client is not initialized. Check configuration and logs.")
    return client