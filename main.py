import uvicorn
from fastapi import FastAPI

# Import settings and client initialization
from app.core.config import settings
from app.core.openai_client import initialize_openai_client

# Import API routers
from app.api.routes import chat as chat_router
from app.api.routes import root as root_router

# --- Application Creation ---
def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    # Initialize the OpenAI client early (optional, can also be done on first request)
    # Doing it here ensures config issues are caught at startup.
    initialize_openai_client()

    # Create FastAPI app instance
    app_instance = FastAPI(
        title="OpenAI Proxy API with Tool & Think Tag Parsing",
        description="A proxy for OpenAI chat completions that parses <tool_call> and <think> tags.",
        version="1.0.0"
    )

    # --- Include Routers ---
    # Include the router for the root endpoint ("/")
    app_instance.include_router(root_router.router, tags=["Status"])
    # Include the router for chat completion endpoints ("/chat/completions", "/v1/chat/completions")
    app_instance.include_router(chat_router.router, prefix="/v1", tags=["Chat Completions v1"]) # Add prefix for v1 explicitly
    app_instance.include_router(chat_router.router, tags=["Chat Completions"]) # Include without prefix too


    return app_instance

# Create the app instance by calling the factory function
app = create_app()

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting server on {settings.host}:{settings.port}")
    # Run the Uvicorn server
    uvicorn.run(
        "main:app", # Reference the app instance created above
        host=settings.host,
        port=settings.port,
        reload=False # Set reload=True for development (watches for file changes)
        # workers=4 # Adjust number of workers for production
    )