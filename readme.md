# Qwen3 Proxy with Tool & Think Tag Parsing ✨

This project provides a FastAPI-based proxy server that sits in front of an OpenAI-compatible API (like OpenAI itself, Fireworks AI, etc.). Its primary purpose is to intercept chat completion responses and parse specific XML-like tags (`<tool_call>` and `<think>`) within the content, transforming them into structured data compliant with the OpenAI API specification, while also supporting standard streaming and non-streaming responses.

**Key Features:**

* **OpenAI Compatible Endpoint:** Exposes `/v1/chat/completions` and `/chat/completions` endpoints, mimicking the official OpenAI API.
* **`<tool_call>` Tag Parsing:**
    * Detects `<tool_call>{ "name": "...", "arguments": {...} }</tool_call>` tags in the AI model's response content.
    * Parses the JSON content within the tags.
    * Removes the tags from the main `content` field.
    * Adds a structured `tool_calls` array to the response message, conforming to OpenAI's function/tool calling format.
    * Sets the `finish_reason` to `tool_calls` when tags are successfully parsed (in both streaming and non-streaming modes).
    * Handles malformed JSON or missing keys gracefully by treating the tag as plain text.
* **`<think>` Tag Parsing:**
    * Detects `<think>...</think>` tags used for reasoning or meta-commentary by the model.
    * Extracts the content within these tags.
    * Removes the tags from the main `content` field.
    * Adds the extracted content to a non-standard `reasoning_content` field in the response message (useful for debugging or observing the model's thought process).
* **Streaming Support:** Fully supports `stream=True` requests, parsing tags on-the-fly and yielding Server-Sent Events (`text/event-stream`) with correctly structured deltas for content and tool calls.
* **Native Tool Call Passthrough:** Correctly handles and passes through tool calls generated natively by the downstream API.
* **Clean Code Structure:** Organized into logical modules (API routes, core services, models, configuration) for better maintainability.
* **Configuration via `.env`:** Easy setup using environment variables.

## Prerequisites

* Python 3.8+
* pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Configuration is managed via a `.env` file in the project root.

1.  **Create a `.env` file:**
    ```bash
    cp .env.example .env # If you create an example file, otherwise just create .env
    ```

2.  **Edit the `.env` file:**

    * **`FIREWORKS_API_KEY` (Required):**
        * Provide *at least one* of these API keys.
        * If `FIREWORKS_API_KEY` is set, it will be used. Otherwise, `OPENAI_API_KEY` will be used.

    * **`HOST` (Optional):**
        * The host address to bind the server to.
        * Defaults to `0.0.0.0` (listens on all available network interfaces).

    * **`PORT` (Optional):**
        * The port number for the server.
        * Defaults to `8000`.

**Example `.env`:**

```dotenv
# Use Fireworks AI
FIREWORKS_API_KEY=your_fireworks_api_key_here

# Optional Server Config
# HOST=127.0.0.1
# PORT=8001# qwen3-parsing-proxy
# qwen3-parsing-proxy

Running the Application

Start the server using Uvicorn:
Bash

python main.py

The server will start on the configured host and port (default: http://0.0.0.0:8000).

For development, you can enable auto-reloading:
Bash

uvicorn main:app --reload --host <your_host> --port <your_port>
# Example: uvicorn main:app --reload --host 0.0.0.0 --port 8000

Usage

Send requests to the /v1/chat/completions (or /chat/completions) endpoint of your running proxy server, just like you would with the OpenAI API.

Example Request (Non-Streaming):
Bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $YOUR_API_KEY" \
  -d '{
    "model": "accounts/fireworks/models/firefunction-v1", # Or your desired model
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to functions."},
      {"role": "user", "content": "What is the weather like in Metz, France?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state/country, e.g. San Francisco, CA"
              },
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'

If the model responds with:

Thinking about the request... <think>User wants weather in Metz, France. I have the get_current_weather tool.</think> Okay, I can look that up. <tool_call>{ "name": "get_current_weather", "arguments": { "location": "Metz, France", "unit": "celsius" } }</tool_call>

The proxy will return:
JSON

{
  "id": "...",
  "object": "chat.completion",
  "created": ...,
  "model": "...",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Okay, I can look that up.", // Tags removed
        "tool_calls": [ // Structured tool call added
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Paris, France\", \"unit\": \"celsius\"}" // Arguments as JSON string
            }
          }
        ],
        "reasoning_content": "User wants weather in Metz, France. I have the get_current_weather tool." // Extracted think content
      },
      "logprobs": null,
      "finish_reason": "tool_calls" // Finish reason updated
    }
  ],
  "usage": { ... },
  "system_fingerprint": "..."
}

Example Request (Streaming):
Bash

curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $YOUR_API_KEY" \
  -d '{
    "model": "accounts/fireworks/models/firefunction-v1",
    "messages": [
      {"role": "user", "content": "Call the dummy function."}
    ],
    "tools": [ ... ], # Include tools if needed
    "stream": true
  }'

The proxy will stream back Server-Sent Events, parsing tags and generating appropriate deltas for content and tool calls as they arrive.
Project Structure

.
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
├── main.py               # Main application entry point
└── app/
    ├── api/              # API endpoint definitions (FastAPI routers)
    ├── core/             # Core components (config, client init)
    ├── models/           # Pydantic data models
    └── services/         # Business logic (tag parsing, streaming, API handling)

Technology Stack

    FastAPI: Web framework
    Pydantic: Data validation and settings management
    OpenAI Python SDK: Interacting with the downstream API
    Uvicorn: ASGI server
    python-dotenv: Loading environment variables
