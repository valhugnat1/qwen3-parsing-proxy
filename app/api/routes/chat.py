import time
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from openai import OpenAI

from app.models.chat import ChatCompletionRequest
from app.core.openai_client import get_openai_client # Dependency to get client
from app.services.openai_handler import OpenAIAPIHandler # Import the handler logic

router = APIRouter()

@router.post("/v1/chat/completions")
@router.post("/chat/completions") # Allow both paths
async def chat_completions(
    request: ChatCompletionRequest,
    client: OpenAI = Depends(get_openai_client) # Inject OpenAI client dependency
):
    """
    Handles chat completion requests, supporting streaming and non-streaming modes,
    as well as parsing for <tool_call> and <think> tags.
    """
    try:
        # 1. Prepare request arguments for OpenAI client
        kwargs = OpenAIAPIHandler.prepare_request_kwargs(request)

        # 2. Execute request using the injected client
        response = client.chat.completions.create(**kwargs)

        # 3. Handle streaming response
        if request.stream:
            # Build the generator that processes chunks and parses tags
            stream_generator = OpenAIAPIHandler.build_streaming_generator(response)
            return StreamingResponse(stream_generator, media_type="text/event-stream")

        # 4. Handle non-streaming response
        else:
            # Construct the base response payload
            response_payload = {
                "id": response.id,
                "object": "chat.completion",
                "created": response.created or int(time.time()), # Use current time if 'created' is missing
                "model": response.model,
                "choices": [], # To be populated below
                "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
                "system_fingerprint": response.system_fingerprint,
            }

            final_choices = []
            for i, choice in enumerate(response.choices):
                message = choice.message
                original_content = message.content
                original_finish_reason = choice.finish_reason

                # Process the message content for tags and structure
                choice_message_dict, final_finish_reason = OpenAIAPIHandler.process_non_streaming_response(
                    response, original_content, original_finish_reason, message
                )

                # Append the processed choice to the list
                final_choices.append({
                    "message": choice_message_dict,
                    "index": i,
                    "finish_reason": final_finish_reason,
                    # Include logprobs if available, excluding None values within
                    "logprobs": choice.logprobs.model_dump(exclude_none=True) if choice.logprobs else None
                })

            # Add the processed choices to the response payload
            response_payload["choices"] = final_choices
            return response_payload

    except Exception as e:
        # Format and raise appropriate HTTPException for any errors
        raise OpenAIAPIHandler.format_error_response(e)