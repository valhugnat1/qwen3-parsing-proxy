import time
import json
import traceback
from typing import Dict, Any, Generator, Optional
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletionMessage, ChatCompletionChunk

from app.models.chat import ChatCompletionRequest
from app.services.content_parser import ContentParser
from app.services.stream_processor import StreamProcessor, StreamingState # Import state too

class OpenAIAPIHandler:
    @staticmethod
    def prepare_request_kwargs(request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare kwargs dictionary for the OpenAI API request from Pydantic model."""
        # Convert ChatMessage Pydantic models to dictionaries, excluding None values
        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        # Get other request parameters, excluding None values
        kwargs = request.model_dump(exclude_none=True)
        # Replace messages list with the processed one
        kwargs["messages"] = messages
        return kwargs

    @staticmethod
    def process_non_streaming_response(
        response: Any, # Type hint could be improved if OpenAI SDK types are stable
        original_content: Optional[str],
        original_finish_reason: str,
        message: ChatCompletionMessage # Use the specific type hint
    ) -> tuple[Dict[str, Any], str]:
        """
        Processes a non-streaming response.
        Parses <tool_call> and <think> tags from content, updates message dictionary,
        and determines the final finish reason.
        """
        # Parse the original content for <tool_call> and <think> tags
        parsed_data = ContentParser.parse_and_clean_content(original_content)
        cleaned_content = parsed_data["cleaned_content"]
        parsed_tools = parsed_data["parsed_tool_calls"]
        reasoning_content = parsed_data["reasoning_content"] # Extract reasoning

        # Start building the final message dictionary for the choice
        choice_message_dict = {
            "role": message.role or "assistant", # Default to assistant if role is missing
            "content": cleaned_content, # Use cleaned content
            # Initialize tool_calls; will be updated based on parsing/original message
            "tool_calls": None
        }

        # Add reasoning_content field if present (non-standard)
        if reasoning_content:
            choice_message_dict["reasoning_content"] = reasoning_content

        final_tool_calls = None
        final_finish_reason = original_finish_reason # Start with original reason

        # --- Determine final tool_calls and finish_reason ---
        if parsed_tools:
            # Use tools parsed from <tool_call> tags
            final_tool_calls = parsed_tools
            # OpenAI spec: If tool_calls are present, content should be null/None
            # We apply this if *only* tool calls remain after cleaning.
            # If there's also cleaned_content, we keep both (might deviate slightly from spec).
            if not cleaned_content:
                choice_message_dict["content"] = None # Set content to None

            # If we parsed tools, the finish reason should be 'tool_calls'
            final_finish_reason = "tool_calls"

        elif message.tool_calls:
            # Use native tool calls if no <tool_call> tags were parsed
            final_tool_calls = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
             # Apply OpenAI spec: content is None if native tool calls exist
            # Again, we only do this if cleaned_content is also empty.
            if not cleaned_content:
                 choice_message_dict["content"] = None

            # If native tool calls exist, the finish_reason should already be 'tool_calls'
            # If not, we perhaps shouldn't force it, but parsing took precedence anyway.
            # final_finish_reason = "tool_calls" # Usually already set by OpenAI

        # Update the message dictionary with the final tool calls
        choice_message_dict["tool_calls"] = final_tool_calls

        return choice_message_dict, final_finish_reason


    @staticmethod
    def build_streaming_generator(response_stream: Any) -> Generator[str, None, None]:
        """
        Builds the generator for processing and yielding streaming responses.
        Handles tag parsing (<tool_call>, <think>) within the stream.
        """
        # Create a state object to track streaming progress and context
        state = StreamingState()

        def stream_generator() -> Generator[str, None, None]:
            try:
                for chunk in response_stream:
                    # --- Basic Chunk Handling ---
                    if not isinstance(chunk, ChatCompletionChunk) or not chunk.choices:
                        # Yield non-standard chunks or metadata chunks directly
                        # Ensure it's serializable JSON first
                        try:
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        except Exception as json_err:
                            print(f"Warning: Could not serialize non-choice chunk: {json_err}")
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta
                    finish_reason = choice.finish_reason # Can be None until the last chunk

                    # Extract potential delta fields
                    current_chunk_role = delta.role if delta else None
                    current_chunk_content = delta.content if delta else None
                    current_chunk_tool_calls = delta.tool_calls if delta else None # List of tool call chunks

                    # --- Determine Role for First Delta ---
                    # The 'role' should only appear in the very first delta message.
                    role_to_yield_on_first_delta = None
                    if not state.role_sent:
                        if current_chunk_role:
                            role_to_yield_on_first_delta = current_chunk_role
                        # If role isn't explicitly in this chunk, but content or tool calls are,
                        # and we haven't sent the role yet, assume 'assistant'.
                        elif current_chunk_content or current_chunk_tool_calls:
                            role_to_yield_on_first_delta = "assistant"
                        # If it's the final chunk with a finish reason but still no role sent,
                        # we might need to add it there (handled later).

                    # --- Handle Native Tool Calls ---
                    if current_chunk_tool_calls:
                        # Reset tag parsing state if native calls appear
                        state.in_think_block = False
                        state.in_tool_call_block = False
                        state.tool_call_buffer = "" # Clear any pending tag buffer
                        # Clear main content buffer too, native calls take precedence
                        # Any buffered content before this native call is likely dropped/ignored
                        # This assumes native calls won't be interleaved mid-tag parsing.
                        state.content_buffer = ""

                        yield from StreamProcessor.yield_native_tool_call(
                            chunk, choice, state, current_chunk_tool_calls, role_to_yield_on_first_delta
                        )
                        # state.role_sent is updated within yield_native_tool_call if role was yielded

                        # Update tool call index based on the *last* tool call index in the chunk
                        last_tool_call = current_chunk_tool_calls[-1]
                        if last_tool_call.index is not None:
                            state.current_tool_call_index = last_tool_call.index + 1
                        # Mark that *some* tool call (native or parsed) has been streamed
                        state.streamed_tool_calls_exist = True # Native calls count

                        # If the same chunk *also* has content (unusual for tool calls), buffer it.
                        if current_chunk_content:
                            state.content_buffer += current_chunk_content
                        continue # Process next chunk

                    # --- Buffer Incoming Content for Tag Parsing ---
                    if current_chunk_content:
                        state.content_buffer += current_chunk_content

                    # --- Process Buffered Content Segment by Segment ---
                    processed_upto_buffer_idx = 0
                    while processed_upto_buffer_idx < len(state.content_buffer):
                        # Find the next relevant tag in the remaining buffer
                        match_pos, action, tag_len = StreamProcessor.find_next_tag(
                            state.content_buffer, processed_upto_buffer_idx, state
                        )

                        # Identify the segment of content *before* the found tag (or end of buffer)
                        # Ensure segment_end doesn't exceed buffer length
                        segment_end = min(match_pos, len(state.content_buffer))
                        content_segment = state.content_buffer[processed_upto_buffer_idx:segment_end]

                        # Process the identified content segment
                        if content_segment:
                            if state.in_tool_call_block:
                                # If inside <tool_call>, append segment to tool call buffer
                                state.tool_call_buffer += content_segment
                            else:
                                # Otherwise, yield the content segment (as content or reasoning)
                                yield from StreamProcessor.yield_content_segment(
                                    chunk, choice, state, content_segment, role_to_yield_on_first_delta
                                )
                                # state.role_sent is updated within yield_content_segment if role was yielded

                        # Move buffer index past the processed segment
                        processed_upto_buffer_idx = segment_end

                        # Handle the tag action if a tag was found at this position
                        if action != "content" and processed_upto_buffer_idx < len(state.content_buffer):
                            # Update state based on the tag action (open/close think/tool_call)
                            StreamProcessor.process_tag_actions(action, state) # Pass tag_len if needed by actions
                            # Move buffer index past the tag itself
                            processed_upto_buffer_idx += tag_len

                            # If we just closed a tool call tag, process the buffered tool call content
                            if action == "tool_call_close":
                                yield from StreamProcessor.process_tool_call_close(
                                    chunk, choice, state, role_to_yield_on_first_delta # Pass role possibility
                                )
                                # state.role_sent might be updated in process_tool_call_close
                                # tool_call_buffer is cleared within process_tool_call_close

                    # Update the main content buffer, removing the processed parts
                    state.content_buffer = state.content_buffer[processed_upto_buffer_idx:]

                    # --- Handle Finish Reason ---
                    if finish_reason:
                        # Before finishing, check if we are in an incomplete state
                        # (e.g., stream ended mid-<tool_call>)
                        if state.in_tool_call_block and state.tool_call_buffer:
                            # If ended inside <tool_call>, yield the buffered content raw
                            # Prepended with the opening tag as it wasn't yielded.
                            print(f"Warning: Stream ended inside <tool_call>. Yielding raw buffer: {state.tool_call_buffer}")
                            raw_tool_content = f"<tool_call>{state.tool_call_buffer}"
                            yield from StreamProcessor.yield_content_segment(
                                chunk, choice, state, raw_tool_content, role_to_yield_on_first_delta
                            )
                        elif state.in_think_block and state.content_buffer:
                             # Should not happen if logic above is correct, but as a safeguard
                             # If ended inside <think> yield remaining buffer as reasoning
                             print(f"Warning: Stream ended inside <think>. Yielding raw buffer: {state.content_buffer}")
                             yield from StreamProcessor.yield_content_segment(
                                chunk, choice, state, state.content_buffer, role_to_yield_on_first_delta
                            )


                        # Yield the final chunk with the appropriate finish reason
                        yield from StreamProcessor.yield_finish_chunk(
                            chunk, choice, state, finish_reason, role_to_yield_on_first_delta
                        )
                        break # Exit the loop, stream has ended

            except Exception as e:
                # Handle potential errors during streaming
                print("Error during stream processing:")
                traceback.print_exc()
                # Yield an error message in the stream event format
                error_payload = {
                    "error": {
                        "message": f"Internal proxy error during stream: {e}",
                        "type": "proxy_error",
                        "code": 500 # Or another appropriate code
                    }
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                # Always send the [DONE] message, even if errors occurred
                yield "data: [DONE]\n\n"

        # Return the generator function itself
        return stream_generator() # Return the generator instance


    @staticmethod
    def format_error_response(e: Exception) -> HTTPException:
        """Formats various exceptions into FastAPI HTTPException."""
        print("Error processing request:")
        traceback.print_exc() # Log the full traceback for server logs

        if isinstance(e, APIStatusError):
            # Error response from the downstream OpenAI API
            status_code = e.status_code
            detail = f"Downstream API error ({status_code}): {e.message or e.body or e.response.text}"
            # Try to parse the body for more structured error info if needed
            # detail_body = getattr(e, 'body', None) or {}
            # detail_msg = detail_body.get('message', e.message or str(e))
            # detail = f"Downstream API error ({status_code}): {detail_msg}"
            return HTTPException(status_code=status_code, detail=detail)
        elif isinstance(e, APIConnectionError):
            # Failed to connect to the downstream API
            return HTTPException(status_code=503, detail=f"Could not connect to downstream API: {e}")
        elif isinstance(e, HTTPException):
             # If it's already an HTTPException (e.g., from validation), re-raise it
             return e
        else:
            # Catch-all for other unexpected errors
            return HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")