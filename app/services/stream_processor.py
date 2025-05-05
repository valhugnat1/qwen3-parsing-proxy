import json
import uuid
from typing import Generator, Tuple, Optional, Any

from app.services.content_parser import ContentParser # Import the parser

# Define StreamingState class here as it's tightly coupled with the processor
class StreamingState:
    def __init__(self):
        self.content_buffer = ""  # Holds raw content from current chunk being processed
        self.current_tool_call_index = 0  # Index for next yielded tool call
        self.role_sent = False  # Track if the initial role delta ('assistant') was sent
        self.streamed_tool_calls_exist = False  # Track if any tool calls were yielded (affects finish_reason)
        self.in_think_block = False  # Currently processing content inside <think>...</think>
        self.in_tool_call_block = False  # Currently processing content inside <tool_call>...</tool_call>
        self.tool_call_buffer = ""  # Buffer for content between <tool_call> and </tool_call> tags

class StreamProcessor:
    @staticmethod
    def process_tag_actions(action: str, state: StreamingState) -> None:
        """Updates the StreamingState based on encountered tags."""
        if action == "think_open":
            state.in_think_block = True
        elif action == "think_close":
            state.in_think_block = False
        elif action == "tool_call_open":
            state.in_tool_call_block = True
            state.tool_call_buffer = ""  # Reset buffer when opening tag is found
        elif action == "tool_call_close":
            state.in_tool_call_block = False
            # Buffer processing happens *after* this in the main loop

    @staticmethod
    def find_next_tag(content_buffer: str, start_idx: int, state: StreamingState) -> Tuple[int, str, int]:
        """
        Finds the next opening or closing tag relevant to the current state.
        Returns: (position, action, tag_length)
        Position is float('inf') if no relevant tag is found.
        Action is 'content' if no tag found or segment before tag.
        """
        first_match_pos = float('inf')
        action = "content"  # Default action: yield content
        tag_len = 0        # Length of the tag found

        # --- Search for Opening Tags (only if not already inside a block) ---
        if not state.in_think_block and not state.in_tool_call_block:
            think_open_pos = content_buffer.find("<think>", start_idx)
            tool_open_pos = content_buffer.find("<tool_call>", start_idx)

            # Find the *earliest* opening tag
            if think_open_pos != -1 and think_open_pos < first_match_pos:
                first_match_pos = think_open_pos
                action = "think_open"
                tag_len = len("<think>")
            if tool_open_pos != -1 and tool_open_pos < first_match_pos:
                first_match_pos = tool_open_pos
                action = "tool_call_open"
                tag_len = len("<tool_call>")

        # --- Search for Closing Tags (only if inside a specific block) ---
        elif state.in_think_block:
            think_close_pos = content_buffer.find("</think>", start_idx)
            if think_close_pos != -1:
                # If we find the closing tag, that's our next action point
                first_match_pos = think_close_pos
                action = "think_close"
                tag_len = len("</think>")
                # No need to check other tags if we are inside <think>

        elif state.in_tool_call_block:
            tool_close_pos = content_buffer.find("</tool_call>", start_idx)
            if tool_close_pos != -1:
                # If we find the closing tag, that's our next action point
                first_match_pos = tool_close_pos
                action = "tool_call_close"
                tag_len = len("</tool_call>")
                # No need to check other tags if we are inside <tool_call>

        # If first_match_pos is still infinity, no relevant tags were found
        # in the remaining buffer from start_idx. Action remains 'content'.
        # The position effectively becomes the end of the buffer.
        return int(first_match_pos) if first_match_pos != float('inf') else len(content_buffer), action, tag_len

    @staticmethod
    def process_tool_call_close(
        chunk: Any, choice: Any, state: StreamingState, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """
        Processes the buffered content when a </tool_call> tag is encountered.
        Yields structured tool call chunks or raw content if parsing fails.
        """
        try:
            # Attempt to parse the buffered content as JSON
            tool_call_data = json.loads(state.tool_call_buffer.strip())
            # Validate structure
            if isinstance(tool_call_data, dict) and "name" in tool_call_data and "arguments" in tool_call_data:
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                # Ensure arguments are correctly formatted as a JSON string
                arguments_str = ContentParser._process_arguments(tool_call_data["arguments"])

                # Yield the chunk for the tool call's name
                yield from StreamProcessor.yield_tool_call_name_chunk(
                    chunk, choice, state, tool_call_id,
                    tool_call_data["name"], role_to_yield
                )

                # Yield the chunk for the tool call's arguments (only if arguments exist)
                # Note: Arguments can be an empty string "" which is valid JSON,
                # so we yield even then. We might reconsider if "" should be skipped.
                # The spec implies an empty arguments string is valid.
                yield from StreamProcessor.yield_tool_call_args_chunk(
                    chunk, choice, state, arguments_str
                )

                state.current_tool_call_index += 1  # Increment for the next tool call
                state.streamed_tool_calls_exist = True # Mark that we yielded a tool call
            else:
                # Parsed JSON lacks required 'name' or 'arguments' keys
                raise ValueError("Parsed tool call missing 'name' or 'arguments'")
        except (json.JSONDecodeError, ValueError) as e:
            # Parsing failed or structure was invalid, yield the raw content instead
            print(f"Warning: Failed to parse tool call JSON ({e}). Yielding raw content: {state.tool_call_buffer}")
            yield from StreamProcessor.yield_raw_tool_content(
                chunk, choice, state, role_to_yield
            )
        finally:
            # Always clear the buffer after processing the closing tag
            state.tool_call_buffer = ""


    @staticmethod
    def yield_tool_call_name_chunk(
        chunk: Any, choice: Any, state: StreamingState, tool_call_id: str, name: str, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """Yields the stream chunk containing the tool call's name and ID."""
        base_chunk_dict = chunk.model_dump(exclude={'choices'})
        delta_payload = {
            "content": None, # Explicitly null content for tool call delta
            "tool_calls": [{
                "index": state.current_tool_call_index,
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": "" # Start with empty arguments string
                }
            }]
        }
        # Add role only if it's the very first delta being sent
        if role_to_yield and not state.role_sent:
            delta_payload["role"] = role_to_yield
            state.role_sent = True # Mark role as sent

        # Remove keys with None values from delta (like 'content' if it was None)
        # Role might be None if it wasn't the first delta.
        # delta_payload = {k: v for k, v in delta_payload.items() if v is not None}

        base_chunk_dict["choices"] = [{
            "index": choice.index,
            "delta": delta_payload,
            "finish_reason": None,
            "logprobs": None # Logprobs usually not in tool call deltas
        }]
        yield f"data: {json.dumps(base_chunk_dict)}\n\n"


    @staticmethod
    def yield_tool_call_args_chunk(
        chunk: Any, choice: Any, state: StreamingState, arguments_str: str
    ) -> Generator[str, None, None]:
        """Yields the stream chunk containing the tool call's arguments."""
        base_chunk_dict = chunk.model_dump(exclude={'choices'})
        delta_payload = {
            # No role or content in argument chunks
            "tool_calls": [{
                "index": state.current_tool_call_index,
                # ID is not needed here, just the function arguments part
                "function": {"arguments": arguments_str}
            }]
        }

        base_chunk_dict["choices"] = [{
            "index": choice.index,
            "delta": delta_payload,
            "finish_reason": None,
            "logprobs": None
        }]
        yield f"data: {json.dumps(base_chunk_dict)}\n\n"

    @staticmethod
    def yield_raw_tool_content(
        chunk: Any, choice: Any, state: StreamingState, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """Yields raw content when tool call parsing fails, including the tags."""
        raw_tool_content = f"<tool_call>{state.tool_call_buffer}</tool_call>"
        yield from StreamProcessor.yield_content_segment(
            chunk, choice, state, raw_tool_content, role_to_yield
        )
        # state.tool_call_buffer is cleared in the calling function (process_tool_call_close)

    @staticmethod
    def yield_content_segment(
        chunk: Any, choice: Any, state: StreamingState, content_segment: str, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """Yields a regular content chunk or a reasoning content chunk."""
        if not content_segment: # Don't yield empty content segments
            return

        base_chunk_dict = chunk.model_dump(exclude={'choices'})
        delta_payload = {}

        # Determine if this is reasoning content (inside <think>) or regular content
        if state.in_think_block:
             # Add non-standard 'reasoning_content' field
            delta_payload["reasoning_content"] = content_segment
            # Standard 'content' might be null or absent for reasoning, TBD based on client needs
            delta_payload["content"] = None # Or omit content key entirely
        else:
            delta_payload["content"] = content_segment

        # Add role only if it's the very first delta being sent
        if role_to_yield and not state.role_sent:
            delta_payload["role"] = role_to_yield
            state.role_sent = True # Mark role as sent

        # Ensure delta is not empty before adding choice
        if not delta_payload:
             return

        # Remove keys with None values before yielding
        delta_payload = {k:v for k,v in delta_payload.items() if v is not None}

        # Only yield if delta has content or role
        if delta_payload:
            yield_choice = {
                "index": choice.index,
                "delta": delta_payload,
                "finish_reason": None,
                "logprobs": None # Logprobs usually not in content deltas
            }
            base_chunk_dict["choices"] = [yield_choice]
            yield f"data: {json.dumps(base_chunk_dict)}\n\n"

    @staticmethod
    def yield_native_tool_call(
        chunk: Any, choice: Any, state: StreamingState, current_chunk_tool_calls: list, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """Yields native tool calls received directly from the downstream API."""
        base_chunk_dict = chunk.model_dump(exclude={'choices'})
        delta_payload = {
            "content": None, # Explicitly null content
            # Dump each tool call model instance, excluding None values within them
            "tool_calls": [tc.model_dump(exclude_none=True) for tc in current_chunk_tool_calls]
        }

        # Add role only if it's the very first delta being sent
        if role_to_yield and not state.role_sent:
            delta_payload["role"] = role_to_yield
            state.role_sent = True # Mark role as sent

        # delta_payload = {k: v for k, v in delta_payload.items() if v is not None}

        base_chunk_dict["choices"] = [{
            "index": choice.index,
            "delta": delta_payload,
            "finish_reason": None,
            "logprobs": None
        }]
        yield f"data: {json.dumps(base_chunk_dict)}\n\n"


    @staticmethod
    def yield_finish_chunk(
        chunk: Any, choice: Any, state: StreamingState, finish_reason: str, role_to_yield: Optional[str]
    ) -> Generator[str, None, None]:
        """Yields the final chunk indicating the reason for stopping."""
        final_chunk_dict = chunk.model_dump(exclude={'choices'})
        final_reason = finish_reason

        # If we streamed any *parsed* tool calls (<tool_call> tags),
        # and the original reason was 'stop', override it to 'tool_calls'.
        # Native tool calls from downstream should already have 'tool_calls' finish_reason.
        if state.streamed_tool_calls_exist and finish_reason == 'stop':
            final_reason = 'tool_calls'

        final_delta = {}
        # Add role if it hasn't been sent at all during the stream (e.g., empty response)
        if role_to_yield and not state.role_sent:
            final_delta["role"] = role_to_yield
            # Note: Sending role in the *final* chunk is unusual but might be necessary
            # if the stream was empty until the finish reason.

        # Ensure delta isn't empty, typically it is for finish chunk unless adding role.
        # OpenAI spec usually has delta: {} or delta: null in the final chunk.
        # Sending delta: {"role": "assistant"} might be unexpected by clients here.
        # Let's stick to the standard empty delta unless forced by no prior role.
        # If final_delta remains empty, clients expect delta to be null or {}.
        # However, pydantic might strip it if empty. Let's send {} for clarity.

        final_chunk_dict["choices"] = [{
            "index": choice.index,
             # Send empty dict if no role needed, or the role if needed.
            "delta": final_delta if final_delta else {},
            "finish_reason": final_reason,
             # Pass through logprobs if they exist in the original final chunk
            "logprobs": choice.logprobs.model_dump(exclude_none=True) if choice.logprobs else None
        }]
        yield f"data: {json.dumps(final_chunk_dict)}\n\n"