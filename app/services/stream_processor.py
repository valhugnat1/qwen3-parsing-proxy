import re
import json
import uuid
from typing import Dict, Any, Optional

class ContentParser:
    @staticmethod
    def parse_and_clean_content(raw_content: Optional[str]) -> Dict[str, Any]:
        """Parse <tool_call> and <think> tags from raw content."""
        if not raw_content:
            return {"cleaned_content": None, "parsed_tool_calls": None, "reasoning_content": None}

        parsed_tool_calls = []
        reasoning_parts = []
        cleaned_parts = []
        last_end = 0

        # Regex to find <tool_call>...</tool_call> or <think>...</think>
        # It captures the content inside each tag type separately.
        tag_regex = re.compile(r"(?:<tool_call>(.*?)</tool_call>)|(?:<think>(.*?)</think>)", re.DOTALL)

        for match in tag_regex.finditer(raw_content):
            start, end = match.span()
            # Append content *before* the current tag
            cleaned_parts.append(raw_content[last_end:start])

            tool_call_content = match.group(1) # Content of <tool_call> if matched
            think_content = match.group(2)     # Content of <think> if matched

            if tool_call_content is not None:  # Process <tool_call>
                try:
                    # Attempt to parse the JSON content within the tag
                    tool_call_data = json.loads(tool_call_content.strip())
                    # Check for required keys
                    if isinstance(tool_call_data, dict) and "name" in tool_call_data and "arguments" in tool_call_data:
                        # Ensure arguments are a valid JSON string
                        arguments_str = ContentParser._process_arguments(tool_call_data["arguments"])

                        # Create the structured tool call object
                        tool_call_id = f"call_{uuid.uuid4().hex[:24]}" # Generate unique ID
                        parsed_tool_calls.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call_data["name"],
                                "arguments": arguments_str
                            }
                        })
                    else:
                        # Invalid structure, treat the whole tag as plain content
                        cleaned_parts.append(raw_content[start:end])
                except json.JSONDecodeError:
                    # Invalid JSON inside tag, treat the whole tag as plain content
                    cleaned_parts.append(raw_content[start:end])
            elif think_content is not None:  # Process <think>
                # Add the content inside <think> tags to reasoning parts
                reasoning_parts.append(think_content.strip())

            last_end = end # Update position for the next iteration

        # Append any remaining content *after* the last tag
        cleaned_parts.append(raw_content[last_end:])

        # Combine cleaned parts and strip whitespace
        final_cleaned_content = "".join(cleaned_parts).strip() or None
        # Combine reasoning parts
        final_reasoning_content = "\n".join(reasoning_parts).strip() if reasoning_parts else None

        return {
            "cleaned_content": final_cleaned_content,
            "parsed_tool_calls": parsed_tool_calls if parsed_tool_calls else None,
            "reasoning_content": final_reasoning_content
        }

    @staticmethod
    def _process_arguments(arguments) -> str:
        """
        Ensures tool call arguments are represented as a valid JSON string.
        If 'arguments' is already a string, it tries to parse and re-dump it
        to validate/normalize formatting. If it's not a valid JSON string,
        it's returned as is. If it's another type (dict, list), it's JSON dumped.
        """
        if isinstance(arguments, str):
            try:
                # Try parsing to see if it's valid JSON already
                arguments_obj = json.loads(arguments)
                # Re-dump to ensure consistent formatting
                return json.dumps(arguments_obj)
            except json.JSONDecodeError:
                # It's a string, but not valid JSON, return as is
                # This handles cases where the model might return non-JSON arguments
                # enclosed in <tool_call>
                return arguments # Return the original string
        elif isinstance(arguments, (dict, list)):
            # Dump dicts or lists directly
             return json.dumps(arguments)
        else:
             # For other types (int, float, bool, None), convert to string representation
             # This might be less common but handles edge cases. Consider if JSON dumping is better.
             return json.dumps(arguments) # Dumps as JSON primitive, e.g. "123", "true", "null"