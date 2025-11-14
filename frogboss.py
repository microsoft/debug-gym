#!/usr/bin/env python3
"""
Custom tool parser for vLLM with Hermes-style <tool_call> tags.
Can be used as a module or run directly as a vLLM entry point.
"""
# import the required packages
from typing import Sequence, Union, List
import json
import re
import uuid
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage, FunctionCall, ToolCall, DeltaToolCall, DeltaFunctionCall
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ExtractedToolCallInformation
try:
    from vllm.entrypoints.chat_utils import make_tool_call_id
except ImportError:
    # Fallback if import fails
    def make_tool_call_id():
        return f"chatcmpl-tool-{uuid.uuid4().hex[:24]}"
from vllm.transformers_utils.tokenizer import AnyTokenizer

# define a tool parser and register it to vllm
# the name list in register_module can be used
# in --tool-call-parser. you can define as many
# tool parsers as you want here.
@ToolParserManager.register_module(["froggy"])
class ExampleToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    # adjust request. e.g.: set skip special tokens
    # to False for tool call output.
    def adjust_request(
            self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        return request

    # implement the tool call parse for stream call
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        # For streaming, we need to handle partial tool calls progressively
        # Check if we're currently in a tool call (between <tool_call> tags)
        
        # If there's no delta text, return None
        if not delta_text:
            return None
        
        # Check if we've started a tool call in the current text
        tool_call_started = '<tool_call>' in current_text and '<tool_call>' not in previous_text
        in_tool_call = '<tool_call>' in current_text and '</tool_call>' not in current_text
        tool_call_completed = '</tool_call>' in current_text and '</tool_call>' not in previous_text
        
        # If we just completed a tool call, parse it
        if tool_call_completed:
            # Extract the completed tool call
            pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(pattern, current_text, re.DOTALL)
            
            if matches:
                # Get the last completed tool call
                match = matches[-1]
                try:
                    # Clean and parse JSON
                    json_str = match.strip()
                    json_str = json_str.replace('\\n', '\x00NEWLINE\x00')
                    json_str = json_str.replace('\\t', '\x00TAB\x00')
                    json_str = json_str.replace('\\r', '\x00RETURN\x00')
                    json_str = json_str.replace('\n', '\\n')
                    json_str = json_str.replace('\t', '\\t')
                    json_str = json_str.replace('\r', '\\r')
                    json_str = json_str.replace('\x00NEWLINE\x00', '\\n')
                    json_str = json_str.replace('\x00TAB\x00', '\\t')
                    json_str = json_str.replace('\x00RETURN\x00', '\\r')
                    
                    tool_data = json.loads(json_str)
                    
                    # Handle both single and multiple tool formats
                    tool_calls = []
                    if "tools" in tool_data and isinstance(tool_data["tools"], list):
                        for idx, tool in enumerate(tool_data["tools"]):
                            tool_call = DeltaToolCall(
                                index=idx,
                                id=make_tool_call_id(),
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool["name"],
                                    arguments=json.dumps(tool.get("arguments", {}), ensure_ascii=False, separators=(',', ':'))
                                )
                            )
                            tool_calls.append(tool_call)
                    else:
                        tool_call = DeltaToolCall(
                            index=0,
                            id=make_tool_call_id(),
                            type="function",
                            function=DeltaFunctionCall(
                                name=tool_data["name"],
                                arguments=json.dumps(tool_data.get("arguments", {}), ensure_ascii=False, separators=(',', ':'))
                            )
                        )
                        tool_calls.append(tool_call)
                    
                    # Return delta with tool calls
                    return DeltaMessage(tool_calls=tool_calls)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    # If parsing fails, just return the delta text
                    pass
        
        # If we're inside a tool call tag, don't stream the content
        # (we'll send it all when the tool call completes)
        if in_tool_call and not tool_call_started:
            return DeltaMessage(content="")
        
        # For regular text (not in tool call), stream it normally
        # But filter out the <tool_call> tag itself
        filtered_delta = delta_text.replace('<tool_call>', '').replace('</tool_call>', '')
        
        if filtered_delta:
            return DeltaMessage(content=filtered_delta)
        
        # Return empty content instead of None to keep the stream alive
        return DeltaMessage(content="")

    # implement the tool parse for non-stream call
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Parse <tool_call>...</tool_call> tags (Hermes format)
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, model_output, re.DOTALL)
        
        tool_calls = []
        for i, match in enumerate(matches):
            try:
                # Clean up the JSON string - handle common escaping issues
                # 1. Strip whitespace
                json_str = match.strip()
                
                # 2. Handle improperly escaped newlines in string values
                # This fixes cases where the model outputs literal newlines in JSON strings
                # We need to preserve newlines that are already properly escaped \\n
                # but escape literal newlines that aren't
                
                # First, temporarily replace properly escaped newlines
                json_str = json_str.replace('\\n', '\x00NEWLINE\x00')
                json_str = json_str.replace('\\t', '\x00TAB\x00')
                json_str = json_str.replace('\\r', '\x00RETURN\x00')
                json_str = json_str.replace('\\\\', '\x00BACKSLASH\x00')
                
                # Now escape any literal newlines/tabs/returns
                json_str = json_str.replace('\n', '\\n')
                json_str = json_str.replace('\t', '\\t')
                json_str = json_str.replace('\r', '\\r')
                
                # Restore the properly escaped ones
                json_str = json_str.replace('\x00NEWLINE\x00', '\\n')
                json_str = json_str.replace('\x00TAB\x00', '\\t')
                json_str = json_str.replace('\x00RETURN\x00', '\\r')
                json_str = json_str.replace('\x00BACKSLASH\x00', '\\\\')
                
                # 3. Parse the cleaned JSON
                tool_data = json.loads(json_str)
                
                # 4. Handle both single tool and multiple tools formats
                # Hermes can output {"name": "...", "arguments": {...}}
                # or {"tools": [{"name": "...", "arguments": {...}}]}
                if "tools" in tool_data and isinstance(tool_data["tools"], list):
                    # Multiple tools format
                    for tool in tool_data["tools"]:
                        tool_call = ToolCall(
                            id=make_tool_call_id(),
                            type="function",
                            function=FunctionCall(
                                name=tool["name"],
                                arguments=json.dumps(tool.get("arguments", {}), ensure_ascii=False, separators=(',', ':'))
                            )
                        )
                        tool_calls.append(tool_call)
                else:
                    # Single tool format
                    tool_call = ToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=tool_data["name"],
                            arguments=json.dumps(tool_data.get("arguments", {}), ensure_ascii=False, separators=(',', ':'))
                        )
                    )
                    tool_calls.append(tool_call)
                    
            except (json.JSONDecodeError, KeyError) as e:
                # If parsing fails, log the error with the problematic JSON
                print(f"Failed to parse tool call: {e}")
                print(f"Problematic JSON (first 200 chars): {match[:200]}")
                continue
        
        # Extract text content (everything before first <tool_call>)
        content = re.split(r'<tool_call>', model_output)[0].strip()
        
        # Important: When there are tool calls, always provide a content value (even if empty string)
        # to prevent "no response was returned" errors in clients like Copilot UI.
        # Only set content to None when there are no tool calls AND no content.
        if not content:
            content = "" if len(tool_calls) > 0 else None
            
        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=content
        )


if __name__ == "__main__":
    # When run as a script, start vLLM with this parser registered
    from vllm.entrypoints.cli.main import main
    main()
