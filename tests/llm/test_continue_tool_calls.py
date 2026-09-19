# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for continuation with truncated tool-calling responses."""
from unittest.mock import MagicMock, patch

from ms_agent.llm.openai_llm import OpenAI
from ms_agent.llm.utils import Message, ToolCall


def _tool_message():
    return Message(
        role='assistant',
        content='partial response',
        tool_calls=[
            ToolCall(
                id='call_1',
                type='function',
                tool_name='write_file',
                arguments='{}',
                index=0,
            )
        ],
    )


def _completion(finish_reason='length'):
    completion = MagicMock()
    completion.choices[0].finish_reason = finish_reason
    return completion


def test_non_stream_truncated_tool_call_does_not_continue():
    """A dangling tool call must return to the agent loop before another LLM call."""
    llm = OpenAI.__new__(OpenAI)
    messages = [Message(role='user', content='write the report')]
    expected = _tool_message()

    with patch.object(llm, '_format_output_message', return_value=expected), \
            patch.object(llm, '_call_llm_for_continue_gen') as continue_call:
        result = llm._continue_generate(
            messages, _completion(), tools=None, max_runs=1)

    assert result is expected
    continue_call.assert_not_called()


def test_non_stream_plain_truncation_still_continues():
    """The guard must not disable continuation for ordinary truncated text."""
    llm = OpenAI.__new__(OpenAI)
    messages = [Message(role='user', content='write the report')]
    first = Message(role='assistant', content='partial response')
    final = Message(role='assistant', content='done')
    continued = _completion('stop')

    with patch.object(
            llm, '_format_output_message', side_effect=[first, final]), \
            patch.object(
                llm,
                '_call_llm_for_continue_gen',
                return_value=continued) as continue_call:
        result = llm._continue_generate(
            messages, _completion(), tools=None, max_runs=1)

    assert result is final
    continue_call.assert_called_once()
