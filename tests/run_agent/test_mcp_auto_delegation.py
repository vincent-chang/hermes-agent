"""Unit tests for MCP auto-delegation in _execute_tool_calls.

Verifies that:
1. _is_mcp_tool correctly identifies MCP tools by toolset prefix
2. _should_delegate_mcp_batch correctly categorizes MCP vs non-MCP calls
3. Batches with 3+ MCP calls AND non-MCP calls trigger delegation
4. Pure MCP batches (no non-MCP) do NOT trigger delegation (normal concurrent execution)
5. Batches with <3 MCP calls do NOT trigger delegation
6. Non-MCP-only batches do NOT trigger delegation
"""

import json
import pytest
from unittest.mock import MagicMock, patch

import run_agent
from run_agent import AIAgent


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict) -> MagicMock:
    """Build a mock tool_call with a function.name and function.arguments."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.id = f"call_{name}"
    return tc


class _MockAssistantMessage:
    """Minimal mock for assistant_message.tool_calls."""
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


# ---------------------------------------------------------------------------------------
# _is_mcp_tool tests
# ---------------------------------------------------------------------------------------

class TestIsMcpTool:
    """Tests for _is_mcp_tool method."""

    def test_mcp_tool_detected_by_prefix(self):
        """MCP tools are identified by their mcp-* toolset."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = False

        with patch.object(run_agent, 'registry') as mock_registry:
            mock_registry.get_toolset_for_tool.return_value = "mcp-tavily"
            assert agent._is_mcp_tool("tavily_search") is True

            mock_registry.get_toolset_for_tool.return_value = "mcp-firecrawl"
            assert agent._is_mcp_tool("firecrawl_scrape") is True

            mock_registry.get_toolset_for_tool.return_value = "mcp-minimax"
            assert agent._is_mcp_tool("minimax_music_generation") is True

    def test_non_mcp_tool_not_detected(self):
        """Built-in tools return False even if name looks like MCP."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        with patch.object(run_agent, 'registry') as mock_registry:
            mock_registry.get_toolset_for_tool.return_value = "web"
            assert agent._is_mcp_tool("web_search") is False

            mock_registry.get_toolset_for_tool.return_value = "terminal"
            assert agent._is_mcp_tool("terminal") is False

            mock_registry.get_toolset_for_tool.return_value = "file"
            assert agent._is_mcp_tool("read_file") is False

    def test_unknown_toolset_returns_false(self):
        """Tools with no registered toolset return False."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        with patch.object(run_agent, 'registry') as mock_registry:
            mock_registry.get_toolset_for_tool.return_value = None
            assert agent._is_mcp_tool("unknown_tool") is False

    def test_registry_error_returns_false(self):
        """Registry exceptions are caught and return False gracefully."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        with patch.object(run_agent, 'registry') as mock_registry:
            mock_registry.get_toolset_for_tool.side_effect = RuntimeError("registry error")
            assert agent._is_mcp_tool("any_tool") is False


# ---------------------------------------------------------------------------------------
# _should_delegate_mcp_batch tests
# ---------------------------------------------------------------------------------------

class TestShouldDelegateMcpBatch:
    """Tests for _should_delegate_mcp_batch decision logic."""

    def _make_agent(self):
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = False
        return agent

    def test_no_delegation_when_fewer_than_3_mcp_calls(self):
        """2 MCP calls + non-MCP → no delegation."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("mcp_tavily_search", {"query": "a"}),
                _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
                _make_tool_call("read_file", {"path": "/tmp/x"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is False
            assert len(mcp) == 2
            assert len(non_mcp) == 1

    def test_delegation_when_3_or_more_mcp_calls_plus_non_mcp(self):
        """3 MCP calls + at least 1 non-MCP → delegation triggers."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("mcp_tavily_search", {"query": "a"}),
                _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
                _make_tool_call("mcp_exa_web_search", {"query": "c"}),
                _make_tool_call("read_file", {"path": "/tmp/x"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is True
            assert len(mcp) == 3
            assert len(non_mcp) == 1

    def test_no_delegation_when_pure_mcp_batch(self):
        """3+ MCP calls but NO non-MCP calls → no delegation (normal concurrent path)."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("mcp_tavily_search", {"query": "a"}),
                _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
                _make_tool_call("mcp_exa_web_search", {"query": "c"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is False
            assert len(mcp) == 3
            assert len(non_mcp) == 0

    def test_no_delegation_for_non_mcp_only_batch(self):
        """Non-MCP only → no delegation."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("read_file", {"path": "/tmp/x"}),
                _make_tool_call("terminal", {"command": "ls"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is False
            assert len(mcp) == 0
            assert len(non_mcp) == 2

    def test_threshold_is_3_exactly(self):
        """Exactly 3 MCP calls with non-MCP → triggers delegation."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("mcp_tool_a", {}),
                _make_tool_call("mcp_tool_b", {}),
                _make_tool_call("mcp_tool_c", {}),
                _make_tool_call("read_file", {"path": "/tmp/x"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is True
            assert len(mcp) == 3

    def test_4_mcp_calls_with_non_mcp_also_triggers(self):
        """4 MCP calls + non-MCP → also triggers."""
        agent = self._make_agent()
        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            calls = [
                _make_tool_call("mcp_a", {}),
                _make_tool_call("mcp_b", {}),
                _make_tool_call("mcp_c", {}),
                _make_tool_call("mcp_d", {}),
                _make_tool_call("terminal", {"command": "ls"}),
            ]
            should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
            assert should_delegate is True
            assert len(mcp) == 4


# ---------------------------------------------------------------------------------------
# Integration: full _execute_tool_calls flow
# ---------------------------------------------------------------------------------------

class TestExecuteToolCallsAutoDelegation:
    """End-to-end test of _execute_tool_calls with mocked MCP delegation."""

    def _make_agent(self):
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = False
            agent._executing_tools = False
        return agent

    def test_delegation_path_taken_for_3_mcp_plus_non_mcp(self):
        """When 3+ MCP + non-MCP, _delegate_mcp_calls is invoked."""
        agent = self._make_agent()
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
            _make_tool_call("mcp_exa_web_search", {"query": "c"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
                with patch.object(agent, '_execute_tool_calls_impl') as mock_impl:
                    mock_impl.return_value = None
                    agent._execute_tool_calls(msg, [], "test-task")
                    # Non-MCP calls should go through normal path
                    mock_impl.assert_called_once()
                    # MCP calls should be delegated
                    mock_delegate.assert_called_once()

    def test_normal_path_when_fewer_than_3_mcp(self):
        """When <3 MCP calls, normal execution path is taken."""
        agent = self._make_agent()
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
                with patch.object(agent, '_execute_tool_calls_impl') as mock_impl:
                    mock_impl.return_value = None
                    agent._execute_tool_calls(msg, [], "test-task")
                    # Should NOT delegate
                    mock_delegate.assert_not_called()
                    # Should go through normal impl
                    mock_impl.assert_called_once()

    def test_normal_path_for_pure_mcp_batch(self):
        """Pure MCP batch (no non-MCP) → normal execution, no delegation."""
        agent = self._make_agent()
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
            _make_tool_call("mcp_exa_web_search", {"query": "c"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
                with patch.object(agent, '_execute_tool_calls_impl') as mock_impl:
                    mock_impl.return_value = None
                    agent._execute_tool_calls(msg, [], "test-task")
                    mock_delegate.assert_not_called()
                    mock_impl.assert_called_once()

    def test_quiet_mode_no_print(self):
        """In quiet_mode, the auto-delegation log message is suppressed."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
            _make_tool_call("mcp_exa_web_search", {"query": "c"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_is_mcp_tool', side_effect=lambda n: n.startswith("mcp_")):
            with patch.object(agent, '_delegate_mcp_calls'):
                with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                    # Should not raise or print
                    agent._execute_tool_calls(msg, [], "test-task")


# ---------------------------------------------------------------------------------------
# _delegate_mcp_calls result parsing
# ---------------------------------------------------------------------------------------

class TestDelegateMcpCallsResultParsing:
    """Tests that _delegate_mcp_calls correctly parses delegate_task results."""

    def _make_agent(self):
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = False
        return agent

    def test_result_dict_with_results_list(self):
        """Result as {results: [...]} is parsed correctly."""
        agent = self._make_agent()
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "b"}),
        ]
        messages = []

        delegate_result = {
            "results": [
                {"result": "search result 1"},
                {"result": "scrape result 2"},
            ]
        }

        with patch.object(agent, '_is_mcp_tool', return_value=True):
            with patch.object(run_agent, 'registry') as mock_registry:
                mock_registry.get_toolset_for_tool.return_value = "mcp-tavily"
                with patch('tools.delegate_tool.delegate_task', return_value=delegate_result):
                    agent._delegate_mcp_calls(calls, messages, "test-task")

        assert len(messages) == 2
        assert messages[0]["content"] == "search result 1"
        assert messages[0]["tool_call_id"] == "call_mcp_tavily_search"
        assert messages[1]["content"] == "scrape result 2"
        assert messages[1]["tool_call_id"] == "call_mcp_firecrawl_scrape"

    def test_fallback_when_result_not_dict(self):
        """Non-dict result is used as fallback for all calls."""
        agent = self._make_agent()
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "a"}),
        ]
        messages = []

        with patch.object(agent, '_is_mcp_tool', return_value=True):
            with patch.object(run_agent, 'registry') as mock_registry:
                mock_registry.get_toolset_for_tool.return_value = "mcp-tavily"
                with patch('tools.delegate_tool.delegate_task', return_value="error: something went wrong"):
                    agent._delegate_mcp_calls(calls, messages, "test-task")

        assert len(messages) == 1
        assert messages[0]["content"] == "error: something went wrong"
