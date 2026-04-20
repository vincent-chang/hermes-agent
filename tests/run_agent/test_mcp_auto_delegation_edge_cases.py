"""Edge case and behavior tests for MCP auto-delegation.

Tests:
- delegate_task result edge cases (fewer results than calls, exception)
- Print message behavior (quiet_mode suppression)
- Multiple delegation turns (model calls MCP delegation multiple times)
- Non-MCP tools executed alongside delegated MCP calls

Run:
    pytest tests/run_agent/test_mcp_auto_delegation_edge_cases.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

import run_agent
from run_agent import AIAgent
from tools.registry import registry


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.id = f"call_{name}"
    return tc


class _MockAssistantMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class FakeMCPServer:
    def __init__(self, name: str, tool_names: list):
        self.name = name
        self.toolset_name = f"mcp-{name}"
        self.tool_names = tool_names

    def register(self):
        for tname in self.tool_names:
            schema = {
                "type": "function",
                "function": {
                    "name": tname,
                    "description": f"Fake {tname}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            registry.register(
                name=tname,
                toolset=self.toolset_name,
                schema=schema,
                handler=lambda args, task_id=None: json.dumps({"ok": True, "tool": tname}),
                check_fn=None,
                is_async=False,
                description=f"Fake {tname}",
            )

    def unregister(self):
        """Remove all registered tools from the registry."""
        for tname in self.tool_names:
            registry._tools.pop(tname, None)


@pytest.fixture
def fake_mcp():
    server = FakeMCPServer("test", [
        "mcp_tool_a", "mcp_tool_b", "mcp_tool_c", "mcp_tool_d",
    ])
    server.register()
    yield server
    server.unregister()


# ---------------------------------------------------------------------------------------
# Test 1: delegate_task returns fewer results than calls
# ---------------------------------------------------------------------------------------

class TestFewerResultsThanCalls:
    """When delegate_task returns fewer results than MCP calls, we handle gracefully."""

    def test_fewer_results_uses_fallback(self, fake_mcp):
        """Fewer results than calls → missing ones get stringified full result."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        # delegate_task returns only 2 results for 3 MCP calls
        def mock_delegate(goals, tasks, max_iterations):
            return {
                "results": [
                    {"result": "result_a"},
                    {"result": "result_b"},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]
        # All 3 MCP calls should get a result (3rd gets full fallback)
        assert len(tool_results) == 3
        assert tool_results[0]["content"] == "result_a"
        assert tool_results[1]["content"] == "result_b"
        # 3rd gets the whole result dict as string
        assert "result_b" in tool_results[2]["content"] or tool_results[2]["content"]


# ---------------------------------------------------------------------------------------
# Test 2: delegate_task raises exception
# ---------------------------------------------------------------------------------------

class TestDelegateTaskException:
    """When delegate_task raises, error is captured in tool result."""

    def test_exception_in_delegate_task(self, fake_mcp):
        """Exception from delegate_tool is caught and returned as error."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        def mock_delegate(goals, tasks, max_iterations):
            raise RuntimeError("subagent failed")

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                # Should not raise — errors are captured
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3
        # All should contain the error
        for tr in tool_results:
            assert "RuntimeError" in tr["content"] or "subagent failed" in tr["content"]


# ---------------------------------------------------------------------------------------
# Test 3: delegate_task returns non-dict result
# ---------------------------------------------------------------------------------------

class TestNonDictDelegateResult:
    """When delegate_task returns a non-dict result (e.g., error string)."""

    def test_string_result_used_for_all(self, fake_mcp):
        """String result is used as-is for all MCP calls."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        def mock_delegate(goals, tasks, max_iterations):
            return "delegate unavailable"

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3
        for tr in tool_results:
            assert tr["content"] == "delegate unavailable"


# ---------------------------------------------------------------------------------------
# Test 4: Print message behavior
# ---------------------------------------------------------------------------------------

class TestQuietModePrint:
    """Verify quiet_mode suppresses the auto-delegation print."""

    def test_prints_when_not_quiet(self, fake_mcp):
        """Non-quiet mode prints the delegation message."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = False
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        def mock_delegate(goals, tasks, max_iterations):
            return {"results": [{"result": "ok"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                with patch('builtins.print') as mock_print:
                    agent._execute_tool_calls(msg, [], "test-task")
                    # Should print the auto-delegation message
                    assert mock_print.called
                    assert "Auto-delegating" in mock_print.call_args[0][0]

    def test_no_print_in_quiet_mode(self, fake_mcp):
        """quiet_mode suppresses the delegation print."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        def mock_delegate(goals, tasks, max_iterations):
            return {"results": [{"result": "ok"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                with patch('builtins.print') as mock_print:
                    agent._execute_tool_calls(msg, [], "test-task")
                    # Should NOT print
                    assert not mock_print.called


# ---------------------------------------------------------------------------------------
# Test 5: Non-MCP tools get executed before delegation
# ---------------------------------------------------------------------------------------

class TestNonMcpExecutedBeforeDelegation:
    """Non-MCP tools are executed via _execute_tool_calls_impl BEFORE delegation."""

    def test_non_mcp_executed_first(self, fake_mcp):
        """Non-MCP tools run through normal impl, THEN MCP delegated."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        execution_order = []

        def mock_impl(inner_msg, messages, task_id, api_count=0):
            # Record non-MCP execution
            for tc in inner_msg.tool_calls:
                execution_order.append(("impl", tc.function.name))
            return None

        def mock_delegate(goals, tasks, max_iterations):
            # Record delegation
            execution_order.append(("delegate", "mcp_calls"))
            return {"results": [{"result": "ok"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', mock_impl):
                agent._execute_tool_calls(msg, [], "test-task")

        # Non-MCP should execute first via impl
        assert execution_order[0] == ("impl", "read_file")
        # Then delegation happens
        assert ("delegate", "mcp_calls") in execution_order


# ---------------------------------------------------------------------------------------
# Test 6: Empty tool_calls list
# ---------------------------------------------------------------------------------------

class TestEmptyToolCalls:
    """Edge case: empty tool_calls list."""

    def test_empty_tool_calls_no_op(self):
        """Empty tool_calls → no delegation, no execution."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        msg = _MockAssistantMessage([])

        with patch('tools.delegate_tool.delegate_task') as mock_del:
            with patch.object(agent, '_execute_tool_calls_impl') as mock_impl:
                agent._execute_tool_calls(msg, [], "test-task")
                mock_del.assert_not_called()
                mock_impl.assert_not_called()


# ---------------------------------------------------------------------------------------
# Test 7: Multiple non-MCP tools alongside MCP
# ---------------------------------------------------------------------------------------

class TestMultipleNonMcpWithMcp:
    """Multiple non-MCP tools should all execute before MCP delegation."""

    def test_all_non_mcp_executed_before_delegation(self, fake_mcp):
        """All non-MCP calls run through impl before delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/a"}),
            _make_tool_call("terminal", {"command": "ls"}),
            _make_tool_call("write_file", {"path": "/tmp/b"}),
        ]
        msg = _MockAssistantMessage(calls)

        impl_calls = []

        def mock_impl(inner_msg, messages, task_id, api_count=0):
            for tc in inner_msg.tool_calls:
                impl_calls.append(tc.function.name)
            return None

        def mock_delegate(goals, tasks, max_iterations):
            return {"results": [{"result": "ok"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', mock_impl):
                agent._execute_tool_calls(msg, [], "test-task")

        # All 3 non-MCP should have been passed to impl
        assert "read_file" in impl_calls
        assert "terminal" in impl_calls
        assert "write_file" in impl_calls
        assert len(impl_calls) == 3


# ---------------------------------------------------------------------------------------
# Test 8: Real delegate_task result structure compatibility
# ---------------------------------------------------------------------------------------

class TestRealDelegateResultStructure:
    """Test against the actual delegate_task result structure from delegate_tool.py."""

    def test_handles_real_delegate_result_format(self, fake_mcp):
        """delegate_tool returns {results: [{result: ..., error: ...}, ...]}."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        # Real delegate_tool result format
        def mock_delegate(goals, tasks, max_iterations):
            return {
                "results": [
                    {"result": "search result for tool_a", "error": None},
                    {"result": "scrape result for tool_b", "error": None},
                    {"result": "extract result for tool_c", "error": None},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3
        assert tool_results[0]["content"] == "search result for tool_a"
        assert tool_results[1]["content"] == "scrape result for tool_b"
        assert tool_results[2]["content"] == "extract result for tool_c"

    def test_error_in_delegate_result(self, fake_mcp):
        """delegate_tool result with error field is handled."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tool_a", {}),
            _make_tool_call("mcp_tool_b", {}),
            _make_tool_call("mcp_tool_c", {}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        def mock_delegate(goals, tasks, max_iterations):
            return {
                "results": [
                    {"result": "ok", "error": None},
                    {"result": "", "error": "Connection timeout"},
                    {"result": "ok", "error": None},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3
        assert tool_results[0]["content"] == "ok"
        assert "Connection timeout" in tool_results[1]["content"]
        assert tool_results[2]["content"] == "ok"
