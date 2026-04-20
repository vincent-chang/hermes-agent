"""Integration tests for MCP auto-delegation with mocked MCP servers.

These tests simulate MCP tools by:
1. Registering fake MCP-style tools directly in the registry
2. Testing the full _execute_tool_calls flow with real delegation

Run:
    pytest tests/run_agent/test_mcp_auto_delegation_integration.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call

import run_agent
from run_agent import AIAgent
from tools.registry import registry


# ---------------------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------------------

class FakeMCPServer:
    """Simulates an MCP server that registers tools in the registry."""

    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = tools
        self.toolset_name = f"mcp-{name}"

    def register(self):
        """Register all tools into the global registry (simulating MCP discovery)."""
        for tool in self.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", f"Fake MCP tool: {tool['name']}"),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            registry.register(
                name=tool["name"],
                toolset=self.toolset_name,
                schema=schema,
                handler=self._make_handler(tool["name"]),
                check_fn=None,
                is_async=False,
                description=tool.get("description", ""),
            )

    def _make_handler(self, tool_name: str):
        """Return a sync handler that returns a fake result."""
        def handler(args, task_id=None):
            return json.dumps({"status": "ok", "tool": tool_name, "args": args})
        return handler

    def unregister(self):
        """Remove all registered tools from the registry."""
        for tool in self.tools:
            registry._tools.pop(tool["name"], None)


@pytest.fixture
def fake_mcp_tavily():
    """Fake Tavily MCP server with 3 search tools."""
    server = FakeMCPServer("tavily", [
        {"name": "mcp_tavily_search", "description": "Search the web"},
        {"name": "mcp_tavily_extract", "description": "Extract content from URLs"},
        {"name": "mcp_tavily_deep_search", "description": "Deep web search"},
    ])
    server.register()
    yield server
    server.unregister()


@pytest.fixture
def fake_mcp_firecrawl():
    """Fake Firecrawl MCP server."""
    server = FakeMCPServer("firecrawl", [
        {"name": "mcp_firecrawl_scrape", "description": "Scrape a URL"},
        {"name": "mcp_firecrawl_crawl", "description": "Crawl a website"},
    ])
    server.register()
    yield server
    server.unregister()


@pytest.fixture
def fake_mcp_exa():
    """Fake Exa MCP server."""
    server = FakeMCPServer("exa", [
        {"name": "mcp_exa_web_search", "description": "Search the web with Exa"},
        {"name": "mcp_exa_findSimilar", "description": "Find similar pages"},
    ])
    server.register()
    yield server
    server.unregister()


def _make_tool_call(name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    tc.id = f"call_{name}"
    return tc


class _MockAssistantMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


# ---------------------------------------------------------------------------------------
# Test 1: Toolset detection via registry
# ---------------------------------------------------------------------------------------

class TestMcpToolsetDetection:
    """Verify _is_mcp_tool correctly identifies toolset via registry."""

    def test_fake_mcp_tool_is_detected(self, fake_mcp_tavily):
        """Registered MCP tool is identified as MCP by toolset prefix."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        assert agent._is_mcp_tool("mcp_tavily_search") is True
        toolset = registry.get_toolset_for_tool("mcp_tavily_search")
        assert toolset == "mcp-tavily"

    def test_regular_tool_not_mcp(self, fake_mcp_tavily):
        """Non-MCP tools are not flagged as MCP."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        # read_file is a built-in, not MCP
        assert agent._is_mcp_tool("read_file") is False

    def test_multiple_mcp_servers(self, fake_mcp_tavily, fake_mcp_firecrawl, fake_mcp_exa):
        """Tools from different MCP servers are all detected as MCP."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        assert agent._is_mcp_tool("mcp_tavily_search") is True
        assert agent._is_mcp_tool("mcp_firecrawl_scrape") is True
        assert agent._is_mcp_tool("mcp_exa_web_search") is True


# ---------------------------------------------------------------------------------------
# Test 2: Decision logic with fake MCP tools
# ---------------------------------------------------------------------------------------

class TestDelegationDecisionWithFakes:
    """Test _should_delegate_mcp_batch using fake MCP tools (no mocking)."""

    def test_3_mcp_plus_non_mcp_triggers(self, fake_mcp_tavily):
        """3 MCP + 1 non-MCP → delegation triggers (the main use case)."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI news"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://example.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "LLM developments"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is True
        assert len(mcp) == 3
        assert len(non_mcp) == 1

    def test_2_mcp_no_delegation(self, fake_mcp_tavily):
        """2 MCP → no delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://example.com"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is False

    def test_3_mcp_alone_no_delegation(self, fake_mcp_tavily):
        """3 MCP without non-MCP → no delegation (pure MCP batch)."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://example.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "LLM"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is False

    def test_cross_server_mcp_counting(self, fake_mcp_tavily, fake_mcp_firecrawl):
        """3 MCP calls from different servers count together."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "https://x.com"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://y.com"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is True
        assert len(mcp) == 3


# ---------------------------------------------------------------------------------------
# Test 3: Full delegation flow with real registry and mocked delegate_task
# ---------------------------------------------------------------------------------------

class TestFullDelegationFlow:
    """End-to-end test: _execute_tool_calls → _delegate_mcp_calls with real registry."""

    def test_delegation_receives_correct_toolsets(self, fake_mcp_tavily, fake_mcp_exa):
        """The delegate_task call includes the correct MCP toolsets."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://example.com"}),
            _make_tool_call("mcp_exa_web_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        captured_calls = {}

        def mock_delegate_task(goals, tasks, max_iterations):
            captured_calls["goals"] = goals
            captured_calls["tasks"] = tasks
            return {"results": [{"result": "ok"} for _ in goals[0] if "tool_name" in _]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate_task):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

        # Verify delegate_task was called with correct toolsets
        assert "goals" in captured_calls
        goal = captured_calls["goals"][0]
        assert "toolsets" in goal
        # Should include both mcp-tavily and mcp-exa
        assert "mcp-tavily" in goal["toolsets"]
        assert "mcp-exa" in goal["toolsets"]
        assert "web" in goal["toolsets"]

    def test_goal_text_contains_all_mcp_calls(self, fake_mcp_tavily):
        """The goal text passed to subagent lists all MCP calls."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI news"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://news.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "GPT-5"}),
            _make_tool_call("terminal", {"command": "ls"}),
        ]
        msg = _MockAssistantMessage(calls)

        captured_goal = {}

        def mock_delegate_task(goals, tasks, max_iterations):
            captured_goal["text"] = goals[0]["goal"]
            return {"results": [{"result": "done"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate_task):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

        # Goal text should mention all 3 MCP tools
        assert "mcp_tavily_search" in captured_goal["text"]
        assert "mcp_tavily_extract" in captured_goal["text"]
        assert "mcp_tavily_deep_search" in captured_goal["text"]

    def test_messages_appended_with_correct_tool_call_ids(self, fake_mcp_tavily):
        """Results are appended to messages with correct tool_call_id."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        def mock_delegate_task(goals, tasks, max_iterations):
            return {
                "results": [
                    {"result": "result_1"},
                    {"result": "result_2"},
                    {"result": "result_3"},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate_task):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        # Should have 3 tool results (one per delegated MCP call)
        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3
        assert tool_results[0]["tool_call_id"] == "call_mcp_tavily_search"
        assert tool_results[1]["tool_call_id"] == "call_mcp_tavily_extract"
        assert tool_results[2]["tool_call_id"] == "call_mcp_tavily_deep_search"
        assert tool_results[0]["content"] == "result_1"


# ---------------------------------------------------------------------------------------
# Test 4: Non-delegation paths still work
# ---------------------------------------------------------------------------------------

class TestNonDelegationPaths:
    """Verify normal execution paths are unaffected."""

    def test_2_mcp_calls_execute_normally(self, fake_mcp_tavily):
        """2 MCP calls → normal concurrent execution, no delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch('tools.delegate_tool.delegate_task') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None) as mock_impl:
                mock_impl.return_value = None
                agent._execute_tool_calls(msg, [], "test-task")

                mock_delegate.assert_not_called()
                mock_impl.assert_called_once()

    def test_non_mcp_only_calls_execute_normally(self):
        """Non-MCP calls → normal execution."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("read_file", {"path": "/tmp/a"}),
            _make_tool_call("terminal", {"command": "ls"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch('tools.delegate_tool.delegate_task') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None) as mock_impl:
                mock_impl.return_value = None
                agent._execute_tool_calls(msg, [], "test-task")

                mock_delegate.assert_not_called()
                mock_impl.assert_called_once()

    def test_pure_3_mcp_batch_no_delegation(self, fake_mcp_tavily):
        """3+ MCP without non-MCP → normal execution, no delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "ML"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch('tools.delegate_tool.delegate_task') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None) as mock_impl:
                mock_impl.return_value = None
                agent._execute_tool_calls(msg, [], "test-task")

                mock_delegate.assert_not_called()
                mock_impl.assert_called_once()


# ---------------------------------------------------------------------------------------
# Test 5: Threshold boundary cases
# ---------------------------------------------------------------------------------------

class TestThresholdBoundaries:
    """Edge cases around the delegation threshold."""

    def test_threshold_2_mcp_no_delegate(self, fake_mcp_tavily, fake_mcp_exa):
        """2 MCP from different servers → no delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_exa_web_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is False

    def test_threshold_3_mcp_delegates(self, fake_mcp_tavily, fake_mcp_exa):
        """3 MCP from mixed servers → delegates."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_exa_web_search", {"query": "ML"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        should_delegate, mcp, non_mcp = agent._should_delegate_mcp_batch(calls)
        assert should_delegate is True
        assert len(mcp) == 3
