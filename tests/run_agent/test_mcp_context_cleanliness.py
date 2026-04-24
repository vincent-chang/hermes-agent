"""Tests verifying main agent context cleanliness when MCP calls are delegated.

These tests verify that:
1. When 3+ MCP calls are delegated, raw MCP JSON results stay in subagent context
2. Main agent context only receives summarized/formatted results
3. Token savings are realized in the main context
4. The delegation threshold logic keeps main context clean for heavy MCP workloads

Run:
    pytest tests/run_agent/test_mcp_context_cleanliness.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

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
    """Simulates an MCP server that registers tools in the registry."""

    def __init__(self, name: str, tools: list):
        self.name = name
        self.tools = tools
        self.toolset_name = f"mcp-{name}"

    def register(self):
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
        def handler(args, task_id=None):
            # Simulate a realistic large MCP response (search results, scraped content, etc.)
            return json.dumps({
                "status": "ok",
                "tool": tool_name,
                "args": args,
                "data": {
                    "results": [
                        {"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Content snippet {i} " * 50}
                        for i in range(10)
                    ],
                    "total_results": 1000,
                    "raw_response": "X" * 500,  # Simulate raw JSON bloat
                }
            })
        return handler

    def unregister(self):
        for tool in self.tools:
            registry._tools.pop(tool["name"], None)


@pytest.fixture
def fake_mcp_tavily():
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
    server = FakeMCPServer("firecrawl", [
        {"name": "mcp_firecrawl_scrape", "description": "Scrape a URL"},
        {"name": "mcp_firecrawl_crawl", "description": "Crawl a website"},
    ])
    server.register()
    yield server
    server.unregister()


@pytest.fixture
def fake_mcp_exa():
    server = FakeMCPServer("exa", [
        {"name": "mcp_exa_web_search", "description": "Search the web with Exa"},
        {"name": "mcp_exa_findSimilar", "description": "Find similar pages"},
    ])
    server.register()
    yield server
    server.unregister()


# ---------------------------------------------------------------------------------------
# Test 1: Raw MCP results do NOT appear in parent messages list
# ---------------------------------------------------------------------------------------

class TestContextCleanliness:
    """Verify that raw MCP JSON responses don't pollute the main context."""

    def test_raw_mcp_results_stay_in_subagent(self, fake_mcp_tavily):
        """When MCP is delegated, the main context gets summaries, not raw JSON."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        # 3 MCP calls + 1 non-MCP → triggers delegation
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI news"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://news.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "GPT-5"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)
        messages = []

        # Simulate what a subagent would return (summarized results)
        def mock_delegate(goals, tasks, max_iterations):
            return {
                "results": [
                    {"result": "Summary: Found 1000 AI news articles about GPT-5..."},
                    {"result": "Summary: Extracted content from news.com showing..."},
                    {"result": "Summary: Deep search revealed 50 GPT-5 related papers..."},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, messages, "test-task")

        # Check that main context got the SUMMARIZED results, not raw MCP JSON
        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3  # 3 delegated MCP calls

        for tr in tool_results:
            # Raw MCP responses are large with "raw_response", "results", etc.
            # Summaries should be concise
            assert "raw_response" not in tr["content"]
            assert "status" not in tr["content"] or "ok" in tr["content"]
            # Summaries should NOT contain the bloated data structure
            assert "total_results" not in tr["content"]
            assert len(tr["content"]) < 500  # Summary should be concise

    def test_direct_mcp_execution_pollutes_context(self, fake_mcp_tavily):
        """When MCP is NOT delegated (< 3 calls), raw results DO enter context."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        # 2 MCP calls → NO delegation, normal execution
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
        ]
        msg = _MockAssistantMessage(calls)

        # Mock the actual MCP handler to return large JSON
        with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
            mock_delegate.return_value = None  # Should NOT be called
            with patch.object(agent, '_execute_tool_calls_impl') as mock_impl:
                def capture_and_return(inner_msg, messages, task_id, api_count=0):
                    # Simulate what _execute_tool_calls_impl does for MCP calls
                    for tc in inner_msg.tool_calls:
                        if tc.function.name.startswith("mcp_"):
                            # Raw MCP response (large JSON)
                            raw_response = json.dumps({
                                "results": [{"title": f"R{i}", "snippet": "X" * 100} for i in range(10)],
                                "total_results": 1000,
                                "raw_response": "X" * 500,
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": raw_response
                            })
                    return None
                mock_impl.side_effect = capture_and_return

                messages = []
                agent._execute_tool_calls(msg, messages, "test-task")

                # Verify delegation was NOT called
                mock_delegate.assert_not_called()

                # Verify raw MCP results ARE in context (pollution!)
                tool_results = [m for m in messages if m["role"] == "tool"]
                assert len(tool_results) == 2
                # These contain the full raw JSON
                assert "raw_response" in tool_results[0]["content"]
                assert "total_results" in tool_results[0]["content"]


# ---------------------------------------------------------------------------------------
# Test 2: Token estimation shows context savings
# ---------------------------------------------------------------------------------------

class TestTokenSavings:
    """Estimate token savings from delegation."""

    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token."""
        return len(text) // 4

    def test_delegation_saves_tokens(self, fake_mcp_tavily):
        """Delegating 3 MCP calls saves tokens in main context."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        # 3 MCP calls + 1 non-MCP
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        # Each raw MCP response would be ~2000 chars
        raw_response_per_call = json.dumps({
            "results": [{"title": f"R{i}", "snippet": "X" * 100} for i in range(10)],
            "total_results": 1000,
            "raw_response": "X" * 500,
        })

        # Summarized response would be ~200 chars
        summary_per_call = "Summary: Found 1000 results about AI/ML topics..."

        def mock_delegate(goals, tasks, max_iterations):
            # Subagent returns summaries instead of raw JSON
            return {
                "results": [
                    {"result": summary_per_call},
                    {"result": summary_per_call},
                    {"result": summary_per_call},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                messages = []
                agent._execute_tool_calls(msg, messages, "test-task")

        tool_results = [m for m in messages if m["role"] == "tool"]

        # With delegation: 3 summaries (~200 chars each) = ~600 chars total
        # Without delegation: 3 raw responses (~2000 chars each) = ~6000 chars total
        total_chars_with_delegation = sum(len(tr["content"]) for tr in tool_results)
        total_chars_without_delegation = len(raw_response_per_call) * 3

        # Delegation should reduce content by ~90%
        assert total_chars_with_delegation < total_chars_without_delegation // 2
        assert total_chars_with_delegation < 1000  # ~200 chars per summary


# ---------------------------------------------------------------------------------------
# Test 3: Realistic web research scenario
# ---------------------------------------------------------------------------------------

class TestRealisticResearchScenario:
    """Simulate a typical web research task with multiple MCP calls."""

    def test_research_task_with_delegation(self, fake_mcp_tavily, fake_mcp_firecrawl, fake_mcp_exa):
        """Typical research: 3+ searches + file write → should delegate."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        # Realistic research scenario:
        # - 3 web searches (MCP)
        # - 1 file write (non-MCP)
        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "latest AI research 2025"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "https://arxiv.org/list/cs.AI"}),
            _make_tool_call("mcp_exa_web_search", {"query": "GPT-5 release date"}),
            _make_tool_call("write_file", {"path": "/tmp/research.md", "content": "# Research"}),
        ]
        msg = _MockAssistantMessage(calls)

        delegation_called = []

        def mock_delegate(goals, tasks, max_iterations):
            delegation_called.append(True)
            return {
                "results": [
                    {"result": "Research summary 1: Found 500 papers on AI research..."},
                    {"result": "Research summary 2: arxiv page shows latest submissions..."},
                    {"result": "Research summary 3: GPT-5 rumored for Q3 2025..."},
                ]
            }

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None) as mock_impl:
                def record_non_mcp(inner_msg, messages, task_id, api_count=0):
                    # Verify non-MCP was executed
                    for tc in inner_msg.tool_calls:
                        assert tc.function.name == "write_file"
                    return None
                mock_impl.side_effect = record_non_mcp

                messages = []
                agent._execute_tool_calls(msg, messages, "test-task")

        # Verify delegation WAS triggered
        assert len(delegation_called) == 1

        # Verify non-MCP was executed first
        mock_impl.assert_called_once()

        # Verify results are summaries
        tool_results = [m for m in messages if m["role"] == "tool"]
        assert len(tool_results) == 3  # Only MCP results, non-MCP was handled separately


# ---------------------------------------------------------------------------------------
# Test 4: Boundary conditions for context cleanliness
# ---------------------------------------------------------------------------------------

class TestBoundaryConditions:
    """Test edge cases around the delegation threshold."""

    def test_2_mcp_plus_1_non_mcp_no_delegation(self, fake_mcp_tavily):
        """2 MCP + non-MCP → NO delegation → context gets raw results."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

            # Should NOT delegate - only 2 MCP calls
            mock_delegate.assert_not_called()

    def test_3_mcp_only_no_delegation(self, fake_mcp_tavily):
        """3 MCP but NO non-MCP → normal concurrent execution, no delegation."""
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

        with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

            # Should NOT delegate - pure MCP batch uses concurrent execution
            mock_delegate.assert_not_called()

    def test_3_mcp_plus_1_non_mcp_delegates(self, fake_mcp_tavily):
        """3 MCP + non-MCP → DELEGATES to keep context clean."""
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

        with patch('tools.delegate_tool.delegate_task') as mock_delegate:
            mock_delegate.return_value = {"results": [{"result": "s"} for _ in range(3)]}
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

            # SHOULD delegate - 3 MCP + non-MCP meets threshold
            mock_delegate.assert_called_once()


# ---------------------------------------------------------------------------------------
# Test 5: Verify toolsets passed to subagent for MCP access
# ---------------------------------------------------------------------------------------

class TestToolsetsForSubagent:
    """Verify subagent gets correct toolsets for MCP access."""

    def test_delegate_receives_correct_toolsets(self, fake_mcp_tavily, fake_mcp_firecrawl, fake_mcp_exa):
        """Subagent must receive all MCP toolsets to have MCP access."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "https://x.com"}),
            _make_tool_call("mcp_exa_web_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        captured_call = {}

        def mock_delegate(goals, tasks, max_iterations):
            # Capture the goals to verify toolsets
            captured_call["goals"] = goals
            return {"results": [{"result": "ok"} for _ in range(3)]}

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

        assert "goals" in captured_call
        goal = captured_call["goals"][0]

        # Critical: toolsets must include MCP toolsets for subagent to access them
        assert "toolsets" in goal
        toolsets = goal["toolsets"]

        # All MCP toolsets should be present
        assert "mcp-tavily" in toolsets, f"mcp-tavily not in {toolsets}"
        assert "mcp-firecrawl" in toolsets, f"mcp-firecrawl not in {toolsets}"
        assert "mcp-exa" in toolsets, f"mcp-exa not in {toolsets}"

        # web toolset should also be included
        assert "web" in toolsets


# ---------------------------------------------------------------------------------------
# Test 6: Multiple rounds of delegation in single conversation
# ---------------------------------------------------------------------------------------

class TestMultipleDelegationRounds:
    """Simulate multiple delegation turns in one conversation."""

    def test_multiple_delegation_turns(self, fake_mcp_tavily):
        """Model can trigger delegation multiple times across turns."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        delegation_count = 0

        def mock_delegate(goals, tasks, max_iterations):
            nonlocal delegation_count
            delegation_count += 1
            return {"results": [{"result": f"summary_{delegation_count}"} for _ in range(3)]}

        # First turn: 3 MCP + non-MCP
        calls_1 = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://x.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "ML"}),
            _make_tool_call("read_file", {"path": "/tmp/1"}),
        ]
        msg_1 = _MockAssistantMessage(calls_1)

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                messages = []
                agent._execute_tool_calls(msg_1, messages, "test-task")

        assert delegation_count == 1

        # Second turn: another 3 MCP + non-MCP
        calls_2 = [
            _make_tool_call("mcp_tavily_search", {"query": "GPT"}),
            _make_tool_call("mcp_tavily_extract", {"url": "https://y.com"}),
            _make_tool_call("mcp_tavily_deep_search", {"query": "LLM"}),
            _make_tool_call("read_file", {"path": "/tmp/2"}),
        ]
        msg_2 = _MockAssistantMessage(calls_2)

        with patch('tools.delegate_tool.delegate_task', mock_delegate):
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg_2, messages, "test-task")

        # Should have delegated again
        assert delegation_count == 2


# ---------------------------------------------------------------------------------------
# Test 7: Cross-server MCP counting
# ---------------------------------------------------------------------------------------

class TestCrossServerCounting:
    """Verify MCP calls from different servers count together."""

    def test_3_different_server_mcp_triggers(self, fake_mcp_tavily, fake_mcp_firecrawl, fake_mcp_exa):
        """3 MCP from different servers + non-MCP → delegates."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "https://x.com"}),
            _make_tool_call("mcp_exa_web_search", {"query": "ML"}),
            _make_tool_call("terminal", {"command": "ls"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch('tools.delegate_tool.delegate_task') as mock_delegate:
            mock_delegate.return_value = {"results": [{"result": "ok"} for _ in range(3)]}
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

            # Should delegate - 3 MCP from different servers + non-MCP
            mock_delegate.assert_called_once()

    def test_2_different_server_mcp_no_trigger(self, fake_mcp_tavily, fake_mcp_firecrawl):
        """2 MCP from different servers + non-MCP → NO delegation."""
        with patch.object(run_agent.AIAgent, '__init__', lambda self, **k: None):
            agent = AIAgent.__new__(AIAgent)
            agent.quiet_mode = True
            agent._executing_tools = False

        calls = [
            _make_tool_call("mcp_tavily_search", {"query": "AI"}),
            _make_tool_call("mcp_firecrawl_scrape", {"url": "https://x.com"}),
            _make_tool_call("read_file", {"path": "/tmp/x"}),
        ]
        msg = _MockAssistantMessage(calls)

        with patch.object(agent, '_delegate_mcp_calls') as mock_delegate:
            with patch.object(agent, '_execute_tool_calls_impl', return_value=None):
                agent._execute_tool_calls(msg, [], "test-task")

            # Should NOT delegate - only 2 MCP calls (threshold is 3)
            mock_delegate.assert_not_called()
