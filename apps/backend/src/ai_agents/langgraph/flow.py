from ai_agents.langgraph.enums.flow_state import FlowStateEnum
from ai_agents.langgraph.node import run_agent_reasoning, tool_node
from langgraph.graph import MessagesState, StateGraph


def _should_continue(state: MessagesState) -> str:
  LAST = -1
  if state['messages'][LAST].tool_calls:
    return FlowStateEnum.ACT
  return FlowStateEnum.END


def get_flow() -> StateGraph[MessagesState]:
  flow = StateGraph(MessagesState)

  # Nodes

  flow.add_node(FlowStateEnum.AGENT_REASON, run_agent_reasoning)
  flow.add_node(FlowStateEnum.ACT, tool_node)

  # Edges

  flow.set_entry_point(FlowStateEnum.AGENT_REASON)
  flow.add_conditional_edges(
    FlowStateEnum.AGENT_REASON,
    _should_continue,
    {
      FlowStateEnum.ACT: FlowStateEnum.ACT,
      FlowStateEnum.END: FlowStateEnum.END,
    },
  )
  flow.add_edge(FlowStateEnum.ACT, FlowStateEnum.AGENT_REASON)

  return flow
