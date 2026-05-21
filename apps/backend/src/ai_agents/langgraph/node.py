from ai_agents.langgraph.react import ReActAgent
from langchain.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode


def run_agent_reasoning(state: MessagesState):
  """
  Run the agent reasoning node.
  """

  SYSTEM_PROMPT = """
You are a helpful assistant that can use tool to answer questions.
"""
  system_message = SystemMessage(SYSTEM_PROMPT)
  react_agent = ReActAgent()
  model = react_agent.get_model()

  res = model.invoke([system_message, *state['messages']])
  return {'messages': [res]}


tool_node = ToolNode(ReActAgent().get_tools())
