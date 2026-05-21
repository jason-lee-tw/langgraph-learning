from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.node import run_agent_reasoning, tool_node
from fastapi import HTTPException
from langchain_core.messages import (
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
)
from langgraph.graph import MessagesState, StateGraph
from modules.chat.dto.chat_dto import ChatHistoryDTO


def _convert_chat_list(chat_list: list[ChatHistoryDTO]) -> list[BaseMessage]:
  """Convert chat history list to proper message list for LLM"""
  result: list[BaseMessage] = []

  for chat in chat_list:
    role = chat.role
    content = chat.content
    message: BaseMessage

    if role == 'user':
      message = HumanMessage(content=content)
    elif role == 'system':
      message = SystemMessage(content=content)
    elif role == 'assistant':
      message = AIMessage(content=content)
    elif role == 'tool':
      message = ToolMessage(content=content)
    else:
      raise HTTPException(
        status_code=400, detail=f'Invalid role `{role}` in chat history.'
      )

    result.append(message)

  return result


def process_chat(chat_list: list[ChatHistoryDTO]) -> AIMessage:
  agent = BaseAgent()

  messages = _convert_chat_list(chat_list)
  agent_res = agent.chat(messages=messages)

  return agent_res


def process_chat_with_graph(chat_list: list[ChatHistoryDTO]):
  AGENT_REASON = 'agent_reason'
  ACT = 'act'
  # LAST = -1

  flow = StateGraph(MessagesState)

  flow.add_node(AGENT_REASON, run_agent_reasoning)
  flow.set_entry_point(AGENT_REASON)

  flow.add_node(ACT, tool_node)
