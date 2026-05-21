from ai_agents.base_agent import BaseAgent
from ai_agents.langgraph.tool import triple
from ai_agents.tavily.tavily_client import TavilyClient
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool


class ReActAgent:
  __model: BaseChatModel

  def __init__(self):
    tools = self.get_tools()

    agent = BaseAgent
    self.__model = agent.get_model()
    self.__model.bind_tools(tools=tools)

  def get_tavily_search_tool(self):
    tavily_search = TavilyClient().searcher
    tavily_search.max_results = 1

    return tavily_search

  def get_model(self) -> BaseChatModel:
    return self.__model

  def get_tools(self) -> list[BaseTool]:
    return [triple, self.get_tavily_search_tool()]
