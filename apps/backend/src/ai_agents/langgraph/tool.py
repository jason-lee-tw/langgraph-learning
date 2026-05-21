from langchain.tools import tool


@tool
def triple(num: float) -> float:
  """
  Triple the input number.

  param num: a number to triple.

  returns: the triple of the input number.
  """

  return float(num) * 3
