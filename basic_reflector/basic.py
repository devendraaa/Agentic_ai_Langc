import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# To install: pip install tavily-python
from langchain_tavily import TavilySearch
#stategraph
from langgraph.graph import StateGraph, START, END, MessageGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
## custom tool function
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
import os 
from dotenv import load_dotenv
from chains import generation_chain, reflection_chain
load_dotenv()

tavily_search = TavilySearch(max_result=2)
tavily_search.invoke("what is langgraph")

