#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes

# 1. Chain definition

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a expert of big cats who ONLY selects one item from this comma separated list:
LIST:["tiger", "lion" , "leopard" , "snow leopard" , "jaguar"]
A user will pass you a query expressed in natural language, and you must select one cat from LIST.
ONLY return a string from LIST, no additional text."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
category_chain = chat_prompt | ChatOpenAI(openai_api_key="") | CommaSeparatedListOutputParser()

# 2. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# 3. Adding chain route
add_routes(
    app,
    category_chain,
    path="/category_chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)