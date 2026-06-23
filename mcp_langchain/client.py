from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
            },
            "weather":{
                "url": "http://127.0.0.1:8000/mcp",
                "transport":"streamable_http",
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model = ChatGroq(
    model="llama-3.3-70b-versatile"
    )
    agent=create_agent(
        model, tools
    )

    math_response = await agent.ainvoke(
        {"messages":[{"role":"user", "content":"what is (3+5)*12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

asyncio.run(main())