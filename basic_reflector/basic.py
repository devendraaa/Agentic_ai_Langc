from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage

from chains import generation_chain, reflection_chain


# ----------------------------
# State Definition
# ----------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ----------------------------
# Create Graph
# ----------------------------
graph = StateGraph(State)


# ----------------------------
# Node Names
# ----------------------------
GENERATE = "generate"
REFLECT = "reflect"


# ----------------------------
# Generate Tweet
# ----------------------------
def generate_node(state: State):

    response = generation_chain.invoke(
        {
            "messages": state["messages"]
        }
    )

    return {
        "messages": [response]
    }


# ----------------------------
# Reflect / Critique
# ----------------------------
def reflect_node(state: State):

    response = reflection_chain.invoke(
        {
            "messages": state["messages"]
        }
    )

    # Reflection becomes the next Human message
    return {
        "messages": [
            HumanMessage(content=response.content)
        ]
    }


# ----------------------------
# Stop Condition
# ----------------------------
def should_continue(state: State):

    # Stop after enough messages
    if len(state["messages"]) >= 6:
        return END

    return REFLECT


# ----------------------------
# Build Graph
# ----------------------------

graph.set_entry_point(GENERATE)

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.add_conditional_edges(
    GENERATE,
    should_continue
)

graph.add_edge(
    REFLECT,
    GENERATE
)


# ----------------------------
# Compile
# ----------------------------
app = graph.compile()


# ----------------------------
# Visualize Graph
# ----------------------------
print(app.get_graph().draw_mermaid())

try:
    app.get_graph().print_ascii()
except Exception:
    pass


# ----------------------------
# Run
# ----------------------------
response = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="AI agents are taking over content creation."
            )
        ]
    }
)


# ----------------------------
# Print Conversation
# ----------------------------
print("\n==============================")
print("Conversation")
print("==============================\n")

for message in response["messages"]:
    print(f"{type(message).__name__}:")
    print(message.content)
    print("-" * 80)