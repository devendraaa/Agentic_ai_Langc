from Multi_hop_retrieval import graph

state = {

    "question":
        input("Question: "),

    "current_query":
        "",

    "documents":
        [],

    "answer":
        "",

    "retrieval_count":
        0,

    "max_hops":
        3,

    "enough_context":
        False,
}

state["current_query"] = state["question"]

result = graph.invoke(state)

print("\n\nAnswer:\n")

print(result["answer"])

