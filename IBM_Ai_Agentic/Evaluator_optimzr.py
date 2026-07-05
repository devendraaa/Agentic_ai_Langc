from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated, List
from langgraph.constants import Send
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from IPython.display import Image, display
import operator
from pprint import pprint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


grades = Literal[
    "ultra-conservative",
    "conservative",
    "moderate",
    "aggressive",
    "high-risk"
]

class state(TypedDict):
    investment_plan: str
    investor_profile: str
    target_grade: grades
    feedback: str
    grade: grades
    n: int

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an investment advisor. Given the investor's profile and their propossed plan,"
         "choose exactly one risk classification from: ultra-conservation, conservative, moderate, aggresive, high-risk."
         "return ONLY the grade"),
         ("user",
          "Investor profile: \n\n {investor_profile}\n\n")
    ]
)

grade_pipe = grade_prompt | llm

def determine_target_grade(state: state):
    """Ask the llm to pick the best-fitting target_grade"""
    response = grade_pipe.invoke({
        "investor_profile": state['investor_profile']
    })

    return {"target_grade": response.content.lower()}

cathie_wood_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    You are a bold, innovation-driven investment advisor inspired by cathie wood.
    Response with a concise investment plan in paragraph form.
        """),
    ("human", "Investor profile: \n\n{investor_profile}")
])

cathie_wood_pipe = cathie_wood_prompt | llm

class Feedback(BaseModel):
    grade: grades = Field(
        description="classify the investment based on risk level, ranging from ultra-conservative to high risk."
    )

    feedback: str = Field(
        description="Provide reasoning for the risk classification assigned to the investment suggestion."
    )

ray_dalio_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
            You are a investment advisor inspired by ray dalios principal but with adaptive  strategy generation.
            You goal is to create varied, scenario-aware investment plans that respond dynamically to economic conditions,
            feedback.    
            """),
            ("Human",
             """
            inverstor_profile:{investor_profile}
            Previous investment plan:{investment_plan}
            Previous strategy grade: {grade}
            evaluation feedback: {feedback}
            Based on.
             """)
        ]
    )

ray_dalio_pipe = ray_dalio_prompt | llm

def investment_plan_generator(state: state) -> dict:
    if state.get("feedback"):
        response = ray_dalio_pipe.invoke({
            "investor_profile": state["investor_profile"],
            "investment_plan": state['investment_plan'],
            "grade": state["grade"],
            "feedback": state['feedback']

        })
    else:
        response = cathie_wood_pipe.invoke({
            "investor_profile": state['investor_profile']
        })
    
    return {"investment_plan": response.content}

evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a investment risk evaluator inspired by warren buffett's value investment philosophy.
     return your assessment in the following format:
     {{
     "grade": "<investment risk level>",
     "feedback": "<concise explanation of the assigned risk level and key reasoning>"
     }}
    """),
    ("human",
     "Evaluate the invstment plan: \n\n {investment_plan}\n\n for the investor profile: "
     "\n\n{investor_profile}\n\n and provide feedback that matches this target risk level: {target_grade}")
])

buffet_eval_pipe = evaluate_prompt | llm.with_structured_output(Feedback)

def evaluate_plan(state: state):
     current_count = state.get('n', 0) + 1

     evaluate_result = buffet_eval_pipe.invoke({
         "investment_plan": state['investment_plan'],
         "investor_profile": state['investor_profile'],
         "target_grade": state['target_grade']
     })

     return {"grade": evaluate_result.grade, "feedback":evaluate_result.feedback, "n": current_count}

def route_investment(state: state, iteration_limit: int = 5):
    current_grade = state.get("grade", "MISSING")
    target_grade = state.get("target_grade", "MISSING")

    match = current_grade == target_grade

    if match: 
        return "Accepted"
    elif state['n'] > iteration_limit:
        return "Accepted"
    else:
        return "Rejected + Feedback"
    

optimizer_builder = StateGraph(state)

optimizer_builder.add_node("determine_target_grade", determine_target_grade)
optimizer_builder.add_node("investment_plan_generator", investment_plan_generator)
optimizer_builder.add_node("evaluate_plan", evaluate_plan)

optimizer_builder.add_edge(START, "determine_target_grade")
optimizer_builder.add_edge("determine_target_grade", "investment_plan_generator")
optimizer_builder.add_edge("investment_plan_generator", "evaluate_plan")

optimizer_builder.add_conditional_edges("evaluate_plan",
                                        lambda state: route_investment(state),
                                        {
                                            "Accepted" : END,
                                            "Rejected + Feedback" : "investment_plan_generator"
                                        },
                                        )
optimizer_workflow = optimizer_builder.compile()


State = optimizer_workflow.invoke({
    "investor_profile":(
        "Age : 30\n"
        "Salary: $110,000\n"
        "Assets: $40,000\n"
        "Goal: Achieve financial independent by age 45\n"
        "Risk tolerance: High"
    )
})
        