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

class State(TypedDict):
    meals: str
    sections: List[Dish] # contains structure dish object with the name, 
                         #ingredients, and instructions for each dish from the orchestration.
    complete_menu : Annotated[List[str], operator.add] # merge results from parallel workers via operator.add.
    final_meal_guide: str # holds the synthesized final meal guide after all dishes have been processed and combined.

class WorkerState(TypedDict):
    section: Dish # contains structure dish object with the name, ingredients, and instructions for each dish from the orchestration.
    complete_menu : Annotated[list, operator.add]

class Dish(TypedDict):
    name: str = Field(
        description="The name of the dish.(e.g., 'Spaghetti Bolognese', 'Chicken Curry', 'Vegetable Stir Fry')")
    ingredients: str = Field(
        description="The ingredients required for the dish.(e.g., '200g spaghetti, 100g minced beef, 1 onion, 2 cloves garlic, 400g canned tomatoes, salt, pepper')")
    location: str = Field(
        description="The cuisine or origin of the dish.(e.g., 'Italian', 'Indian', 'Chinese')")

#Dishes schema for a list of dish objects, each containing the name, ingredients, and instructions for a specific dish.
class Dishes(TypedDict):
    sections: List[Dish] = Field(
        description="A list of dish objects, each containing the name, ingredients, and instructions for a specific dish."
    )

dish_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an assistant that helps users create a meal guide based on their provided meals. and structure grocery list."
         " You will receive a list of meals from the user, and your task is to generate a structured meal {meals} \n\n"
         "guide that includes the name of the dish,a-comma seperated list of ingredients, and the cuisine or culture origin of the food. "
         "Please ensure that the meal guide is clear, organized, and easy to follow."
        )
    ]
)

planner_pipe = dish_prompt | llm.with_structured_output(Dishes)

def orchestrate(state: State) -> State:
    """
    Orchestrates the meal guide creation process by taking a list of meals from the user, generating a structured meal guide.
    """

    # use the planner_pipe to generate a structured meal guide based on the provided meals
    dish_descriptions = planner_pipe.invoke({"meals": state["meals"]})

    # return the list of dish sections to be processed in parallel by the worker nodes
    return {
        "sections": dish_descriptions.sections
    }

def assign_workers(state: State):
    """
    Assigns each dish section to a worker node for parallel processing.
    Each worker will receive a single dish section to process.
    """
    return [Send("chef_worker", {"section": s}) for s in state["sections"]]

chef_prompt = ChatPromptTemplate.from_messages([
    "system",
    "You are a world class chief from {location}. \n\n"
    "Please introduce yourself briefly and present a details walkthrough for preparing the dish: {name}.\n"
    "Your response should include:\n"
    "start with hello with your name and background\n"
    "a clear list of preparation steps\n"
    "a full explanation of the cooking process\n\n"
    "use the following ingredents: {ingredents}."
])

chef_pipe = chef_prompt | llm

def chef_worker(state: WorkerState):
    meal_plan = chef_pipe.invoke({
        "name": state['section'].name,
        "location": state['sections'].location,
        "ingredients": state['section'].ingredients
    })

    return {"completed_menu": [meal_plan.content]}

def synthesizer(state: State):
    complete_sections = state["completed_menu"]
    complete_menu = "\n\n--\n\n".join(complete_sections)
    return {"final_meal_guide": complete_menu}

orchestrate_worker_buider = StateGraph(State)

orchestrate_worker_buider.add_node("orchestrator", orchestrate)
orchestrate_worker_buider.add_node("chef_worker", chef_worker)
orchestrate_worker_buider.add_node("synthesizer", synthesizer)

orchestrate_worker_buider.add_conditional_edges("orchestrator", assign_workers, ["chef_worker"])

orchestrate_worker_buider.add_edge("chef_worker", "synthesizer")
orchestrate_worker_buider.add_edge(START, "orchestrator")
orchestrate_worker_buider.add_edge("synthesizer", END)
orchestrate_worker = orchestrate_worker_buider.compile()

state = orchestrate_worker.invoke({"meal":"Please prepare Italian pasta, Mexcian tacos, Indian cury Thai stir-fry, and American burger"})
