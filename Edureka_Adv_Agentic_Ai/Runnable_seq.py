import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
    

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
   )

# STEP 1: Enrichment function (simulated external data)
def enrich_product_data(product_name: str) -> dict:
    """
    Simulate fetching extra product data from a catalog or external API.
    In a real system this could be a DB query or requests.get() call.
    """
    catalog = {
        "SmartDesk": {"category": "Office Furniture", "price": 499},
        "EcoBottle": {"category": "Sustainable Lifestyle", "price": 29},
        "NoiseBlock Pro": {"category": "Audio Equipment", "price": 199},
    }

    enrichment = catalog.get(
        product_name,
        {"category": "General", "price": 99},
    )

    enriched_data = {
        "product_name": product_name,
        "category": enrichment["category"],
        "price": enrichment["price"],
    }

    print("Enriched data:", enriched_data)
    return enriched_data


# STEP 2: Build prompt for Gemini
def build_prompt(enriched: dict) -> str:
    """
    Convert enriched data into a natural language prompt.
    """
    prompt = f"""
You are a product marketing copywriter.

Product name: {enriched['product_name']}
Category: {enriched['category']}
Price: ${enriched['price']}

Write a compelling 2–3 sentence product description that:
- Highlights the category and value
- Sounds friendly and professional
- Encourages the customer to buy
"""
    return prompt.strip()


# STEP 3: Wrap Gemini call in a function so it can be a Runnable
def llm_call(prompt: str) -> str:
    response = llm.invoke(prompt)
    # For text-only prompts, response.text is convenient
    return response.text

# Wrap functions as Runnables
enrichment_runnable = RunnableLambda(enrich_product_data)
prompt_runnable = RunnableLambda(build_prompt)
llm_runnable = RunnableLambda(llm_call)

# Build the RunnableSequence pipeline or chain:
#    input (str) -> enrich -> prompt -> llm -> output (str)
runnable_sequence: RunnableSequence = (
    enrichment_runnable
    | prompt_runnable
    | llm_runnable
)


if __name__ == "__main__":
    print("=== RunnableSequence with Data Enrichment (Gemini) ===")
    product_name = input(
        "Enter a product name (e.g., SmartDesk, EcoBottle, NoiseBlock Pro): "
    )

    result = runnable_sequence.invoke(product_name)

    print("\n--- Generated Marketing Description ---")
    print(result)
