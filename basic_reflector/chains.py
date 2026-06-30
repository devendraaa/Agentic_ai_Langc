from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "you are a twitter influencer assistant tasked with writing excellent twitter posts."
         "Generating the best twitter post possible for the users request."
         "if the user provide critique, response with a review version of your previous attempts." 
        ),
         MessagesPlaceholder(variable_name="messages")
    ]
)

reflector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "your are a virtual twitter influencer grading a tweet. Generate critique and recommendations for the users tweet."
         "Always provide details recommedations, including request for length, virality, style, etc."),
         MessagesPlaceholder(variable_name="messages")
    ]
)



llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

generation_chain = generate_prompt | llm

reflection_chain = reflector_prompt | llm

