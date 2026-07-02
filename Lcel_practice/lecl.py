import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_groq import ChatGroq

def build_llm_runnable() -> RunnableLambda:

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    

    llm = ChatGroq(
    model="llama-3.3-70b-versatile"
   )
    
    def call_groq(prompt: str) -> str:
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.text
        
        return str(response)
    
    return RunnableLambda(call_groq)

def _normalize_prompt_value(value: str) -> str:

    if isinstance(value, tuple) and len(value) == 2:
        return str(value[1])
    return str(value)


NORMALIZE_PROMPT = RunnableLambda(_normalize_prompt_value)

def build_feature_design_chain(llm: RunnableLambda):
    """
    step 1: turn hight level idea into structure requirements.
    step 2: turn requirements into a technical spec (api + data + edge case).
    """

    requirements_prompt = PromptTemplate.from_template(
        (
            "you are a product minded engineer.\n"
            "convert the following feature idea into clear, structure requirements.\n"
            "feature idea: {feature_idea}\n"
            
        )
    )

    requirements_chain = (
        requirements_prompt | NORMALIZE_PROMPT | llm
    )

    # step 2: requirements to technical spec
    spec_prompt = PromptTemplate.from_template(
        (
            "you are a product minded engineer.\n"
            "based on the following requirements, generate a technical spec for the feature.\n"
            "requirements:\n {requirements} \n\n"
            "including section:\n"
            " high-level Architecture\n"
            " API design\n"
            " Data design\n"
            " Edge cases\n"
        )
    )

    multistep_chain = (
        {"feature_idea": RunnablePassthrough()} 
        | {"requirements": requirements_chain} 
        | spec_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    return multistep_chain

# example 2: Service analysis workflow (parallel branches)
def build_service_analysis_chain(llm: RunnableLambda):
    """
    Branching in Parallel: Service analysis workflow
        -summary: what it does and when to use it
        -risk: what are the risks and limitations
        -test: high level test plan
    """

    summary_prompt = PromptTemplate.from_template(
        (
            "you are a product minded engineer.\n"
            "analyze the following service and provide a detailed analysis.\n"
            "service description: {service_description}\n"
            
        )
    )

    risk_prompt = PromptTemplate.from_template(
        ( 
              "you are a product minded engineer.\n"
                "analyze the following service and provide a detailed risk analysis.\n"
                "service description: {service_description}\n"
                
          )
    )

    test_prompt = PromptTemplate.from_template(
        (
            "you are a product minded engineer.\n"
                "analyze the following service and provide a high level test plan.\n"
                "service description: {service_description}\n"
                
          )
    )

    summary_chain = (
        {"service": RunnablePassthrough()}
        | summary_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    risk_chain = (
        {"service": RunnablePassthrough()}
        | risk_prompt
        | NORMALIZE_PROMPT
        | llm
    )

    test_chain = (
        {"service": RunnablePassthrough()}
        | test_prompt
        | NORMALIZE_PROMPT
        | llm
    ) 

    # runnble map to run the three chains in parallel
    parallel_branches = RunnableMap(
        {
            "summary": summary_chain,
            "risk": risk_chain,
            "test": test_chain
        }
    )

    full_chain = parallel_branches

    return full_chain

def main():
    llm = build_llm_runnable()

    print("=== Feature Design Chain ===")
    print("demo 2 : design multi step lcel workflow")
    print("=" * 60)

    #feature design
    print("=== Feature Design Chain ===")
    feature_idea = input(
        "enter a feature idea to generate requirements and technical spec: "
    ).strip()

    if feature_idea:
        feature_design_chain = build_feature_design_chain(llm)
        result = feature_design_chain.invoke({"feature_idea": feature_idea})
        print("=== Feature Design Result ===")
        print(result)


    # example 2: service analysis
    print("=== Service Analysis Chain ===")
    service_description = input(
        "enter a service description to analyze: "
    ).strip()
    
    if service_description:
        service_analysis_chain = build_service_analysis_chain(llm)
        analysis = service_analysis_chain.invoke({"service_description": service_description})
        print("=== Service Analysis Result ===")

        print("\n[service summary]")
        print(analysis["summary"])

        print("\n[service risk analysis]")
        print(analysis["risk"])

        print("\n[service test plan]")
        print(analysis["test"])

if __name__ == "__main__":
    main()
        

    


