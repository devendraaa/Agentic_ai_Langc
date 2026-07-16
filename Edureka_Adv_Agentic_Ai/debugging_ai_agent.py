# Demonstration 2.1: Building an Intelligent Tool Router

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from dotenv import load_dotenv

# STEP 1: Configure Gemini (Google Generative AI)

load_dotenv()

# Read API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-flash"
    temperature=0,
)

# STEP 2: Define DevOps Tools using @tool decorator

@tool
def KubernetesDiagnostics(query: str) -> str:
    """Use for Kubernetes cluster issues, pod failures, container crashes, deployment problems, resource constraints."""
    # Simulated K8s diagnostics
    return (
        f"K8s Diagnostic: Checking pod status, resource limits, and node health for "
        f"'{query}'. Status: 3 pods in CrashLoopBackOff, memory pressure detected on node-2."
    )

@tool
def DatabaseAnalyzer(query: str) -> str:
    """Use for database performance issues, slow queries, connection problems, query optimization, indexing."""
    # Simulated DB analysis
    return (
        f"DB Analysis for '{query}': Query execution time: 2.3s. Recommendation: "
        f"Add index on user_id column. Current load: 78% CPU, 12K active connections."
    )

@tool
def CICDPipeline(query: str) -> str:
    """Use for CI/CD pipeline status, build failures, deployment issues, test failures, pipeline configuration."""
    # Simulated pipeline info
    return (
        f"Pipeline Status for '{query}': Build #453 failed at test stage. Error: "
        f"Unit test 'test_auth_flow' timeout. Last successful deploy: 2 hours ago (commit: a7f3c2d)."
    )

@tool
def LogAggregator(query: str) -> str:
    """Use for searching application logs, error tracking, log pattern analysis, debugging specific errors."""
    # Simulated log search
    return (
        f"Log Search for '{query}': Found 247 error entries in last 1h. Top error: "
        f"Connection timeout to payment-service (156 occurrences). Spike detected at 14:23 UTC."
    )

@tool
def InfrastructureMetrics(query: str) -> str:
    """Use for system metrics, performance monitoring, resource utilization, alerts, infrastructure health."""
    # Simulated metrics
    return (
        f"Metrics for '{query}': CPU: 87% (threshold: 80%), Memory: 65%, Disk I/O: 450 IOPS. "
        f"Alert: API gateway response time increased 40% in last 15 min."
    )

# Collect tools into a list
tools = [
    KubernetesDiagnostics,
    DatabaseAnalyzer,
    CICDPipeline,
    LogAggregator,
    InfrastructureMetrics,
]

# STEP 3: Build Routing Prompt and Chain

# Build human-readable tool descriptions
tool_descriptions = "\n".join(
    [f"- {tool.name}: {tool.description}" for tool in tools]
)

routing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent DevOps tool router. Analyze the technical query and select the MOST appropriate tool.

Available tools:
{tool_descriptions}

Respond with ONLY the exact tool name. Choose one: KubernetesDiagnostics, DatabaseAnalyzer, CICDPipeline, LogAggregator, InfrastructureMetrics.
"""
        ),
        ("human", "{query}"),
    ]
)

# Chain: prompt → LLM → plain string output
routing_chain = routing_prompt | llm | StrOutputParser()

# STEP 4: Router Class

class DevOpsToolRouter:
    def __init__(self, tools, routing_chain):
        # Map tool name → tool object
        self.tools = {tool.name: tool for tool in tools}
        self.routing_chain = routing_chain

    def route_and_execute(self, query: str):
        print(f"\nINCIDENT: {query}")
        print("=" * 80)

        # Step 1: Ask LLM which tool to use
        selected_tool = self.routing_chain.invoke(
            {
                "tool_descriptions": tool_descriptions,
                "query": query,
            }
        ).strip()

        # Clean formatting issues
        selected_tool = selected_tool.replace("`", "").strip().strip('"').strip("'")

        print(f"Routing Decision: {selected_tool}")

        # Step 2: Execute selected tool
        result = None
        if selected_tool in self.tools:
            tool = self.tools[selected_tool]
            print(f"Executing: {tool.name}...")
            result = tool.invoke(query)
            print(f"Output:\n{result}")
        else:
            print(f"Error: Tool '{selected_tool}' not found in registry")

        return result

# STEP 5: Run Demo Scenarios

if __name__ == "__main__":
    router = DevOpsToolRouter(tools, routing_chain)

    print("ENTERPRISE DEVOPS ASSISTANT - INTELLIGENT TOOL ROUTING")
    print("=" * 80)

    # Scenario 1: Container orchestration issue
    router.route_and_execute(
        "Our production pods keep restarting in the payment service"
    )

    # Scenario 2: Database performance problem
    router.route_and_execute(
        "Users are reporting slow query performance on the orders table"
    )

    # Scenario 3: CI/CD failure
    router.route_and_execute(
        "The latest deployment failed in the staging environment"
    )

    # Scenario 4: Error investigation
    router.route_and_execute(
        "We're seeing 500 errors spike in the authentication service logs"
    )

    # Scenario 5: Infrastructure monitoring
    router.route_and_execute(
        "Check if our API servers are hitting resource limits"
    )

    print("\nDEMONSTRATION COMPLETE - All queries routed to appropriate tools")
    print("=" * 80)