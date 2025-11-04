"""
LangGraph agent for querying Synthea Neo4j database with natural language.
"""
import json
from datetime import datetime
from typing import Annotated, TypedDict, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from llm_factory import create_llm, get_main_agent_model, get_router_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from neo4j_utils import Neo4jConnection
from cypher_subagent import query_cypher_subgraph, get_cypher_subgraph
from langsmith.run_helpers import traceable
from logging_utils import log
from pydantic import BaseModel, Field


# Initialize Neo4j connection
neo4j_conn = Neo4jConnection()
neo4j_conn.connect()


# Define tools for the agent
@tool
@traceable(name="get_patient_procedures", run_type="tool", tags=["tool", "patient", "procedures"])
def get_patient_procedures(patient_name: str, limit: int = 30) -> str:
    """
    Get procedures that have been performed on a patient, sorted by most recent first.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        A formatted string containing the list of procedures with dates and encounter information, limited to most recent entries.
    """
    log("TOOL", f"Called with patient_name='{patient_name}', limit={limit}", "get_patient_procedures")

    try:
        results = neo4j_conn.get_patient_procedures(patient_name, limit=limit)

        if not results:
            log("TOOL", f"No procedures found for '{patient_name}'", "get_patient_procedures")
            return f"No procedures found for patient '{patient_name}'. The patient may not exist or has no recorded procedures."

        # Only mention limit if we hit the limit (results count equals limit)
        if len(results) >= limit:
            output = f"Procedures for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
        else:
            output = f"Procedures for {results[0]['patient_name']}:\n\n"

        for i, record in enumerate(results, 1):
            output += f"{i}. {record['procedure_description']}\n"
            output += f"   Date: {record['encounter_date']}\n"
            output += f"   Encounter Type: {record['encounter_type']}\n\n"

        log("TOOL", f"Found {len(results)} procedures", "get_patient_procedures")
        return output
    except Exception as e:
        log("TOOL", f"Error - {str(e)}", "get_patient_procedures", level="error")
        return f"Error retrieving procedures: {str(e)}"


@tool
@traceable(name="get_patient_conditions", run_type="tool", tags=["tool", "patient", "conditions"])
def get_patient_conditions(patient_name: str, limit: int = 30) -> str:
    """
    Get medical conditions/diagnoses for a patient, sorted by most recent first.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        A formatted string containing the list of conditions with dates, limited to most recent entries.
    """
    try:
        results = neo4j_conn.get_patient_conditions(patient_name, limit=limit)

        if not results:
            return f"No conditions found for patient '{patient_name}'. The patient may not exist or has no recorded conditions."

        # Only mention limit if we hit the limit (results count equals limit)
        if len(results) >= limit:
            output = f"Conditions for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
        else:
            output = f"Conditions for {results[0]['patient_name']}:\n\n"

        for i, record in enumerate(results, 1):
            output += f"{i}. {record['condition_description']}\n"
            output += f"   Diagnosed: {record['encounter_date']}\n\n"

        return output
    except Exception as e:
        return f"Error retrieving conditions: {str(e)}"


@tool
@traceable(name="get_patient_medications", run_type="tool", tags=["tool", "patient", "medications"])
def get_patient_medications(patient_name: str, limit: int = 30) -> str:
    """
    Get medications prescribed to a patient, sorted by most recent first.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        A formatted string containing the list of medications with prescription dates, limited to most recent entries.
    """
    try:
        results = neo4j_conn.get_patient_medications(patient_name, limit=limit)

        if not results:
            return f"No medications found for patient '{patient_name}'. The patient may not exist or has no prescribed medications."

        # Only mention limit if we hit the limit (results count equals limit)
        if len(results) >= limit:
            output = f"Medications for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
        else:
            output = f"Medications for {results[0]['patient_name']}:\n\n"

        for i, record in enumerate(results, 1):
            output += f"{i}. {record['medication']}\n"
            output += f"   Prescribed: {record['prescribed_date']}\n\n"

        return output
    except Exception as e:
        return f"Error retrieving medications: {str(e)}"


@tool
@traceable(name="get_patient_encounters", run_type="tool", tags=["tool", "patient", "encounters"])
def get_patient_encounters(patient_name: str, limit: int = 30) -> str:
    """
    Get healthcare encounters for a patient, sorted by most recent first.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        A formatted string containing the list of encounters with dates and types, limited to most recent entries.
    """
    try:
        results = neo4j_conn.get_patient_encounters(patient_name, limit=limit)

        if not results:
            return f"No encounters found for patient '{patient_name}'. The patient may not exist."

        # Only mention limit if we hit the limit (results count equals limit)
        if len(results) >= limit:
            output = f"Encounters for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
        else:
            output = f"Encounters for {results[0]['patient_name']}:\n\n"

        for i, record in enumerate(results, 1):
            output += f"{i}. {record['encounter_type']}\n"
            output += f"   Date: {record['encounter_date']}\n"
            output += f"   Type: {', '.join(record['encounter_labels'])}\n\n"

        return output
    except Exception as e:
        return f"Error retrieving encounters: {str(e)}"


@tool
@traceable(name="search_patients", run_type="tool", tags=["tool", "patient", "search"])
def search_patients(search_term: str = None, limit: int = 30) -> str:
    """
    Search for patients by name, sorted by relevance. Use this when you need to find a patient or list patients.

    Args:
        search_term: Optional. Part of the patient's first or last name to search for. If not provided, returns recent patients.
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        A formatted string containing matching patients with their IDs and birth dates, sorted by name similarity.
    """
    try:
        results = neo4j_conn.search_patients(search_term, limit=limit)

        if not results:
            return f"No patients found matching '{search_term}'."

        if search_term:
            output = f"Found patients matching '{search_term}' (showing up to {limit} results, sorted by relevance):\n\n"
        else:
            output = f"Found patients (showing up to {limit} most recent):\n\n"

        for i, record in enumerate(results, 1):
            output += f"{i}. {record['patient_name']} (ID: {record['patient_id']})\n"
            output += f"   Birth Date: {record['birth_date']}\n\n"

        return output
    except Exception as e:
        return f"Error searching patients: {str(e)}"


@tool
@traceable(name="get_database_schema", run_type="tool", tags=["tool", "schema", "database"])
def get_database_schema() -> str:
    """
    Get information about the database schema and structure.
    Use this to understand how data is organized and what relationships exist.

    Returns:
        A description of the database schema including entities and relationships.
    """
    return neo4j_conn.get_database_schema()


@tool
@traceable(name="execute_custom_query", run_type="tool", tags=["tool", "custom", "cypher"])
def execute_custom_query(user_question: str) -> str:
    """
    Execute a custom database query for questions that don't fit the standard tools.
    This tool uses a specialized Cypher subgraph (LangGraph) to construct and execute custom queries.

    Use this tool when:
    - The user asks for complex aggregations or analytics
    - The user wants to find relationships between multiple entities
    - The question requires custom filtering or grouping
    - None of the other specialized tools can answer the question
    - The user asks about providers, organizations, or other non-patient entities
    - The user wants time-based analysis or trends

    Examples of when to use this tool:
    - "Which providers treated the most patients?"
    - "How many emergency visits were there in 2023?"
    - "What's the most common procedure performed?"
    - "Show me patients who had both diabetes and hypertension"
    - "Which organizations have the highest patient volume?"

    Args:
        user_question: The exact question the user asked in natural language

    Returns:
        Query results formatted as a readable string, or an error message if the query fails
    """
    try:
        # Invoke the Cypher subgraph with the user's question
        # The subgraph will:
        # 1. Generate appropriate Cypher query
        # 2. Execute the query using execute_cypher_query tool
        # 3. Format and return results
        response = query_cypher_subgraph(user_question)
        return response

    except Exception as e:
        return f"Error executing custom query: {str(e)}"


# Define the agent state
class AgentState(TypedDict):
    """State of the agent conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    executed_queries: list  # Track all Cypher queries executed in this conversation
    iteration_count: int  # Track number of reflection loop iterations
    route_decision: str  # Router's decision: "cypher" or "agent"


# Create the tools list (WITHOUT execute_custom_query - will use cypher node instead)
tools = [
    get_patient_procedures,
    get_patient_conditions,
    get_patient_medications,
    get_patient_encounters,
    search_patients,
    get_database_schema
]


# ============================================================================
# Model-Based Routing (Pydantic model for structured output)
# ============================================================================

class RouteDecision(BaseModel):
    """Classification of the user's query intent for routing decisions."""

    route: Literal["standard_tools", "custom_analytics", "conversational", "schema_info"] = Field(
        description=(
            "The routing decision based on query analysis:\n"
            "- 'standard_tools': Patient-specific queries that can use existing tools "
            "(e.g., get patient medications, procedures, conditions, search patients)\n"
            "- 'custom_analytics': Complex analytical queries requiring custom Cypher "
            "(e.g., most common diagnosis, patient demographics, statistics, aggregations)\n"
            "- 'conversational': Greetings, thanks, farewells - no data query needed\n"
            "- 'schema_info': Questions about database structure or schema"
        )
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why this route was chosen"
    )


# Custom function that wraps ToolNode to track Cypher queries
def create_query_tracking_tool_node(tools_list):
    """Create a tool node function that tracks Cypher queries."""
    base_tool_node = ToolNode(tools_list)

    def query_tracking_tool_node(state: AgentState) -> dict:
        """Execute tools and capture their Cypher queries."""
        # Log tool execution
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    log("TOOL", f"Executing with args: {tool_args}", tool_name)
                break

        # Execute the tools normally
        result = base_tool_node.invoke(state)

        # Log tool results
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    log("TOOL", f"RESULT: {msg.content[:300]}...")

        # Track queries that were executed
        new_queries = []

        # Get the last AI message with tool calls
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call["name"]

                    # Check if a query was executed by checking neo4j_conn
                    if neo4j_conn.last_executed_query:
                        log("DATABASE", f"Query executed: {neo4j_conn.last_executed_query[:200]}...")
                        new_queries.append({
                            "query": neo4j_conn.last_executed_query,
                            "params": neo4j_conn.last_query_params,
                            "source": tool_name,
                            "timestamp": datetime.now().isoformat(),
                            "tool_args": tool_call.get("args", {})
                        })
                        # Reset for next query
                        neo4j_conn.last_executed_query = None
                        neo4j_conn.last_query_params = None
                break

        # Update state with tracked queries
        existing_queries = state.get("executed_queries", [])
        updated_queries = existing_queries + new_queries

        return {
            **result,
            "executed_queries": updated_queries
        }

    return query_tracking_tool_node


# Create cypher subgraph node wrapper
def create_cypher_node():
    """
    Creates a node function that wraps the cypher subgraph.
    Extracts the generated Cypher query and adds it to main graph state.
    """
    cypher_subgraph = get_cypher_subgraph()

    def invoke_cypher_with_tracking(state: AgentState) -> dict:
        """Invoke cypher subgraph and track the generated query."""
        # Extract user question from the last message
        user_question = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        if not user_question:
            user_question = "Unknown query"

        # Invoke subgraph with its own state structure
        subgraph_result = cypher_subgraph.invoke({
            "messages": [],
            "user_question": user_question,
            "generated_cypher": "",
            "cypher_explanation": "",
            "query_results": ""
        })

        # Extract results
        final_response = subgraph_result["messages"][-1].content if subgraph_result["messages"] else "No response"
        generated_cypher = subgraph_result.get("generated_cypher", "Query not captured")

        # Create query tracking record
        new_query = {
            "query": generated_cypher,
            "params": {},
            "source": "cypher_subgraph",
            "timestamp": datetime.now().isoformat(),
            "question": user_question
        }

        # Update state
        existing_queries = state.get("executed_queries", [])
        updated_queries = existing_queries + [new_query]

        return {
            "messages": [AIMessage(content=final_response)],
            "executed_queries": updated_queries
        }

    return invoke_cypher_with_tracking


# Initialize the LLM with tools
def create_agent(api_key: str = None):
    """
    Create and configure the LangGraph agent.

    Args:
        api_key: SambaNova API key (defaults to SAMBANOVA_API_KEY env var)

    Returns:
        Compiled LangGraph agent
    """
    # Initialize LLM (using configured provider and model)
    model = get_main_agent_model()
    llm = create_llm(
        model=model,
        temperature=0,
        api_key=api_key
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # ========================================================================
    # Clean Routing Functions
    # ========================================================================
    def router_node(state: AgentState) -> dict:
        """
        Entry router: classifies user query and routes to cypher or agent.
        This runs FIRST for each new user message.

        IMPORTANT: Resets executed_queries to empty list so only queries
        from the current turn are tracked.
        """
        messages = state["messages"]

        # Reset executed_queries for this new user input
        # This ensures only queries from the current turn are shown
        log("APP", "Router: Resetting executed_queries for new user message")

        # Find the user's question
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            log("APP", "No user query found")
            return {"route_decision": "agent", "executed_queries": []}

        log("APP", f"Router processing query: {user_query[:80]}...")

        # Create router LLM using provider-specific config
        try:
            router_model = get_router_model()
            routing_llm = create_llm(
                model=router_model,
                temperature=0,
            ).with_structured_output(RouteDecision)
        except Exception as e:
            log("APP", f"Router LLM creation failed: {e}, defaulting to agent", level="error")
            return {"route_decision": "agent", "executed_queries": []}

        # Classification prompt
        prompt = f"""Classify this healthcare database query:

Query: {user_query}

Categories:
1. **custom_analytics**: Counting, aggregation, statistics, analytics
   Examples: "How many patients?", "Average age", "Most common diagnosis", "Total procedures"

2. **standard_tools**: Specific patient queries, searches, schema questions
   Examples: "What procedures did John have?", "Search for patient Maria", "Show database schema"

Classify as custom_analytics or standard_tools."""

        try:
            decision: RouteDecision = routing_llm.invoke(prompt)
            log("APP", f"Router decision: {decision.route} - {decision.reasoning}")

            route = "cypher" if decision.route == "custom_analytics" else "agent"
            return {"route_decision": route, "executed_queries": []}

        except Exception as e:
            log("APP", f"Router error: {e}, defaulting to agent", level="error")
            return {"route_decision": "agent", "executed_queries": []}

    def route_from_router(state: AgentState) -> str:
        """Route from router node based on classification."""
        route = state.get("route_decision", "agent")
        log("APP", f"Routing to: {route}")
        return route

    def route_from_agent(state: AgentState) -> str:
        """Route from agent: tools if tool calls, otherwise summary."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            log("APP", f"Agent called tools: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tools"

        log("APP", "Agent done, going to summary")
        return "summary"

    def route_from_cypher(state: AgentState) -> str:
        """Cypher always goes to summary."""
        log("APP", "Cypher done, going to summary")
        return "summary"

    def route_from_tools(state: AgentState) -> str:
        """Tools always go to summary."""
        log("APP", "Tools done, going to summary")
        return "summary"


    # Define the function that calls the model
    def call_model(state: AgentState) -> dict:
        """Call the LLM with the current state."""
        messages = state["messages"]

        # Increment iteration count
        current_iteration = state.get("iteration_count", 0)
        new_iteration = current_iteration + 1

        log("APP", f"call_model invoked: incrementing iteration from {current_iteration} to {new_iteration}")

        # Add system message if this is the first message
        if len(messages) == 1:
            system_message = """You are a helpful medical database assistant that helps users query patient information from the Synthea database.

The database contains patient healthcare records including:
- Patient demographics
- Healthcare encounters (visits)
- Procedures performed
- Diagnoses
- Medications prescribed

TOOL SELECTION GUIDELINES:

**Use the specialized tools for specific patient queries:**
- get_patient_procedures: When users ask about a specific patient's procedures. Use the patient's name (first, last, or full name).
- get_patient_conditions: When users ask about a specific patient's conditions/diagnoses. Use the patient's name (first, last, or full name).
- get_patient_medications: When users ask about a specific patient's medications. Use the patient's name (first, last, or full name).
- get_patient_encounters: When users ask about a specific patient's encounters/visits. Use the patient's name (first, last, or full name).
- search_patients: When you need to find or search for patients by name. Returns a list of matching patients.

**RESULT LIMITS:**
All tools return a maximum of 30 results by default, sorted by most recent first (for timestamped data).
- The default limit of 30 is sufficient for most queries and provides good performance
- Only increase the limit parameter if the user explicitly requests more results (e.g., "show me all 100 procedures" or "give me the last 50 medications")
- For patient search, results are sorted by name similarity to the search term
- For patient data (procedures, medications, conditions, encounters), results are sorted by most recent date first

**TOOL RESULT HANDLING:**
- The query results may be limited by a LIMIT clause and that is indicated in the response text (commonly LIMIT 30)
- If the query results are already limited, do not limit them further and show them as-is

**For complex analytics and aggregations:**
These questions will automatically be routed to a specialized Cypher query generator.
You don't need to select a tool - the system will handle it automatically.

Examples of questions that get special routing:
- "How many patients are in the database?"
- "What's the most common procedure?"
- "Which procedures were performed most frequently?"
- "Show me patients with the highest expenses"

Always be clear and helpful in your responses. If a patient is not found, suggest searching by name.
When presenting results, format them clearly and concisely."""

            messages = [HumanMessage(content=system_message)] + messages

        response = llm_with_tools.invoke(messages)

        if hasattr(response, 'content') and response.content:
            log("AGENT", f"Response content: {response.content[:200]}...", "MAIN")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log("AGENT", f"Made tool calls: {[tc['name'] for tc in response.tool_calls]}", "MAIN")

        return {
            "messages": [response],
            "iteration_count": new_iteration
        }

    # Define the function that generates the final response
    def call_model_summary(state: AgentState) -> dict:
        """
        Generate the final response to the user.
        Synthesizes information from tool calls or cypher queries into a natural answer.
        """
        log("APP", "call_model_summary invoked: generating final response")

        messages = state["messages"]

        # Create a system prompt for final response generation
        summary_system_message = """You are a helpful medical database assistant.

Your task is to provide a clear, natural final response to the user based on the information gathered.

CRITICAL - OUTPUT FORMAT:
- ONLY provide the answer in plain, natural language
- DO NOT include any Cypher queries, SQL, or code in your response
- DO NOT use special tags like <|python_start|> or code blocks
- DO NOT format results as tables with pipes (|) or dashes (-)
- The Cypher queries are tracked separately and shown to the user separately
- Your response should ONLY contain the natural language answer

INSTRUCTIONS:
1. Review the conversation history, including any tool outputs or query results
2. Extract ALL specific data from previous responses (names, numbers, dollar amounts, dates, etc.)
3. PRESERVE exact numerical values - if you see "$4,570,388.07" use that EXACT amount
4. DO NOT add uncertainty or qualifiers like "I would need more context" when data is already provided
5. The query results may be limited by a LIMIT clause and that is indicated in the response text (commonly LIMIT 30)
6. If the query results are already limited, do not limit them further and show them as-is
7. Provide a direct, helpful answer to the user's question using the EXACT data provided
8. Be conversational and natural - don't use unnecessary preambles
9. If data was successfully retrieved: Present the findings clearly with all specific values
10. If there were errors: Explain what went wrong in user-friendly terms
11. DO NOT return empty responses, symbols like "[]", or placeholder text

CRITICAL - HANDLING LIMITED RESULTS:
- If the query has ORDER BY date DESC and mentions a limit, say "most recent"
- If the result count is less than the limit mentioned in the tool output, the tool will NOT mention the limit

RESPONSE STYLE:
- Answer the question directly - no preambles or filler phrases like "To provide a more comprehensive answer"
- DO NOT add phrases like "I would need more context" or "based on available information" when complete data is provided
- If you have specific numbers/amounts in the conversation, USE THEM - don't say you need more information
- Pay attention to singular vs plural in the user's question:
  - "Which patient..." (singular) → Provide ONLY ONE result (the top/most relevant)
  - "Which patients..." (plural) → Provide multiple results (top 10-20)
  - "What is the most..." (singular) → ONE result
  - "What are the top..." (plural) → Multiple results
- For single items: Use a simple sentence, NOT a bulleted or numbered list
- For multiple items (2+): Use numbers, bullets, or clear sentences as appropriate
- Be concise but complete

EXAMPLES:

Good Example 1 (successful count query):
"There are 5,885 patients in the database."

Good Example 2 (single result - no bullet list):
"Leonor133 Dicki44 has spent the most on treatments with total expenses of $4,570,388.07."

Good Example 3 (another single result - no bullet):
"The most common procedure is Electrocardiogram, performed 1,245 times."

Good Example 4 (patient data with LIMIT 30 - multiple items):
"Ethan766 has had the following procedures (showing the most recent 30):
1. Electrocardiogram (2023-06-15)
2. Chest X-ray (2023-03-22)
3. Blood pressure monitoring (2023-01-10)
[...27 more results]"

Good Example 5 (exactly 30 results with LIMIT 30):
"The most recent 30 diagnoses for patient Smith include:
- Hypertension
- Type 2 Diabetes
[...28 more]

Note: There may be additional diagnoses beyond these 30 records."

Good Example 6 (error case):
"I was unable to locate patient Ethan766 in the database. The patient may not exist in the system, or the identifier may be incorrect. Try searching by full name instead."

Good Example 7 (analytics - multiple items):
"The most frequently performed procedures are:
1. Electrocardiogram - 1,245 times
2. Blood pressure check - 892 times
3. Chest X-ray - 654 times"

Good Example 8 (search results with limit):
"Found 30 patients matching 'John' (showing most relevant matches):
1. John Smith - DOB: 1985-03-15
2. John Doe - DOB: 1990-07-22
[...28 more]"

IMPORTANT - Extracting Data from Previous Responses:
If you see a previous AI message like: "The patient who has spent the most on treatments is Leonor133 Dicki44, with total healthcare expenses of $4,570,388.07."

Your response should preserve the exact data:
CORRECT: "Leonor133 Dicki44 has spent the most on treatments with total expenses of $4,570,388.07."
WRONG: "Leonor133 Dicki44 has spent the most on treatments." (missing amount!)
WRONG: "To provide a more comprehensive answer, I would need more context..." (data is already there!)
WRONG: "Based on the available information, Leonor133 Dicki44 has spent the most..." (don't add uncertainty!)

Extract the name AND the amount, and present them clearly without preambles.

Provide your response now."""

        # Add the summary system message to the conversation
        summary_messages = [HumanMessage(content=summary_system_message)] + messages

        log("APP", f"call_model_summary: Processing {len(messages)} messages to generate final response")

        # Log the last few messages for debugging
        for i, msg in enumerate(messages[-5:]):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:150] if hasattr(msg, 'content') else "No content"
            log("APP", f"  Message {i}: {msg_type} - {content_preview}")

        # Call the LLM without tools (just for generating the final response)
        response = llm.invoke(summary_messages)

        log("APP", f"call_model_summary FULL response: {response.content if hasattr(response, 'content') else 'No content'}")

        return {
            "messages": [response]
        }

    # Create query-tracking tool node
    tool_node = create_query_tracking_tool_node(tools)

    # Create cypher subgraph node
    cypher_node = create_cypher_node()

    # Build the graph with clean routing
    workflow = StateGraph(AgentState)

    # ========================================================================
    # CLEAN GRAPH STRUCTURE:
    # START → router → [cypher → summary] OR [agent → tools → summary] → END
    # ========================================================================
    log("APP", "Building graph with simplified routing")

    # Add all nodes
    workflow.add_node("router", router_node)  # Entry point - classifies query
    workflow.add_node("agent", call_model)     # Standard tool-calling agent
    workflow.add_node("tools", tool_node)      # Tool execution
    workflow.add_node("cypher", cypher_node)   # Analytics cypher subgraph
    workflow.add_node("summary", call_model_summary)  # Final response generation

    # Set router as entry point
    workflow.set_entry_point("router")

    # Router routes to cypher or agent
    workflow.add_conditional_edges(
        "router",
        route_from_router,
        {
            "cypher": "cypher",
            "agent": "agent"
        }
    )

    # Agent routes to tools or summary
    workflow.add_conditional_edges(
        "agent",
        route_from_agent,
        {
            "tools": "tools",
            "summary": "summary"
        }
    )

    # Cypher always goes to summary
    workflow.add_conditional_edges(
        "cypher",
        route_from_cypher,
        {
            "summary": "summary"
        }
    )

    # Tools always go to summary
    workflow.add_conditional_edges(
        "tools",
        route_from_tools,
        {
            "summary": "summary"
        }
    )

    # Summary always ends
    workflow.add_edge("summary", END)

    # Compile and return
    return workflow.compile()


@traceable(name="query_agent", tags=["agent", "main"])
def query_agent(agent, user_message: str, conversation_history: list = None, executed_queries: list = None) -> tuple[str, list, list]:
    """
    Query the agent with a user message.

    Args:
        agent: Compiled LangGraph agent
        user_message: User's question
        conversation_history: Previous conversation messages
        executed_queries: Previous executed queries

    Returns:
        Tuple of (response_text, updated_conversation_history, executed_queries)
    """
    if conversation_history is None:
        conversation_history = []
    if executed_queries is None:
        executed_queries = []

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_message))

    # Run the agent with state including executed_queries and iteration_count
    result = agent.invoke({
        "messages": conversation_history,
        "executed_queries": executed_queries,
        "iteration_count": 0  # Initialize iteration count for each new query
    })

    # Get the final response
    final_message = result["messages"][-1]
    response_text = final_message.content

    # Update conversation history and queries from state
    conversation_history = result["messages"]
    executed_queries = result.get("executed_queries", [])

    return response_text, conversation_history, executed_queries
