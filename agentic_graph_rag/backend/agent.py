"""
LangGraph agent for querying Synthea Neo4j database with natural language.
"""
import json
from datetime import datetime
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from llm_factory import create_llm, get_main_agent_model, get_validation_model
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import Command
from neo4j_utils import Neo4jConnection
from cypher_subagent import query_cypher_subgraph
from langsmith.run_helpers import traceable
from logging_utils import log
from pydantic import BaseModel, Field


# Initialize Neo4j connection
neo4j_conn = Neo4jConnection()
neo4j_conn.connect()


# Define tools for the agent
@tool
@traceable(name="get_patient_procedures", run_type="tool", tags=["tool", "patient", "procedures"])
def get_patient_procedures(patient_name: str, limit: int = 30, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Get procedures that have been performed on a patient, sorted by most recent first.

    Note: Returns the actual number of procedures found, up to the limit specified.
    The output will indicate "(showing up to X most recent)" ONLY if the result count equals the limit.
    If fewer results are found, the output will NOT mention the limit.

    Args:
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    log("TOOL", f"Called with patient_name='{patient_name}', limit={limit}", "get_patient_procedures")

    try:
        results, query, params = neo4j_conn.get_patient_procedures(patient_name, limit=limit)

        if not results:
            log("TOOL", f"No procedures found for '{patient_name}'", "get_patient_procedures")
            output = f"No procedures found for patient '{patient_name}'. The patient may not exist or has no recorded procedures."
        else:
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

        # Create query tracking entry
        query_entry = {
            "query": query,
            "params": params,
            "source": "get_patient_procedures",
            "tool_args": {"patient_name": patient_name, "limit": limit},
            "timestamp": datetime.now().isoformat()
        }

        # Return Command that updates both messages (for LLM) and executed_queries (for tracking)
        return Command(
            update={
                "messages": [ToolMessage(content=output, tool_call_id=tool_call_id)],
                "executed_queries": [query_entry]
            }
        )
    except Exception as e:
        log("TOOL", f"Error - {str(e)}", "get_patient_procedures", level="error")
        error_msg = f"Error retrieving procedures: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]
            }
        )


@tool
@traceable(name="get_patient_conditions", run_type="tool", tags=["tool", "patient", "conditions"])
def get_patient_conditions(patient_name: str, limit: int = 30, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Get medical conditions/diagnoses for a patient, sorted by most recent first.

    Note: Returns the actual number of conditions found, up to the limit specified.
    The output will indicate "(showing up to X most recent)" ONLY if the result count equals the limit.
    If fewer results are found, the output will NOT mention the limit.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    try:
        results, query, params = neo4j_conn.get_patient_conditions(patient_name, limit=limit)

        if not results:
            output = f"No conditions found for patient '{patient_name}'. The patient may not exist or has no recorded conditions."
        else:
            # Only mention limit if we hit the limit (results count equals limit)
            if len(results) >= limit:
                output = f"Conditions for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
            else:
                output = f"Conditions for {results[0]['patient_name']}:\n\n"

            for i, record in enumerate(results, 1):
                output += f"{i}. {record['condition_description']}\n"
                output += f"   Diagnosed: {record['encounter_date']}\n\n"

        # Create query tracking entry
        query_entry = {
            "query": query,
            "params": params,
            "source": "get_patient_conditions",
            "tool_args": {"patient_name": patient_name, "limit": limit},
            "timestamp": datetime.now().isoformat()
        }

        return Command(
            update={
                "messages": [ToolMessage(content=output, tool_call_id=tool_call_id or "")],
                "executed_queries": [query_entry]
            }
        )
    except Exception as e:
        error_msg = f"Error retrieving conditions: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id or "")]
            }
        )


@tool
@traceable(name="get_patient_medications", run_type="tool", tags=["tool", "patient", "medications"])
def get_patient_medications(patient_name: str, limit: int = 30, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Get medications prescribed to a patient, sorted by most recent first.

    Note: Returns the actual number of medications found, up to the limit specified.
    The output will indicate "(showing up to X most recent)" ONLY if the result count equals the limit.
    If fewer results are found, the output will NOT mention the limit.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    try:
        results, query, params = neo4j_conn.get_patient_medications(patient_name, limit=limit)

        if not results:
            output = f"No medications found for patient '{patient_name}'. The patient may not exist or has no prescribed medications."
        else:
            # Only mention limit if we hit the limit (results count equals limit)
            if len(results) >= limit:
                output = f"Medications for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
                log("TOOL", f"Hit limit: {len(results)} >= {limit}, including limit language", "get_patient_medications")
            else:
                output = f"Medications for {results[0]['patient_name']}:\n\n"
                log("TOOL", f"Under limit: {len(results)} < {limit}, NOT including limit language", "get_patient_medications")

            for i, record in enumerate(results, 1):
                output += f"{i}. {record['medication']}\n"
                output += f"   Prescribed: {record['prescribed_date']}\n\n"

            log("TOOL", f"Tool output starts with: {output[:100]}", "get_patient_medications")

        # Create query tracking entry
        query_entry = {
            "query": query,
            "params": params,
            "source": "get_patient_medications",
            "tool_args": {"patient_name": patient_name, "limit": limit},
            "timestamp": datetime.now().isoformat()
        }

        return Command(
            update={
                "messages": [ToolMessage(content=output, tool_call_id=tool_call_id or "")],
                "executed_queries": [query_entry]
            }
        )
    except Exception as e:
        error_msg = f"Error retrieving medications: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id or "")]
            }
        )


@tool
@traceable(name="get_patient_encounters", run_type="tool", tags=["tool", "patient", "encounters"])
def get_patient_encounters(patient_name: str, limit: int = 30, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Get healthcare encounters for a patient, sorted by most recent first.

    Note: Returns the actual number of encounters found, up to the limit specified.
    The output will indicate "(showing up to X most recent)" ONLY if the result count equals the limit.
    If fewer results are found, the output will NOT mention the limit.

    Args:
        patient_name: The patient's name (can be first name, last name, or full name like 'John Smith')
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    try:
        results, query, params = neo4j_conn.get_patient_encounters(patient_name, limit=limit)

        if not results:
            output = f"No encounters found for patient '{patient_name}'. The patient may not exist."
        else:
            # Only mention limit if we hit the limit (results count equals limit)
            if len(results) >= limit:
                output = f"Encounters for {results[0]['patient_name']} (showing up to {limit} most recent):\n\n"
            else:
                output = f"Encounters for {results[0]['patient_name']}:\n\n"

            for i, record in enumerate(results, 1):
                output += f"{i}. {record['encounter_type']}\n"
                output += f"   Date: {record['encounter_date']}\n"
                output += f"   Type: {', '.join(record['encounter_labels'])}\n\n"

        # Create query tracking entry
        query_entry = {
            "query": query,
            "params": params,
            "source": "get_patient_encounters",
            "tool_args": {"patient_name": patient_name, "limit": limit},
            "timestamp": datetime.now().isoformat()
        }

        return Command(
            update={
                "messages": [ToolMessage(content=output, tool_call_id=tool_call_id or "")],
                "executed_queries": [query_entry]
            }
        )
    except Exception as e:
        error_msg = f"Error retrieving encounters: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id or "")]
            }
        )


@tool
@traceable(name="search_patients", run_type="tool", tags=["tool", "patient", "search"])
def search_patients(search_term: str = None, limit: int = 30, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
    """
    Search for patients by name, sorted by relevance. Use this when you need to find a patient or list patients.

    Note: Returns the actual number of patients found, up to the limit specified.
    The output will indicate "showing up to X results" accordingly.

    Args:
        search_term: Optional. Part of the patient's first or last name to search for. If not provided, returns recent patients.
        limit: Maximum number of results to return (default: 30). Only increase this if the user explicitly asks for more results.
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    try:
        results, query, params = neo4j_conn.search_patients(search_term, limit=limit)

        if not results:
            output = f"No patients found matching '{search_term}'."
        else:
            if search_term:
                output = f"Found patients matching '{search_term}' (showing up to {limit} results, sorted by relevance):\n\n"
            else:
                output = f"Found patients (showing up to {limit} most recent):\n\n"

            for i, record in enumerate(results, 1):
                output += f"{i}. {record['patient_name']} (ID: {record['patient_id']})\n"
                output += f"   Birth Date: {record['birth_date']}\n\n"

        # Create query tracking entry
        query_entry = {
            "query": query,
            "params": params,
            "source": "search_patients",
            "tool_args": {"search_term": search_term, "limit": limit},
            "timestamp": datetime.now().isoformat()
        }

        return Command(
            update={
                "messages": [ToolMessage(content=output, tool_call_id=tool_call_id or "")],
                "executed_queries": [query_entry]
            }
        )
    except Exception as e:
        error_msg = f"Error searching patients: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id or "")]
            }
        )


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
def execute_custom_query(user_question: str, tool_call_id: Annotated[str, InjectedToolCallId] = None) -> Command:
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
        tool_call_id: Internal parameter for tracking (automatically provided by LangGraph)

    Returns:
        Command object that updates both LLM-visible messages and internal query tracking state.
    """
    try:
        # Import here to avoid circular dependency
        from cypher_subagent import get_cypher_subgraph

        # Get the cypher subgraph and invoke it
        cypher_subgraph = get_cypher_subgraph()

        # Invoke subgraph with its own state structure
        subgraph_result = cypher_subgraph.invoke({
            "messages": [],
            "user_question": user_question,
            "generated_cypher": "",
            "cypher_explanation": "",
            "query_results": ""
        })

        # Extract the response and generated cypher
        response = subgraph_result["messages"][-1].content if subgraph_result["messages"] else "No response"
        generated_cypher = subgraph_result.get("generated_cypher", "")

        # Create query tracking entry with all details
        query_entry = {
            "query": generated_cypher,
            "params": {},
            "source": "execute_custom_query",
            "tool_args": {"user_question": user_question},
            "question": user_question,  # Include the question for frontend display
            "timestamp": datetime.now().isoformat()
        }

        return Command(
            update={
                "messages": [ToolMessage(content=response, tool_call_id=tool_call_id or "")],
                "executed_queries": [query_entry] if generated_cypher else []  # Only track if query was generated
            }
        )

    except Exception as e:
        error_msg = f"Error executing custom query: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id or "")]
            }
        )


# Reducer function for executed_queries - concatenates new queries to existing list
def add_queries(existing: list, new: list) -> list:
    """
    Reducer that concatenates new queries to the existing list.
    This is needed for the Command pattern - when tools return {"executed_queries": [...]},
    LangGraph uses this reducer to append (not replace) the queries.
    """
    return existing + new


# Define the agent state
class AgentState(TypedDict):
    """State of the agent conversation."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    executed_queries: Annotated[list, add_queries]  # Track all Cypher queries executed (with reducer)
    iteration_count: int  # Track number of reflection loop iterations
    route_decision: str  # Router's decision: "cypher" or "agent"

    # Note: executed_queries serves as our "user_info_for_display" channel - it stores
    # structured query data that the caller can access, separate from LLM messages.
    # The reducer ensures that when tools return new queries via Command, they are
    # appended to the list rather than replacing it.


# Create the tools list (INCLUDING execute_custom_query for analytics)
tools = [
    get_patient_procedures,
    get_patient_conditions,
    get_patient_medications,
    get_patient_encounters,
    search_patients,
    get_database_schema,
    execute_custom_query  # Added back for main agent to handle analytics
]


# ============================================================================
# Validation Decision Model (Pydantic model for structured output)
# ============================================================================

class ValidationDecision(BaseModel):
    """Validation of user query relevance to Synthea database schema."""

    is_relevant: bool = Field(
        description=(
            "Whether the user's query is relevant to the Synthea medical database:\n"
            "- True: Query is about patients, medical records, procedures, conditions, "
            "medications, encounters, providers, or organizations in the database\n"
            "- False: Query is unrelated to medical records or asks about topics outside "
            "the database scope (e.g., general knowledge, weather, news, other domains)"
        )
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) of why the query is or isn't relevant"
    )


# NOTE: Query tracking is now handled via the Command pattern in tools.
# Tools return Command objects that update both messages and executed_queries state.
# The standard ToolNode handles Command returns automatically with the reducer.


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
    # Validation Node Function
    # ========================================================================
    def validation_node(state: AgentState) -> dict:
        """
        Entry validation: checks if user query is relevant to Synthea database.
        This runs FIRST for each new user message.
        """
        messages = state["messages"]

        # Find the user's question
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            log("APP", "No user query found in validation")
            return {"messages": messages}

        log("APP", f"Validation processing query: {user_query[:80]}...")

        # Create validation LLM using provider-specific config
        try:
            validation_model = get_validation_model()
            validation_llm = create_llm(
                model=validation_model,
                temperature=0,
            ).with_structured_output(ValidationDecision)
        except Exception as e:
            log("APP", f"Validation LLM creation failed: {e}, proceeding to agent", level="error")
            return {"messages": messages}

        # Get database schema for validation context
        schema_description = neo4j_conn.get_database_schema()

        # Validation prompt with schema context
        validation_prompt = f"""You are a query validator for the Synthea medical records database.

DATABASE SCHEMA:
{schema_description}

USER QUERY: {user_query}

Your task: Determine if the user's query is relevant to this medical database.

RELEVANT queries include:
- Questions about patients, their medical history, procedures, conditions, medications
- Questions about healthcare encounters, providers, organizations
- Statistical queries about medical data (counts, averages, trends)
- Searching for or retrieving medical records
- Questions about the database structure or schema

IRRELEVANT queries include:
- General knowledge questions unrelated to medical records
- Questions about current events, news, weather
- Questions about topics outside healthcare/medical domain
- Requests for information not stored in this database

Classify this query as relevant (True) or irrelevant (False)."""

        try:
            decision: ValidationDecision = validation_llm.invoke(validation_prompt)
            log("APP", f"Validation decision: {decision.is_relevant} - {decision.reasoning}")

            if not decision.is_relevant:
                # Query is not relevant - return rejection message
                rejection_message = "I am sorry - I can only answer questions that pertain to the Synthea medical records."
                log("APP", f"Query rejected as irrelevant: {user_query[:80]}...")
                return {
                    "messages": messages + [AIMessage(content=rejection_message)]
                }

            # Query is relevant - proceed with normal flow
            log("APP", "Query validated as relevant, proceeding to agent")
            return {"messages": messages}

        except Exception as e:
            log("APP", f"Validation error: {e}, proceeding to agent", level="error")
            return {"messages": messages}

    # ========================================================================
    # Simplified Routing Functions
    # ========================================================================
    def route_from_validation(state: AgentState) -> str:
        """Route from validation: if query was rejected, end. Otherwise, go to agent."""
        messages = state["messages"]
        last_message = messages[-1]

        # Check if the last message is a rejection from validation
        if isinstance(last_message, AIMessage) and "I am sorry - I can only answer questions that pertain to the Synthea medical records" in last_message.content:
            log("APP", "Query rejected by validation, ending")
            return "end"

        log("APP", "Query validated, routing to agent")
        return "agent"

    def route_from_agent(state: AgentState) -> str:
        """Route from agent: tools if tool calls, otherwise end."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            log("APP", f"Agent called tools: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tools"

        log("APP", "Agent done with final response, ending")
        return "end"


    # Define the function that calls the model
    def call_model(state: AgentState) -> dict:
        """Call the LLM with the current state."""
        messages = state["messages"]

        # Increment iteration count
        current_iteration = state.get("iteration_count", 0)
        new_iteration = current_iteration + 1

        log("APP", f"call_model invoked: incrementing iteration from {current_iteration} to {new_iteration}")

        # Detect if we have tool results (synthesis mode)
        has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages[-10:])

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
- get_database_schema: When users ask about database structure or schema information.

**Use execute_custom_query for complex analytics:**
- Aggregations and counts ("How many patients...", "What's the total...", "Average age...")
- Top-N queries ("Which patients spent the most...", "Most common procedures...", "Top 10...")
- Time-based analysis ("Patients admitted in 2023...", "Encounters last month...")
- Multi-entity relationships ("Patients with both diabetes AND hypertension...")
- Provider/organization queries ("Which providers treated...", "Organizations with highest volume...")
- Statistical queries ("Distribution of...", "Trends over time...")

**RESULT LIMITS:**
All tools return a maximum of 30 results by default, sorted by most recent first (for timestamped data).
- The default limit of 30 is sufficient for most queries and provides good performance
- Only increase the limit parameter if the user explicitly requests more results (e.g., "show me all 100 procedures" or "give me the last 50 medications")
- For patient search, results are sorted by name similarity to the search term
- For patient data (procedures, medications, conditions, encounters), results are sorted by most recent date first

**TOOL RESULT HANDLING:**
- The query results may be limited by a LIMIT clause and that is indicated in the response text (commonly LIMIT 30)
- If the query results are already limited, do not limit them further and show them as-is

**MULTI-STEP QUERIES:**
If a query requires multiple pieces of information, call ALL relevant tools in a single response.

Example: "What procedures and medications did John Smith have?"
→ Call BOTH get_patient_procedures AND get_patient_medications in the same response

**RESPONDING TO USERS:**
When you receive results from tool calls, synthesize the information into a clear, natural answer.
- Extract specific details (names, dates, numbers, amounts) from tool results
- Present information in a conversational way
- For multiple tool results, organize information logically
- Be concise but complete - answer the user's question directly
- If data was successfully retrieved, present findings with all specific values
- If there were errors, explain what went wrong in user-friendly terms

Always be clear and helpful in your responses. If a patient is not found, suggest searching by name."""

            messages = [HumanMessage(content=system_message)] + messages

        # If we have tool results, inject synthesis guidance
        elif has_tool_results and current_iteration > 0:
            # Count how many tool results we have
            tool_count = sum(1 for msg in messages[-10:] if isinstance(msg, ToolMessage))

            log("APP", f"Synthesis mode: {tool_count} tool result(s) detected")

            synthesis_instruction = """You have received results from your tool calls.

*** CRITICAL RULE #1 - LIMIT LANGUAGE ***
DO NOT ADD PHRASES LIKE "showing up to 30" OR "top 30" UNLESS THE TOOL OUTPUT CONTAINS THEM!
- Check the FIRST LINE of the tool result message
- If it says "(showing up to X most recent)" → copy that exact phrase
- If it does NOT have that phrase → do NOT add any "showing up to" or "top X" language
- Instead, just state the actual count (e.g., "has 21 medications")

IMPORTANT - SYNTHESIZE YOUR FINAL ANSWER NOW:

1. Review all tool results and extract specific data (names, dates, numbers, amounts)
2. Synthesize the information into a clear, natural answer to the user's question
3. DO NOT make additional tool calls - just provide the final response
4. Present information in a conversational, user-friendly way

FORMATTING GUIDELINES:

**For Questions Requesting Multiple Records (procedures, medications, conditions, encounters, etc.):**
- ALWAYS present the results as a numbered list
- Include a brief introductory sentence with the patient name and count
- Check the tool output's first line for limit language:
  * Contains "(showing up to X most recent)" → Say "has had at least X [items]. Here are X of the most recent [items]:"
  * Does NOT contain limit language → Say "has X [items]:" where X is the actual count
- Then list ALL items from the tool results
- Examples:
  * Tool says "(showing up to 30 most recent)" → "Ethan766 Nolan344 has had at least 30 procedures. Here are 30 of the most recent procedures:"
  * Tool has no limit phrase → "Ethan766 Nolan344 has 21 medications:"

**For Single Item or Summary Queries:**
- Present findings naturally in clear sentences
- Example: "John Smith's most recent procedure was an Electrocardiogram on 2023-06-15."

**For Multiple Tool Results (combining different types of data):**
- Organize information logically (chronologically, by category, or by relationship)
- Use connecting phrases to create natural flow
- Use lists for each data type if appropriate
- Example: "John Smith has had several procedures including an Electrocardiogram on 2023-06-15 and a Chest X-ray on 2023-03-22. He is currently taking Metformin for Type 2 Diabetes and Lisinopril for Hypertension."

CRITICAL RULES:
- Use EXACT numbers, dates, and amounts from tool results - don't approximate
- Don't add uncertainty phrases like "approximately" or "based on available information" when complete data is provided
- Plain natural language ONLY - NO Cypher queries, SQL, or code in your response
- NO special tags or table formatting with pipes (|) or dashes (-)
- PRESERVE RESULT COUNT LANGUAGE: If the tool output mentions a specific count (e.g., "showing up to 30") or doesn't mention a limit at all, use that exact language. DO NOT add phrases like "top 30" or "most recent 30" if the tool output doesn't include them.
- When the tool output contains a numbered or bulleted list, preserve that list format in your response
- FORMAT TIMESTAMPS: Convert ISO 8601 timestamps (e.g., "2023-01-01T12:48:47.000000000+00:00") to natural date format (e.g., "January 1, 2023" or "2023-01-01"). Strip time and timezone unless specifically relevant.

Provide your synthesized answer now:"""

            messages = messages + [HumanMessage(content=synthesis_instruction)]

        response = llm_with_tools.invoke(messages)

        if hasattr(response, 'content') and response.content:
            log("AGENT", f"Response content: {response.content[:200]}...", "MAIN")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            log("AGENT", f"Made tool calls: {[tc['name'] for tc in response.tool_calls]}", "MAIN")

        return {
            "messages": [response],
            "iteration_count": new_iteration
        }

    # NOTE: call_model_summary() removed - agent now synthesizes its own responses
    # after receiving tool results using conditional synthesis instruction injection

    # Create standard tool node - tools return Command objects that handle query tracking
    tool_node = ToolNode(tools)

    # Build the graph with validation and simplified routing
    workflow = StateGraph(AgentState)

    # ========================================================================
    # OPTIMIZED GRAPH STRUCTURE WITH VALIDATION:
    # START → validation → [agent → tools → agent (synthesis)] OR [end] → END
    #
    # Key features:
    # - Validation node checks query relevance before processing
    # - No router node (agent handles tool selection including execute_custom_query)
    # - No cypher node (now a tool that agent can call)
    # - No summary node (agent synthesizes its own tool results)
    # - LLM calls: validation (1) + agent (1-2) = 2-3 total (vs 3-4 before)
    # ========================================================================
    log("APP", "Building optimized graph with validation and agent self-synthesis")

    # Add nodes
    workflow.add_node("validation", validation_node)  # Entry validation - checks relevance
    workflow.add_node("agent", call_model)            # Main agent - tool selection and synthesis
    workflow.add_node("tools", tool_node)             # Tool execution

    # Set validation as entry point
    workflow.set_entry_point("validation")

    # Validation routes to agent or end
    workflow.add_conditional_edges(
        "validation",
        route_from_validation,
        {
            "agent": "agent",
            "end": END
        }
    )

    # Agent routes to tools or end
    workflow.add_conditional_edges(
        "agent",
        route_from_agent,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Tools always route back to agent for synthesis (deterministic)
    workflow.add_edge("tools", "agent")

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
        executed_queries: Previous executed queries (not used, reset each turn)

    Returns:
        Tuple of (response_text, updated_conversation_history, executed_queries)
    """
    if conversation_history is None:
        conversation_history = []

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_message))

    # Run the agent with state - reset executed_queries to empty list for each new turn
    # This ensures only queries from the current turn are returned
    result = agent.invoke({
        "messages": conversation_history,
        "executed_queries": [],  # Reset to empty list for each new query
        "iteration_count": 0  # Initialize iteration count for each new query
    })

    # Get the final response
    final_message = result["messages"][-1]
    response_text = final_message.content

    # Update conversation history and get only the queries from this turn
    conversation_history = result["messages"]
    executed_queries = result.get("executed_queries", [])

    return response_text, conversation_history, executed_queries
