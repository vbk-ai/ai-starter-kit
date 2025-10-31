# Agentic Graph RAG for Healthcare Data

An AI-powered agentic system that enables natural language queries over the Synthea healthcare graph database using intelligent routing, pre-built tools, and dynamic Cypher query generation. Built with LangGraph, FastAPI, and a modern web interface.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Agent Tools & Architecture](#agent-tools--architecture)
- [Cypher Subagent Technical Guide](#cypher-subagent-technical-guide)
- [Development](#development)
- [Deployment](#deployment)
- [What Was Built](#what-was-built)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Technologies Used](#technologies-used)

## Overview

This project implements an agentic Graph RAG (Retrieval-Augmented Generation) system for healthcare data analysis. It combines:

- **Intelligent Query Routing**: Automatically classifies queries and routes to appropriate handlers
- **Pre-built Tools**: Fast, optimized queries for common patient data operations
- **Dynamic Cypher Generation**: LLM-powered query construction for complex analytics
- **Graph Database**: Neo4j with Synthea synthetic patient data (5,885 patients, 1.2M+ encounters)
- **Configurable LLMs**: Support for Anthropic Claude and SambaNova models

The system intelligently determines whether to use pre-built queries or generate custom Cypher based on query complexity, optimizing for both performance and flexibility.

## Quick Start

Get up and running in 5 minutes!

### Prerequisites Checklist

- [ ] Python 3.11 or higher installed
- [ ] Neo4j running with synthea-sample database
- [ ] LLM API key (Anthropic or SambaNova)

### Installation Steps

1. **Navigate to Project Directory**
```bash
cd agentic_graph_rag
```

2. **Create Virtual Environment**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Dependencies**
```bash
uv pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Copy the example file
cp backend/.env.example backend/.env

# Edit backend/.env and add your API keys
# nano backend/.env  # or use your favorite editor
```

**Required in .env:**
```env
# API Keys - Add the key for your chosen provider
ANTHROPIC_API_DEV_KEY=sk-ant-your-key-here  # If using Anthropic
SAMBANOVA_API_KEY=your-key-here  # If using SambaNova

# Provider Configuration - Choose one (can be switched at runtime via UI)
PROVIDER=anthropic  # or 'sambanova'

# Provider-Specific Model Configurations
# Anthropic Configuration (Claude models)
ANTHROPIC_MAIN_AGENT_LLM=claude-sonnet-4-5-20250929
ANTHROPIC_ROUTER_LLM=claude-haiku-4-5-20251001
ANTHROPIC_CYPHER_AGENT_LLM=claude-sonnet-4-5-20250929

# SambaNova Configuration (DeepSeek models)
SAMBANOVA_MAIN_AGENT_LLM=DeepSeek-V3.1
SAMBANOVA_ROUTER_LLM=DeepSeek-V3.1
SAMBANOVA_CYPHER_AGENT_LLM=DeepSeek-V3.1

# Neo4j Database Configuration
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# Server Configuration (optional)
PORT=8000
MAX_AGENT_ITERATIONS=3

# Debug Logging (optional - set to 'true' to enable)
AGENT_DEBUG_LOGGING=false
TOOL_DEBUG_LOGGING=false
DATABASE_DEBUG_LOGGING=false
APP_DEBUG_LOGGING=false
```

5. **Test Connection**
```bash
python test_connection.py
```

You should see:
```
✓ Successfully connected to Neo4j
✓ Query successful: Found 5885 patients
✓ All tests completed successfully!
```

6. **Start Server**

**Option A: Using the startup script (Linux/Mac)**
```bash
./start_server.sh
```

**Option B: Manual start**
```bash
cd backend
python server.py
```

7. **Access the Application**

Open your browser and go to: **http://localhost:8000/app**

### Try These Sample Queries

1. "What procedures has Ethan had?" (matches by first name)
2. "Show me patients named John" (patient search)
3. "What medications is taking Smith?" (matches by last name)
4. "Which providers treated the most patients?" (analytics query)
5. "How many patients are in the database?" (count aggregation)

## Key Features

- **Natural Language Queries**: Ask questions in plain English about patient records
- **LangGraph Agent**: Uses LangGraph with ToolNode for intelligent query routing
- **Cypher Subagent**: Specialized AI agent for constructing custom Cypher queries for complex questions
- **Multi-Provider Support**:
  - Switch between Anthropic and SambaNova providers at runtime via UI dropdown
  - Provider-specific model configurations (main agent, router, cypher agent)
  - Seamless provider switching without server restart
- **Dual-Mode Querying**:
  - Pre-built tools for common queries (patient procedures, conditions, medications)
  - Custom query generation for complex analytics and aggregations
- **Flexible Patient Matching**: Query by first name, last name, or full name
- **Neo4j Integration**: Directly queries the Synthea-sample database
- **RESTful API**: FastAPI backend with automatic documentation
- **Modern UI**: Clean, responsive web interface with provider selection
- **Session Management**: Maintains conversation context across multiple queries
- **Query Transparency**: See all executed Cypher queries in collapsible widgets
- **Granular Debug Logging**: Four separate logging controls (Agent, Tool, Database, App)

## Architecture

### Backend (Python)
- **LangGraph**: Orchestrates the AI agent workflow with tool calling
- **LangChain**: Provides LLM integration and tool abstractions
- **FastAPI**: High-performance web server
- **Neo4j Driver**: Direct database connectivity
- **Multi-LLM Support**: Configurable LLM providers (Anthropic Claude, SambaNova DeepSeek)

### Frontend
- **Pure HTML/JavaScript**: No framework dependencies
- **Modern CSS**: Gradient design with smooth animations
- **Real-time Communication**: Fetch API for backend integration

### Database
- **Neo4j**: Graph database containing Synthea patient data
- **Database**: synthea-sample

## Project Structure

```
agentic_graph_rag/
├── backend/
│   ├── agent.py               # Main LangGraph agent with tools
│   ├── cypher_subagent.py     # Specialized Cypher query generator
│   ├── neo4j_utils.py         # Neo4j connection utilities
│   ├── server.py              # FastAPI server
│   ├── .env                   # Environment configuration (not in git)
│   └── .env.example           # Environment variables template
├── frontend/
│   └── index.html             # Web interface
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── test_connection.py         # Neo4j connection test
├── test_cypher_subagent.py    # Cypher subagent tests
└── README.md                  # This file
```

## Running the Application

### Start the Server

From the `agentic_graph_rag` directory:

```bash
cd backend
python server.py
```

The server will start on `http://localhost:8000` (or the port specified in .env)

You should see:
```
╔══════════════════════════════════════════════════╗
║     Synthea Chatbot Server Starting...          ║
╚══════════════════════════════════════════════════╝

Server will be available at: http://localhost:8000
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
Chat Interface: http://localhost:8000/app
```

### Access the Application

- **Web Interface**: http://localhost:8000/app
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Usage

### Query Types

The chatbot supports two types of queries:

#### 1. Standard Queries (Pre-built Tools)
For common patient-specific questions, the agent uses pre-built tools with flexible name matching:

**Patient Procedures**
- "What procedures has Ethan had?" (matches first name)
- "Show me all procedures for John Smith" (matches full name)
- "What procedures did Smith have?" (matches last name)

**Patient Conditions**
- "What conditions does Sarah have?"
- "List all diagnoses for John Smith"
- "What's wrong with Thompson?" (last name match)

**Patient Medications**
- "What medications is David taking?"
- "Show me drugs prescribed to Maria Johnson"
- "What is Brown taking?" (last name match)

**Patient Search**
- "Find patients named John"
- "Search for patients with last name Smith"
- "Show me patients named Maria Garcia"

**Patient Encounters**
- "What encounters has Ethan had?"
- "Show me hospital visits for John Smith"
- "List encounters for Williams" (last name match)

**Name Matching Features:**
- **First name only**: "Ethan" → matches any patient with first or last name containing "Ethan"
- **Last name only**: "Smith" → matches any patient with first or last name containing "Smith"
- **Full name**: "John Smith" → matches patients where full name contains "John Smith"

#### 2. Complex Queries (Cypher Subagent)
For analytics, aggregations, and complex questions, the agent uses a specialized Cypher subagent that constructs custom queries:

**Provider Analytics**
- "Which providers treated the most patients?"
- "Show me providers specializing in cardiology"
- "What's the average patient load per provider?"

**Procedure Analytics**
- "What's the most common procedure performed?"
- "How many procedures were done in 2023?"
- "Which procedures cost the most?"

**Organization Queries**
- "Which organizations have the highest patient volume?"
- "Show me all healthcare organizations in Boston"

**Time-Based Analysis**
- "How many emergency visits were there in 2023?"
- "Show me patient visits by month for last year"
- "What's the trend in wellness visits over time?"

**Multi-Entity Queries**
- "Show me patients who had both diabetes and hypertension"
- "Which providers work at multiple organizations?"
- "Find patients with more than 5 emergency visits"

**Aggregations**
- "What's the average age of patients with heart conditions?"
- "Count encounters by type"
- "Show me the distribution of conditions across patients"

### How It Works

#### Standard Query Flow
1. **User Query**: You type a natural language question
2. **Main Agent Analysis**: The LangGraph agent analyzes the query
3. **Tool Selection**: Agent chooses appropriate pre-built tool
4. **Tool Execution**: Direct Neo4j query via ToolNode
5. **Response Generation**: LLM synthesizes results into natural language
6. **Context Maintenance**: Conversation history preserved for follow-ups

#### Complex Query Flow (with Cypher Subgraph)
1. **User Query**: You ask a complex analytical question
2. **Router Analysis**: Lightweight router LLM classifies query as "custom_analytics"
3. **Cypher Subgraph Invocation**: Main agent routes to Cypher node
4. **Cypher Generation**: Specialized Cypher subgraph constructs custom query
   - Subgraph has detailed knowledge of Synthea schema (400+ lines)
   - Generates optimized, safe queries with proper LIMIT clauses
   - Uses structured output for consistency
   - Explains what the query does
5. **Query Execution**: Custom Cypher executed against Neo4j with error handling
6. **Result Formatting**: Results formatted as readable table
7. **Response Generation**: Main agent presents results with context
8. **Query Transparency**: Generated Cypher query shown to user

**Key Benefits of This Architecture:**
- Main agent stays lightweight and focused on conversation flow
- Cypher subgraph has deep schema knowledge and query expertise
- Separation of concerns: routing → query construction → execution
- Configurable LLM models for different roles (router, main agent, cypher agent)
- Custom queries are validated and executed safely with comprehensive error handling
- Query tracking for debugging and observability

## API Endpoints

### POST /chat
Submit a chat message and get a response.

**Request:**
```json
{
  "message": "What procedures has Ethan had?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "Ethan has had the following procedures...",
  "session_id": "session-id",
  "executed_queries": [
    {
      "source": "tool",
      "query": "MATCH (p:Patient)...",
      "timestamp": "2025-01-30T10:30:00Z"
    }
  ]
}
```

### POST /chat/reset
Reset conversation history for a session.

**Query Parameter:**
- `session_id` (optional): Session to reset (default: "default")

### GET /provider
Get the current LLM provider configuration.

**Response:**
```json
{
  "provider": "anthropic",
  "models": {
    "main_agent": "claude-sonnet-4-5-20250929",
    "router": "claude-haiku-4-5-20251001",
    "cypher_agent": "claude-sonnet-4-5-20250929"
  }
}
```

### POST /provider
Switch the LLM provider at runtime.

**Request:**
```json
{
  "provider": "sambanova"
}
```

**Response:**
```json
{
  "provider": "sambanova",
  "message": "Switched to sambanova provider",
  "models": {
    "main_agent": "DeepSeek-V3.1",
    "router": "DeepSeek-V3.1",
    "cypher_agent": "DeepSeek-V3.1"
  }
}
```

### GET /health
Check server health status.

### GET /sessions
List all active sessions.

## Database Schema

The Synthea database follows this structure:

**Entities:**
- Patient: Demographic information
- Encounter: Healthcare visits
- Procedure: Medical procedures
- Condition: Medical conditions/diagnoses
- Drug: Medications
- Provider: Healthcare providers
- Organization: Healthcare organizations

**Key Relationships:**
- (Patient)-[:HAS_ENCOUNTER]->(Encounter)
- (Encounter)-[:HAS_PROCEDURE]->(Procedure)
- (Encounter)-[:HAS_CONDITION]->(Condition)
- (Encounter)-[:HAS_DRUG]->(Drug)

## Agent Tools & Architecture

The main LangGraph agent uses intelligent routing to direct queries to the appropriate handler:

### Query Routing
The agent classifies each query into one of four routes:
1. **standard_tools**: Patient-specific queries → Uses pre-built tools
2. **custom_analytics**: Complex analytical queries → Routes to Cypher subgraph
3. **conversational**: Greetings, thanks, farewells → Direct response
4. **schema_info**: Database structure questions → Uses schema tool

### Standard Tools (Pre-built Queries)
1. **get_patient_procedures**: Retrieve patient procedures (with configurable limit)
2. **get_patient_conditions**: Retrieve patient conditions (with configurable limit)
3. **get_patient_medications**: Retrieve patient medications (with configurable limit)
4. **get_patient_encounters**: Retrieve patient encounters (with configurable limit)
5. **search_patients**: Search for patients by name
6. **get_database_schema**: Get database schema information

### Cypher Subgraph (Complex Analytics)
For complex analytical queries, the agent routes to a separate Cypher subgraph that:
- Uses a specialized LLM agent (configured via CYPHER_AGENT_MODEL) with comprehensive Synthea schema knowledge
- Constructs safe, optimized Cypher queries with proper LIMIT clauses
- Handles analytics, aggregations, and multi-entity queries
- Provides query explanations and transparency
- Returns formatted results to the main agent

### Graph Structure
```
User Query → Agent (Routing)
             ├── standard_tools → Tool Node → Agent
             ├── custom_analytics → Cypher Subgraph → Agent
             ├── conversational → Direct Response → END
             └── schema_info → Schema Tool → Agent
```

## Cypher Subagent Technical Guide

### Overview

The Cypher Subagent is a specialized AI agent that constructs custom Cypher queries for complex questions that don't fit the pre-built query tools. It operates as a sub-component of the main LangGraph agent.

### Architecture Flow

```
User Question
    ↓
Router LLM (Classify Query Intent)
    ↓
[standard_tools | custom_analytics | conversational | schema_info]
    ↓ custom_analytics
Cypher Subgraph (Configured LLM)
    ↓
[Generate Cypher Query]
    ↓
Neo4j Database
    ↓
Results → Main Agent → User
```

### Key Features

#### 1. Deep Schema Knowledge
The Cypher subagent has a comprehensive 400+ line system prompt that includes:
- Complete node type definitions with all properties
- All relationship types and directions
- Multi-label node handling
- Common query patterns
- Cypher syntax guidelines
- Database statistics

#### 2. Safe Query Generation
- **Always includes LIMIT** clauses (default 50)
- **Validates** patient identification patterns
- **Optimizes** for performance
- **Explains** what the query does
- **Formats** results as readable tables

#### 3. Intelligent Tool Routing
The main agent knows when to delegate to the subagent:
- Analytics and aggregations
- Provider/organization queries
- Time-based analysis
- Multi-entity relationships
- Complex filtering/grouping

### When the Subagraph is Used

#### Routes to Cypher Subgraph (custom_analytics)
- "Which providers treated the most patients?"
- "What's the most common procedure?"
- "How many emergency visits in 2023?"
- "Show patients with both diabetes and hypertension"
- "Average age of patients with heart conditions"
- "Count encounters by type"

#### Routes to Standard Tools
- "What procedures has Ethan766 had?" → get_patient_procedures
- "List medications for John" → get_patient_medications
- "Find patients named Smith" → search_patients

### Schema Knowledge in Subagent

The subagent knows about:

**Node Types (26 labels)**
- Patient (5,885 records)
- Encounter (1,274,720 records)
- Procedure, Condition, Drug
- Provider, Organization
- And 19 more...

**Relationships (18 types)**
- HAS_ENCOUNTER (1,274,720)
- HAS_DRUG (569,538)
- HAS_DIAGNOSIS (517,612)
- And 15 more...

**Properties for Each Node**
Example - Patient node:
- id, firstName, lastName
- birthDate, age
- expenses, income
- city, county, location
- And more...

**Multi-Label Handling**
Encounters have type labels:
- Base: Encounter
- Types: Ambulatory, Emergency, Inpatient, Wellness, etc.

Query pattern: `MATCH (e:Emergency)` or `WHERE labels(e) CONTAINS 'Emergency'`

### Query Generation Process

#### Step 1: Analyze Question
```
User: "Which providers treated the most patients?"
```

#### Step 2: Generate Cypher
```cypher
MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_PROVIDER]->(prov:Provider)
RETURN
    prov.name AS provider_name,
    prov.speciality AS specialty,
    count(DISTINCT p) AS patient_count
ORDER BY patient_count DESC
LIMIT 20
```

#### Step 3: Explain Query
```
This query finds all providers, counts distinct patients they've treated,
and orders by patient count to show the busiest providers.
```

#### Step 4: Execute & Format
```
Query Results (20 records):

provider_name | specialty | patient_count
------------------------------------------
Dr. Smith     | Cardiology | 145
Dr. Johnson   | General    | 132
...

Cypher Query Used:
[query shown above]
```

### Implementation Details

#### File Structure
```
backend/
├── cypher_subagent.py       # Cypher query generator subgraph
├── agent.py                 # Main agent with routing logic
└── neo4j_utils.py          # Database utilities and execute_custom_cypher method
```

#### Key Components

**1. Cypher Subagent** ([cypher_subagent.py](backend/cypher_subagent.py))
```python
def query_cypher_subgraph(user_question: str):
    """
    Query the Cypher subgraph to generate and execute a custom Cypher query.
    Uses configured CYPHER_AGENT_MODEL with comprehensive schema knowledge.
    Returns structured output with query, explanation, and results.
    """
```

**2. Cypher Node** ([agent.py](backend/agent.py))
```python
def cypher_node(state: AgentState) -> dict:
    """
    LangGraph node that handles custom analytics queries.
    1. Extracts user question from state
    2. Calls Cypher subgraph to generate and execute query
    3. Returns results as AI message back to agent
    """
```

**3. Custom Cypher Executor** ([neo4j_utils.py](backend/neo4j_utils.py))
```python
def execute_custom_cypher(self, cypher_query: str):
    # Safe execution with error handling
    # Returns: {success, results, message, result_count}
```

### Benefits of This Architecture

#### 1. Separation of Concerns
- **Main Agent**: Routes queries, maintains conversation, synthesizes responses
- **Cypher Subagent**: Expert in query construction, schema knowledge

#### 2. Cost Optimization
- Standard queries: Fast, minimal LLM usage (pre-built queries)
- Complex queries: Uses Cypher agent LLM only when needed
- Router: Lightweight model (Haiku) for classification

#### 3. Maintainability
- Schema changes: Update subagent prompt only
- New query types: Add examples to subagent
- Main agent: Stays simple and focused

#### 4. Safety
- Queries validated before execution
- LIMIT clauses enforced
- Error handling at multiple levels

#### 5. Transparency
- Generated queries shown to user
- Query explanation provided
- User can learn Cypher patterns

### Testing the Subagent

```bash
# Ensure environment is configured (backend/.env)
# Then run tests
python test_cypher_subagent.py
```

Tests verify:
1. Query generation works
2. Cypher execution works
3. End-to-end flow works

#### Example Test Output
```
Testing Cypher Subagent - Query Generation
------------------------------------------------------------
Question: Which providers treated the most patients?
✓ Query generated successfully

Cypher Query:
MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)-[:HAS_PROVIDER]->(prov:Provider)
RETURN prov.name AS provider_name, count(DISTINCT p) AS patient_count
ORDER BY patient_count DESC
LIMIT 20

Explanation:
Finds all providers and counts distinct patients they've treated.
```

### Extending the Subagent

#### Adding New Query Patterns

Edit [cypher_subagent.py](backend/cypher_subagent.py) and update `SYNTHEA_SCHEMA_DETAILED`:

```python
SYNTHEA_SCHEMA_DETAILED = """
...existing schema...

## NEW QUERY PATTERNS

### Finding Care Plans
11. Get care plans for patient:
    MATCH (p:Patient)-[:HAS_ENCOUNTER]->(e:Encounter)<-[:CARE_PLAN_START]-(cp:CarePlan)
    WHERE p.firstName = 'Name'
    RETURN cp.code, e.date
"""
```

#### Adding Safety Rules

Update the system prompt in `generate_cypher_query()`:

```python
## ADDITIONAL SAFETY RULES

8. **Never delete data** - Only use MATCH and RETURN
9. **Validate dates** - Use datetime() function
10. **Check for null** - Use WHERE field IS NOT NULL
```

### Performance Characteristics

- **Query Generation**: 2-5 seconds (Cypher agent LLM call)
- **Query Execution**: <1 second (Neo4j)
- **Total Time**: 3-6 seconds for complex queries
- **Standard Queries**: 2-5 seconds (direct tool execution)

### Security Considerations

1. **No User Input in Cypher**: User questions passed to LLM, not directly to Cypher
2. **Read-Only Queries**: Subagent instructed to only generate MATCH/RETURN
3. **LIMIT Enforcement**: Prevents large result sets
4. **Error Handling**: Database errors don't expose internals
5. **Query Validation**: Subagent validates before execution

### Future Enhancements

Possible improvements:
1. **Query Caching**: Cache common query patterns
2. **Query Optimization**: Learn from slow queries
3. **Multi-Step Queries**: Break complex questions into multiple queries
4. **Query Validation**: Pre-validate with Cypher parser
5. **Feedback Loop**: Learn from successful/failed queries
6. **Query Templates**: Store and reuse common patterns

### Summary

The Cypher Subagent provides a powerful extension to the chatbot, enabling it to answer complex analytical questions while maintaining:
- **Safety**: Validated, read-only queries
- **Efficiency**: Only used when needed
- **Transparency**: Queries shown to user
- **Maintainability**: Separate from main agent
- **Extensibility**: Easy to add new patterns

This dual-agent architecture combines the best of both worlds: fast pre-built queries for common questions, and flexible custom queries for complex analysis.

## Deployment

### Deploying to Another Computer

1. **Copy the entire `agentic_graph_rag` directory**

2. **Install dependencies** (as described in Installation section)

3. **Configure environment variables** in `.env` file

4. **Ensure Neo4j access** - Update NEO4J_URI if database is on a different host

5. **Run the server**

### Production Considerations

**Infrastructure**:
- Use a production ASGI server (Gunicorn with Uvicorn workers)
- Set up proper CORS origins in `server.py`
- Use HTTPS/SSL
- Configure reverse proxy (Nginx/Caddy)

**Security & Access**:
- Implement authentication/authorization
- Add rate limiting (per API key/IP)
- API key management
- Input validation/sanitization

**Operations**:
- Environment-specific configuration
- Logging and monitoring
- Database connection pooling
- Query result caching
- Cost tracking (LLM API usage)

**Current State**: ✅ Functional prototype with error handling, session management, and complete documentation

## What Was Built

### Backend Components

#### [backend/neo4j_utils.py](backend/neo4j_utils.py)
- Neo4j connection management
- Pre-built query functions for common operations:
  - `get_patient_procedures()` - Retrieve patient procedures
  - `get_patient_conditions()` - Retrieve patient conditions
  - `get_patient_medications()` - Retrieve patient medications
  - `get_patient_encounters()` - Retrieve patient encounters
  - `search_patients()` - Search for patients by name
  - `get_database_schema()` - Get schema information
  - `execute_custom_cypher()` - Execute custom Cypher queries

#### [backend/agent.py](backend/agent.py)
- LangGraph agent with ToolNode integration
- 6 standard tools wrapped from neo4j_utils functions
- Intelligent query routing (standard tools, custom analytics, conversational, schema)
- Cypher subgraph integration for complex queries
- Conversation state management with query tracking
- Configurable LLM support (Anthropic, SambaNova)
- Iteration limits and reflection loops

#### [backend/cypher_subagent.py](backend/cypher_subagent.py)
- Specialized Cypher query generator
- Comprehensive Synthea schema knowledge (400+ lines)
- LLM-powered query construction (uses configured CYPHER_AGENT_MODEL)
- Safe query validation and execution
- Structured output with query explanation

#### [backend/server.py](backend/server.py)
- FastAPI REST API server
- CORS configuration for frontend
- Session-based conversation management
- Health check endpoints
- Static file serving for frontend

### Frontend

#### [frontend/index.html](frontend/index.html)
- Single-page application (no framework required)
- Modern gradient UI design
- Real-time chat interface
- Example queries for quick testing
- Session management
- Status indicators
- Responsive design

### Configuration & Documentation

- **requirements.txt** - All Python dependencies
- **backend/.env.example** - Environment configuration template
- **backend/.env** - Environment configuration (not in git)
- **.gitignore** - Git ignore rules
- **README.md** - Comprehensive documentation (this file)
- **start_server.sh** - Automated startup script
- **test_connection.py** - Connection testing utility
- **test_cypher_subagent.py** - Cypher subagent tests

## Key Design Decisions

1. **LangGraph with ToolNode**: Provides structured agent workflow with automatic tool routing
2. **Pure HTML/JS Frontend**: Minimal dependencies, easy to understand and modify
3. **Session-based Conversations**: Maintains context without database overhead
4. **FastAPI**: Modern, fast, with automatic API documentation
5. **Direct Neo4j Queries**: No ORM overhead, optimal for graph traversal
6. **Environment Variables**: Secure configuration management
7. **Dual-Agent Architecture**: Main agent for routing, Cypher subagent for complex queries

## Extensibility

### Easy to Add
- **New Tools**: Add functions in [agent.py](backend/agent.py) with `@tool` decorator
- **New Queries**: Add methods to [neo4j_utils.py](backend/neo4j_utils.py)
- **UI Features**: Modify [frontend/index.html](frontend/index.html)
- **API Endpoints**: Add routes to [server.py](backend/server.py)

### Integration Points
- **Authentication**: Add middleware in FastAPI
- **Additional LLMs**: Swap OpenAI for other providers
- **Frontend Frameworks**: Replace HTML with React/Vue/Svelte
- **Deployment**: Docker, Kubernetes, cloud platforms

## Performance & Cost

### Performance Characteristics
- **Standard Query Response**: 2-5 seconds (depends on LLM)
- **Complex Query Response**: 3-6 seconds (includes Cypher generation)
- **Database Queries**: Sub-second (Neo4j graph traversal)
- **Concurrent Users**: Depends on server resources (FastAPI is async)
- **Memory Usage**: ~200MB base + conversation histories

### Cost Considerations
- **Anthropic Claude API**:
  - Claude Sonnet 4.5: ~$0.003-0.015 per query (main/cypher agents)
  - Claude Haiku 4.5: ~$0.0001-0.001 per query (router)
- **SambaNova**: Varies based on model and usage
- **Neo4j**: Free (Community Edition) or AuraDB pricing
- **Hosting**: Minimal (can run on small VPS)

## Testing & Validation

- ✅ Neo4j connection tested successfully
- ✅ Query functions tested with real data
- ✅ Found 5885 patients in database
- ✅ Successfully retrieved procedures for test patients
- ✅ Agent initialization working
- ✅ API endpoints functional
- ✅ Cypher subagent query generation tested

## Troubleshooting

### Common Issues

**1. Cannot connect to Neo4j**
- Verify Neo4j is running: `neo4j status`
- Check connection details in `.env`
- Ensure database name is correct (synthea-sample)

**2. LLM API errors**
- Verify your API key is set correctly in `.env`:
  - `ANTHROPIC_API_DEV_KEY` for Anthropic
  - `SAMBANOVA_API_KEY` for SambaNova
- Check PROVIDER is set correctly (`anthropic` or `sambanova`)
- Verify API key has sufficient credits/access
- Ensure no rate limiting issues

**3. Module import errors**
- Verify virtual environment is activated
- Reinstall dependencies: `uv pip install -r requirements.txt`

**4. Frontend cannot connect to backend**
- Ensure backend server is running
- Check CORS settings in `server.py`
- Verify API_BASE_URL in `index.html` matches your server URL

## Development

### Running in Development Mode

The server runs in reload mode by default, automatically restarting on code changes.

### Adding New Tools

1. Define a new tool function in [agent.py](backend/agent.py:90)
2. Add the `@tool` decorator
3. Add to the `tools` list
4. The agent will automatically discover and use it

### Customizing the UI

Edit [frontend/index.html](frontend/index.html:1) to customize:
- Colors and styling (CSS)
- Layout and components (HTML)
- Behavior and API calls (JavaScript)

## Technologies Used

### Core Framework
- **LangGraph**: Agent workflow orchestration with graph-based routing
- **LangChain**: LLM framework and tool abstractions
- **FastAPI**: Modern async web framework
- **Neo4j Driver**: Graph database connectivity
- **Uvicorn**: ASGI server

### LLM Providers (Configurable)
- **Anthropic Claude**: Sonnet 4.5 (main/cypher agents), Haiku 4.5 (router)
- **SambaNova**: DeepSeek-V3.1 (alternative provider)

### Additional Libraries
- **LangSmith**: Tracing and observability
- **Pydantic**: Data validation and structured outputs
- **Python-dotenv**: Environment configuration

## Acknowledgments & Credits

- **Database**: Synthea synthetic patient data from Patient Journey demo notebooks
- **Reference**: patientJourney_graphEDA.ipynb and reference_notebooks/neoUtils.py
- **Framework**: LangGraph by LangChain
- **LLMs**: Anthropic Claude (Sonnet 4.5, Haiku 4.5) and SambaNova DeepSeek-V3.1

## License

This project is provided as-is for educational and development purposes.

---

**Status**: ✅ Production-Ready Prototype
**Version**: 1.0.0
**Last Updated**: January 2025
