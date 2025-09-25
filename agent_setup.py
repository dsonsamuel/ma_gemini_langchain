from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain.llms import VertexAI
from langchain.schema import SystemMessage
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph, START, MessagesState
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Initialize the output parser
output_parser = CommaSeparatedListOutputParser()

# Retrieve format instructions for the output parser
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# Replace with your BigQuery project ID and dataset
project_id = 'playpen-355dd5'
dataset_id = 'GemCore_Payments'

# A shared LLM instance
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

def get_sql_agent():
    """Initializes and returns a LangChain SQL agent for BigQuery."""
    bigquery_db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}?credentials_path=bq_sa.json")
    toolkit = SQLDatabaseToolkit(db=bigquery_db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return sql_agent

def get_general_agent():
    """Initializes and returns a general purpose LLM chain."""
    prompt = PromptTemplate(
        template="You are a helpful assistant. Greet the user and handoff to the next agent with user {input}..",
    )
    return LLMChain(llm=llm, prompt=prompt)
# agent_setup.py (continued)

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()

def get_multi_agent_router():
    """Creates a router chain to dispatch queries to the correct sub-agent."""
    sql_agent = get_sql_agent()
    general_agent = get_general_agent()

    # Define a description for each agent
    destinations = [
        "sql_agent: For answering questions about the company's BigQuery database, such as employee records or sales figures.Provide your response with a JSON format having keys 'answer'",
        "general_agent: For answering general questions not related to the database.",
    ]

    # Create the prompt for the LLM to act as a router
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations='\n'.join(destinations))
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser = json_parser,
        #output_parser = json_parser,
    )

    
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    # Define the chains the router can call
    destination_chains = {
        "sql_agent": sql_agent,
        "general_agent": general_agent,
    }

    # Create the final multi-prompt chain
    multi_agent_router = SequentialChain(
        chains=[general_agent,sql_agent],
        input_variables=["input"],
        verbose=True,
    )
    return multi_agent_router
multi_agent_router = get_multi_agent_router()

