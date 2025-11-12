from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent_setup import get_multi_agent_router
import logging
import traceback

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure cors
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)



# Initialize the multi-agent router once at startup
multi_agent_router = get_multi_agent_router()

# Define the data model for the incoming request body
class Query(BaseModel):
    question: str

@app.post("/ask_database")
async def ask_database(query: Query):
    """
    An API endpoint to ask questions using a multi-agent system.
    The router determines which specialized agent to use.
    """
    try:
        # Invoke the multi-agent router with the user's question
        response = multi_agent_router.invoke({"input": query.question})
        logger.info(f"Response from chain: {response}")

        # Attempt to extract the output key robustly
        final_answer = None
        
        # Method 1: Check for known keys ('answer' and 'text')
        if 'output' in response:
            final_answer = response['output']
        elif 'text' in response:
            final_answer = response['text']
        
        # Method 2 (Fallback): If the response is a dictionary with only one other key, use that
        if not final_answer and isinstance(response, dict) and len(response) == 1:
            key_name = next(iter(response))
            # Ensure it's not the 'input' key that might be returned
            if key_name != 'input':
                final_answer = response[key_name]
        
        if final_answer:
            return {"response": final_answer}
        else:
            logger.error(f"Missing a valid output key in chain response: {response}")
            raise HTTPException(status_code=500, detail="Missing expected output from agent.")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

