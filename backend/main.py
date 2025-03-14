import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from query_engine import query_dataset  # Import the query function

# Initialize FastAPI app
app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
def query_endpoint(request: QueryRequest):
    """Handles incoming user queries and returns dataset results."""
    response = query_dataset(request.query)
    return {"response": response}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
