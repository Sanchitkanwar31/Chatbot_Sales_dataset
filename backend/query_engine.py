import pandas as pd
import os
import time
import google.api_core
import google.generativeai as genai
import re
from fastapi import FastAPI
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel
from functools import lru_cache  # ✅ Caches dataset schema to reduce API calls

# define Google Gemini API Key (Replace with your actual key)
GOOGLE_API_KEY = " API Key"

# configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# load CSV file into Pandas DataFrame
csv_path = os.path.join(os.path.dirname(__file__), "../dataset/orders.csv")

# check dataset file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file not found: {csv_path}")

df = pd.read_csv(csv_path, encoding="latin1")


app = FastAPI()

# LangChain Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

@lru_cache(maxsize=10)
def get_dataset_columns():
    """Returns column names to reduce API calls."""
    return list(df.columns)

def generate_pandas_query(user_query):
    """Generates a Pandas query from user input using Gemini AI."""
    prompt = f"""
    The dataset is stored in a Pandas DataFrame named `df` with columns: {', '.join(get_dataset_columns())}.
    
    User Query: "{user_query}"
    
    Generate **only a valid Pandas query** (without explanations, print statements, or extra Python code).
    
    Example Input: "Show total sales for last month."
    Example Output: df[df['OrderDate'] >= '2024-02-01']['TotalSales'].sum()
    
    Return only the Pandas query.
    """

    for attempt in range(3):  #  Retry mechanism
        try:
            response = llm.invoke(prompt)

            # extract valid Pandas query from AIMessage object
            pandas_query = re.sub(r"```.*?\n|\n```", "", response.content.strip())

            # Ensure AI generates a valid Pandas query
            if "df.query(" not in pandas_query and "df[" not in pandas_query and "df." not in pandas_query:
                return "Error: AI generated an invalid query."

            return pandas_query
        except google.api_core.exceptions.ResourceExhausted:
            print(f"API quota exceeded. Retrying in {2**attempt} seconds...")
            time.sleep(2**attempt)
    return "Error: API limit reached. Try again later."

def query_dataset(user_query):
    """Executes AI-generated Pandas query on dataset."""
    try:
        pandas_query = generate_pandas_query(user_query)
        print(f"Generated Pandas Query: {pandas_query}")

        if "Error" in pandas_query:
            return {"error": "AI generated an invalid query. Try rephrasing your question."}

        # safer Query Execution (No `eval()`)
        if "query" in pandas_query:
            result = df.query(pandas_query.replace("df.query(", "").strip(")"))
        else:
            result = eval(pandas_query, {"df": df, "pd": pd})  

        # format Response
        if isinstance(result, (pd.Series, list, int, float)):
            return {"response": result.tolist() if hasattr(result, 'tolist') else result}

        return result.to_dict(orient="records") if not result.empty else {"message": "No matching records found."}

    except SyntaxError:
        return {"error": "Syntax error in AI-generated query. Try rephrasing your question."}
    except Exception as e:
        return {"error": str(e)}

# 🔧 Create LangChain Tool for Querying Dataset
query_tool = Tool(
    name="query_dataset",
    func=query_dataset,
    description="Fetch sales data dynamically based on user queries."
)

# AI Agent with Tools
agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Request Model
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
def query_endpoint(request: QueryRequest):
    """API endpoint for querying the dataset."""
    response = query_dataset(request.query)
    return {"response": response}
