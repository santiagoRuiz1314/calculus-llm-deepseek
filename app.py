from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import your OllamaCalculusSolver class
from calculus_solver import OllamaCalculusSolver

app = FastAPI(title="Calculus Solver API", 
              description="API for solving calculus problems using Ollama models",
              version="1.0.0")

# Add CORS middleware to allow OpenWebUI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your OpenWebUI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Pydantic models for request/response validation
class ProblemRequest(BaseModel):
    problem: str
    temperature: Optional[float] = 0.7
    model: Optional[str] = "deepseek-r1"

class SolutionResponse(BaseModel):
    problem: str
    solution: str
    model_used: str

class ExampleRequest(BaseModel):
    problem: str
    solution: str

class StatusResponse(BaseModel):
    status: str
    message: str

# Initialize the solver (will be created per request to avoid state issues)
def get_solver(model_name="deepseek-r1", base_url="http://ollama:11434"):
    return OllamaCalculusSolver(model_name=model_name, base_url=base_url)

# API endpoints
@app.get("/")
async def root():
    return {"message": "Calculus Solver API is running. Use /docs for API documentation."}

@app.post("/solve", response_model=SolutionResponse)
async def solve_problem(request: ProblemRequest):
    try:
        solver = get_solver(model_name=request.model)
        solution = solver.solve_problem(request.problem, request.temperature)
        
        return {
            "problem": request.problem,
            "solution": solution,
            "model_used": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")

@app.post("/examples/add", response_model=StatusResponse)
async def add_example(request: ExampleRequest):
    try:
        solver = get_solver()
        solver.add_example(request.problem, request.solution)
        solver.save_examples()
        return {"status": "success", "message": f"Example added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding example: {str(e)}")

@app.get("/examples/list", response_model=List[Dict[str, str]])
async def list_examples():
    try:
        solver = get_solver()
        return solver.examples
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing examples: {str(e)}")

@app.get("/models", response_model=List[str])
async def list_models():
    try:
        import requests
        base_url = "http://ollama:11434"
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name") for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Run the API with uvicorn when file is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)