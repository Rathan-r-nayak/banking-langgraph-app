from typing import List, Literal
from pydantic import BaseModel, Field

class Task(BaseModel):
    task_id: str = Field(description="A unique identifier for the task (e.g., 'task_1', 'task_2').")
    objective: str = Field(
        description="One sentence describing exactly what this worker must extract or generate based on the documents."
    )
    description: str = Field(description="A specific, actionable instruction for the worker agent to execute. Do not combine multiple actions into one task.")
    
    # ADD THIS LINE: This forces the LLM to choose the correct agent
    agent: Literal["account_agent", "transaction_agent", "knowledge_agent"] = Field(
        description="The specialized agent assigned to execute this task."
    )

class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="A list of distinct, parallel tasks required to answer the user's overall request.")