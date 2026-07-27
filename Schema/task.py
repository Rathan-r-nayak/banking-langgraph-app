from typing import List
from pydantic import BaseModel, Field

class Task(BaseModel):
    task_id: str = Field(description="A unique identifier for the task (e.g., 'task_1', 'task_2').")
    objective: str = Field(
        description="One sentence describing exactly what this worker must extract or generate based on the documents."
    )
    description: str = Field(description="A specific, actionable instruction for the worker agent to execute. Do not combine multiple actions into one task.")


class TaskPlan(BaseModel):
    tasks: List[Task] = Field(description="A list of distinct, parallel tasks required to answer the user's overall request.")