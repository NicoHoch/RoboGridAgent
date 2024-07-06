from langchain_core.pydantic_v1 import BaseModel, Field


class Move(BaseModel):
    direction: str = Field(description="The direction to take")
    steps: int = Field(description="The number of fields to move")


class IsFinished(BaseModel):
    finished: bool = Field(description="If the robot has reached the flag or not")
