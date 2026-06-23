from pydantic import BaseModel, Field
from typing import Dict, Optional

class ToolResponse(BaseModel):
    tool_name : Optional[str]
    search_input : str = Field(description="Seach input for tool")
    result : Optional[Dict]