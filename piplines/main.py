from typing import TypedDict, Literal, List, Annotated
from operator import add
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# nested dict: failure signal schema
class FailureSignal(TypedDict):
    detected: bool
    failure_type: str  # maps to MAST taxonomy
    confidence: float
    severity: Literal["low", "medium", "high"] 

# nested dict: subagent output schema
class SubAgentOutput(TypedDict):
    agent_id: str
    output: str
    domain: str

# nested dict: recovery action record schema
class ActionRecord(TypedDict):
    action: str
    outcome: str  # "success" | "failure"
    attempt: int

# graph schema
class AgentState(TypedDict):

    task: str
    task_id: str
    domain: Literal["reasoning", "coding", "math"] # only 3 possible domains for routing
    attempt_count: int
    sub_agent_outputs: Annotated[List[SubAgentOutput], add]
    synthesized_output: str
    failure_signal: FailureSignal # type, confidence, severity
    action_history: Annotated[List[ActionRecord], add]
    final_output: str
    benchmark_source: str

# llm schema for domain identification
class DomainRoute(TypedDict):

    domain: Literal["reasoning", "coding", "math"]

llm_domain_identifier = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0
)

llm_domain_identifier = llm_domain_identifier.with_structured_output(DomainRoute)