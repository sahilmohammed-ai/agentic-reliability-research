from typing import TypedDict, Literal, List, Annotated
from operator import add
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents.worker import worker

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
    outcome: str
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

# StateGraph wiring — P0: single generic worker, no orchestrator/evaluator/recovery loop yet

graph = StateGraph(AgentState)

graph.add_node("worker", worker)

graph.add_edge(START, "worker")
graph.add_edge("worker", END)

app = graph.compile()
