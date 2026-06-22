from typing import TypedDict, Literal, List, Annotated
from operator import add
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents.orchestrator import orchestrator
from agents.mini_orchestators import reasoning_orchestrator, coding_orchestrator, math_orchestrator
from agents.evaluator import evaluator
from agents.recovery import recovery
from agents.final_synthesizer import final_synthesizer
from piplines.conditions import route_to_domain, route_after_eval

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

# llm schema for domain identification
class DomainRoute(TypedDict):

    domain: Literal["reasoning", "coding", "math"]

llm_domain_identifier = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0
)

llm_domain_identifier = llm_domain_identifier.with_structured_output(DomainRoute)


# StateGraph wiring

graph = StateGraph(AgentState)

# nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("reasoning_orchestrator", reasoning_orchestrator)
graph.add_node("coding_orchestrator", coding_orchestrator)
graph.add_node("math_orchestrator", math_orchestrator)
graph.add_node("evaluator", evaluator)
graph.add_node("recovery", recovery)
graph.add_node("final_synthesizer", final_synthesizer)

# entry
graph.add_edge(START, "orchestrator")

# main orchestrator routes to a domain mini-orchestrator
graph.add_conditional_edges("orchestrator", route_to_domain, {
    "reasoning": "reasoning_orchestrator",
    "coding": "coding_orchestrator",
    "math": "math_orchestrator",
})

# each domain mini-orchestrator hands its synthesized output to the evaluator
graph.add_edge("reasoning_orchestrator", "evaluator")
graph.add_edge("coding_orchestrator", "evaluator")
graph.add_edge("math_orchestrator", "evaluator")

# evaluator routes to recovery on failure, else to the final synthesizer
graph.add_conditional_edges("evaluator", route_after_eval, {
    "recovery": "recovery",
    "final_synthesizer": "final_synthesizer",
})

# recovery loops back to the orchestrator to re-attempt
graph.add_edge("recovery", "orchestrator")

# final synthesizer ends the run
graph.add_edge("final_synthesizer", END)

app = graph.compile()
