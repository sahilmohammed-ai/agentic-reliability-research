# parallel sub-agents for the math mini-orchestrator.

DOMAIN = "math"


def math_agent_1(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "math_agent_1", "output": "", "domain": DOMAIN}]}


def math_agent_2(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "math_agent_2", "output": "", "domain": DOMAIN}]}


def math_agent_3(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "math_agent_3", "output": "", "domain": DOMAIN}]}
