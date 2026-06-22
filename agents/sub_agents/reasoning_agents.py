# parallel sub-agents for the reasoning mini-orchestrator.

DOMAIN = "reasoning"


def reasoning_agent_1(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "reasoning_agent_1", "output": "", "domain": DOMAIN}]}


def reasoning_agent_2(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "reasoning_agent_2", "output": "", "domain": DOMAIN}]}


def reasoning_agent_3(state: dict) -> dict:
    
    return {"sub_agent_outputs": [{"agent_id": "reasoning_agent_3", "output": "", "domain": DOMAIN}]}
