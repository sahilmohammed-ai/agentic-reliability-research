from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# generic base agent for Anthropic llm response
def anthropic_base_agent(state: dict, system_prompt: str, llm: ChatAnthropic) -> dict:
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['input'])
    ]

    response = llm.invoke(messages)

    return {"output": response.content}