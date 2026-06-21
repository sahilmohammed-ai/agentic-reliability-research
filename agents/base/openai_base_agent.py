from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# generic base agent for OpenAI llm response
def openai_base_agent(state: dict, system_prompt: str, llm: ChatOpenAI) -> dict:
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['input'])
    ]

    response = llm.invoke(messages)

    return {"output": response.content}

