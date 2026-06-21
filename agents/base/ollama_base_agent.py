from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# generic base agent for Ollama llm response
def ollama_base_agent(state: dict, system_prompt: str, llm: ChatOllama) -> dict:
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['input'])
    ]

    response = llm.invoke(messages)

    return {"output": response.content}

