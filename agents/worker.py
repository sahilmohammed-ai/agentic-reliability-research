# generic domain-tagged solver — P0: single Opus call, no K-sampling, no decomposition yet

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

WORKER_SYSTEM_PROMPT = "You are a careful problem solver. Show your reasoning, then give the final answer on its own line as 'Answer: <value>'."

llm = ChatAnthropic(model="claude-opus-4-8")


def worker(state: dict) -> dict:

    messages = [
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=state["task"]),
    ]

    response = llm.invoke(messages)

    return {"synthesized_output": response.content, "final_output": response.content}
