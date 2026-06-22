def route_to_domain(state: dict) -> str:
    return state["domain"]   # "reasoning" | "coding" | "math"

def route_after_eval(state: dict) -> str:
    return "recovery" if state["failure_signal"]["detected"] else "final_synthesizer"
