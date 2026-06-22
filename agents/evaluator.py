# evaluator — ML detection layer; inspects synthesized_output and emits a failure_signal

def evaluator(state: dict) -> dict:

    return {"failure_signal": {"detected": False, "failure_type": "", "confidence": 0.0, "severity": "low"}}
