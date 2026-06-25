from piplines.main import app

# one hardcoded GSM8K task — P0 goal is just a real, visible LangSmith trace
task = {
    "task": (
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April and May?"
    ),
    "task_id": "gsm8k_p0_001",
    "domain": "math",
    "attempt_count": 0,
    "sub_agent_outputs": [],
    "synthesized_output": "",
    "failure_signal": {"detected": False, "failure_type": "", "confidence": 0.0, "severity": "low"},
    "action_history": [],
    "final_output": "",
    "benchmark_source": "gsm8k",
}

if __name__ == "__main__":
    result = app.invoke(task)
    print(result["final_output"])
