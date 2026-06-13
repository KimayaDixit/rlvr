import json

with open("artifacts/eval_qwen_vl_all_tasks/trajectories.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        if "messages" in ep["task_id"] or "schedule" in ep["task_id"]:
            print("=== TASK:", ep["task_id"], "===")
            for s in ep["trajectory"]:
                action = s["action"]
                error = s["last_action_error"][:80]
                print(f"  step {s['step']}: {action} | error: {error}")
            print()
