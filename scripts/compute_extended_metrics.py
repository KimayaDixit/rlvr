"""
Extended Evaluation Metrics
Computes additional metrics from existing trajectories.jsonl files.
No rerunning required — works entirely from saved artifacts.

Usage:
    .\.conda\python.exe scripts\compute_extended_metrics.py

Reads from:
    artifacts/eval_qwen_vl_all_tasks/trajectories.jsonl
    artifacts/eval_full_four_way/trajectories.jsonl
    (or any trajectories.jsonl you point it at)

Outputs:
    artifacts/extended_metrics/extended_metrics.json
    artifacts/extended_metrics/extended_metrics_summary.csv
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config — add any trajectories.jsonl paths you want to analyse
# ---------------------------------------------------------------------------

ARTIFACT_DIRS = [
    "artifacts/eval_qwen_vl_all_tasks",
    "artifacts/eval_full_four_way",
    "artifacts/eval_qwen_self_rlvr",
    "artifacts/eval_gemma4_self_rlvr",
    "artifacts/eval_gemma4_full",
]

OUTPUT_DIR = Path("artifacts/extended_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIMAL_STEPS = {
    "browsergym/social_rlvr.report.extract_tracking_code": 2,
    "browsergym/social_rlvr.gallery.aesthetic_travel_to_meera": 2,
    "browsergym/social_rlvr.messages.last_five_new_year": 15,
    "browsergym/social_rlvr.orders.priority_followup": 4,
    "browsergym/social_rlvr.schedule.design_review_shared_slot": 5,
}

ACTION_RE = re.compile(
    r"^(click|fill|select_option|noop|scroll|press|keyboard_press)\(.*\)$"
)

# ---------------------------------------------------------------------------
# Load all episodes from all artifact dirs
# ---------------------------------------------------------------------------

def load_episodes(artifact_dirs):
    all_episodes = []
    for d in artifact_dirs:
        path = Path(d) / "trajectories.jsonl"
        if not path.exists():
            print(f"  Skipping (not found): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ep = json.loads(line)
                    ep["_source"] = d
                    all_episodes.append(ep)
        print(f"  Loaded: {path} ({sum(1 for l in open(path) if l.strip())} episodes)")
    return all_episodes


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def is_valid_action(action: str) -> bool:
    """True if action is a real action (not noop fallback)."""
    if not action or action == "noop(100)":
        return False
    return bool(ACTION_RE.match(action.strip()))


def compute_metrics(episode: dict) -> dict:
    trajectory = episode.get("trajectory", [])
    policy = episode.get("policy", "unknown")
    task_id = episode.get("task_id", "unknown")
    success = episode.get("success", False)
    steps = len(trajectory)
    optimal = OPTIMAL_STEPS.get(task_id, steps)

    if steps == 0:
        return None

    actions = [s.get("action", "") for s in trajectory]
    errors = [s.get("last_action_error", "") for s in trajectory]

    # --- Metric 1: Per-task success (just success flag per episode) ---
    task_short = task_id.split(".")[-1] if "." in task_id else task_id

    # --- Metric 2: Action Validity Rate ---
    valid_count = sum(1 for a in actions if is_valid_action(a))
    action_validity_rate = round(valid_count / steps, 4) if steps > 0 else 0.0

    # --- Metric 3: First Step Success ---
    # Did the first action contribute to eventual success?
    # Proxy: was the first action a valid non-noop action?
    first_action_valid = is_valid_action(actions[0]) if actions else False

    # --- Metric 4: Action Diversity Score ---
    unique_actions = len(set(actions))
    action_diversity = round(unique_actions / steps, 4) if steps > 0 else 0.0

    # --- Metric 5: Error Recovery Rate ---
    # When step N has an error, did step N+1 use a different action?
    error_recovery_count = 0
    error_total = 0
    for i in range(len(trajectory) - 1):
        if errors[i]:  # step i had an error
            error_total += 1
            if actions[i + 1] != actions[i]:  # model changed action
                error_recovery_count += 1
    error_recovery_rate = (
        round(error_recovery_count / error_total, 4) if error_total > 0 else None
    )

    # --- Metric 6: Wasted Steps (successful episodes only) ---
    wasted_steps = (steps - optimal) if success else None

    # --- Metric 7: Stuck Loop Detection ---
    # Did the model repeat the same action 3+ times in a row?
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i - 1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    stuck_loop_detected = max_consecutive >= 3

    # --- Metric 8: Partial Progress Score ---
    # For tasks with incremental verifier messages (e.g. messages task)
    # Extract max fraction achieved from verifier messages
    partial_progress = 0.0
    for step in trajectory:
        msg = step.get("verifier_message", "")
        # e.g. "sent valid greeting to 3/5 required recipients"
        import re as re_module
        m = re_module.search(r"(\d+)/(\d+)", msg)
        if m:
            num, denom = int(m.group(1)), int(m.group(2))
            partial_progress = max(partial_progress, num / denom if denom > 0 else 0)
        if step.get("success"):
            partial_progress = 1.0
    partial_progress = round(partial_progress, 4)

    # --- Metric 9: Action Type Distribution ---
    action_types = defaultdict(int)
    for a in actions:
        if not a:
            action_types["empty"] += 1
        elif a.startswith("noop"):
            action_types["noop"] += 1
        elif a.startswith("click"):
            action_types["click"] += 1
        elif a.startswith("fill"):
            action_types["fill"] += 1
        elif a.startswith("select_option"):
            action_types["select_option"] += 1
        else:
            action_types["other"] += 1

    return {
        "policy": policy,
        "task_id": task_id,
        "task_short": task_short,
        "source": episode.get("_source", ""),
        "success": success,
        "steps": steps,
        "optimal_steps": optimal,

        # Core metrics (already in summary.csv)
        "reward": episode.get("reward", 0.0),
        "rrr": episode.get("rrr", 0.0),

        # Extended metrics
        "action_validity_rate": action_validity_rate,
        "first_action_valid": first_action_valid,
        "action_diversity_score": action_diversity,
        "error_recovery_rate": error_recovery_rate,
        "wasted_steps": wasted_steps,
        "stuck_loop_detected": stuck_loop_detected,
        "partial_progress_score": partial_progress,

        # Action type breakdown
        "n_click": action_types.get("click", 0),
        "n_fill": action_types.get("fill", 0),
        "n_select_option": action_types.get("select_option", 0),
        "n_noop": action_types.get("noop", 0),
        "n_other": action_types.get("other", 0),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics per policy
# ---------------------------------------------------------------------------

def aggregate_by_policy(metrics_list):
    by_policy = defaultdict(list)
    for m in metrics_list:
        by_policy[m["policy"]].append(m)

    summary = []
    for policy, episodes in sorted(by_policy.items()):
        n = len(episodes)

        def avg(key):
            vals = [e[key] for e in episodes if e[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def pct(key):
            vals = [e[key] for e in episodes]
            return round(sum(1 for v in vals if v) / len(vals), 4) if vals else 0.0

        # per-task success rate
        task_success = defaultdict(list)
        for e in episodes:
            task_success[e["task_short"]].append(e["success"])
        task_success_rates = {
            t: round(sum(v) / len(v), 4)
            for t, v in task_success.items()
        }

        summary.append({
            "policy": policy,
            "episodes": n,
            "overall_success_rate": avg("reward"),

            # Extended
            "mean_action_validity_rate": avg("action_validity_rate"),
            "first_action_valid_rate": pct("first_action_valid"),
            "mean_action_diversity": avg("action_diversity_score"),
            "mean_error_recovery_rate": avg("error_recovery_rate"),
            "mean_wasted_steps": avg("wasted_steps"),
            "stuck_loop_rate": pct("stuck_loop_detected"),
            "mean_partial_progress": avg("partial_progress_score"),

            # Action type averages
            "mean_clicks_per_episode": avg("n_click"),
            "mean_fills_per_episode": avg("n_fill"),
            "mean_selects_per_episode": avg("n_select_option"),
            "mean_noops_per_episode": avg("n_noop"),

            # Per-task breakdown
            **{f"success_{t}": task_success_rates.get(t, 0.0)
               for t in ["extract_tracking_code", "aesthetic_travel_to_meera",
                         "last_five_new_year", "priority_followup",
                         "design_review_shared_slot"]},
        })

    return summary


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------

def print_results(summary):
    print()
    print("=" * 80)
    print("EXTENDED METRICS SUMMARY")
    print("=" * 80)

    for row in summary:
        print(f"\nPolicy: {row['policy']}")
        print(f"  Episodes:                  {row['episodes']}")
        print(f"  Overall Success Rate:      {row['overall_success_rate']}")
        print(f"  Action Validity Rate:      {row['mean_action_validity_rate']}")
        print(f"  First Action Valid Rate:   {row['first_action_valid_rate']}")
        print(f"  Action Diversity Score:    {row['mean_action_diversity']}")
        print(f"  Error Recovery Rate:       {row['mean_error_recovery_rate']}")
        print(f"  Wasted Steps (success):    {row['mean_wasted_steps']}")
        print(f"  Stuck Loop Rate:           {row['stuck_loop_rate']}")
        print(f"  Partial Progress Score:    {row['mean_partial_progress']}")
        print(f"  Action breakdown:")
        print(f"    clicks/ep:               {row['mean_clicks_per_episode']}")
        print(f"    fills/ep:                {row['mean_fills_per_episode']}")
        print(f"    selects/ep:              {row['mean_selects_per_episode']}")
        print(f"    noops/ep:                {row['mean_noops_per_episode']}")
        print(f"  Per-task success rates:")
        for t in ["extract_tracking_code", "aesthetic_travel_to_meera",
                  "last_five_new_year", "priority_followup",
                  "design_review_shared_slot"]:
            val = row.get(f"success_{t}", "N/A")
            print(f"    {t}: {val}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading episodes...")
    episodes = load_episodes(ARTIFACT_DIRS)
    print(f"Total episodes loaded: {len(episodes)}")

    if not episodes:
        print("No episodes found. Check ARTIFACT_DIRS paths.")
        return

    print()
    print("Computing metrics...")
    metrics_list = []
    for ep in episodes:
        m = compute_metrics(ep)
        if m:
            metrics_list.append(m)

    print(f"Metrics computed for {len(metrics_list)} episodes.")

    # aggregate
    summary = aggregate_by_policy(metrics_list)

    # print
    print_results(summary)

    # save episode-level metrics
    episode_out = OUTPUT_DIR / "episode_extended_metrics.json"
    with open(episode_out, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"\nEpisode-level metrics saved to: {episode_out}")

    # save summary CSV
    summary_out = OUTPUT_DIR / "extended_metrics_summary.csv"
    if summary:
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Summary CSV saved to: {summary_out}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
