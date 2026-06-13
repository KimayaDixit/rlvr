import gymnasium as gym
import social_rlvr_web.browsergym_tasks

env = gym.make(
    "browsergym/social_rlvr.schedule.design_review_shared_slot",
    headless=True,
    slow_mo=0,
)
obs, _ = env.reset()

axtree = obs.get("axtree_object") or {}
print("=== ALL AXTREE NODES WITH REAL BIDS ===")
for node in axtree.get("nodes", []):
    bid = node.get("browsergym_id")
    if not bid:
        continue
    role = node.get("role", {})
    name = node.get("name", {})
    role_v = role.get("value", "") if isinstance(role, dict) else str(role)
    name_v = name.get("value", "") if isinstance(name, dict) else str(name)
    props = node.get("properties", [])
    checked = next((p.get("value", {}).get("value") for p in props if p.get("name") == "checked"), None)
    val = node.get("value", {})
    val_v = val.get("value", "") if isinstance(val, dict) else str(val) if val else ""
    print(f"  bid={bid} role={role_v} name=\"{name_v}\" value=\"{val_v}\" checked={checked}")

env.close()
