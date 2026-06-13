import gymnasium as gym
import social_rlvr_web.browsergym_tasks

env = gym.make(
    "browsergym/social_rlvr.schedule.design_review_shared_slot",
    headless=True,
    slow_mo=0,
)
obs, _ = env.reset()

print("=== OBS KEYS ===")
for k, v in obs.items():
    if isinstance(v, str):
        print(f"  {k}: (str, len={len(v)}) {v[:80]}")
    elif isinstance(v, list):
        print(f"  {k}: (list, len={len(v)})")
    else:
        print(f"  {k}: {type(v)}")

print()
print("=== DOM ELEMENTS ===")
dom = obs.get("dom_object") or obs.get("dom") or []
if isinstance(dom, list):
    for el in dom:
        print(f"  [{el.get('index','?')}] {el.get('tag')} role={el.get('role')} name=\"{el.get('name')}\" text=\"{el.get('text','')[:60]}\"")
elif isinstance(dom, dict):
    for node in dom.get("nodes", [])[:40]:
        role = node.get("role", {})
        name = node.get("name", {})
        role_v = role.get("value", "") if isinstance(role, dict) else str(role)
        name_v = name.get("value", "") if isinstance(name, dict) else str(name)
        bid = node.get("browsergym_id", "?")
        print(f"  bid={bid} role={role_v} name=\"{name_v}\"")

print()
print("=== GOAL / INSTRUCTION ===")
print(obs.get("goal") or obs.get("instruction") or "(not found)")

print()
print("=== AXTREE NODES (first 30) ===")
axtree = obs.get("axtree_object") or {}
if isinstance(axtree, dict):
    for node in axtree.get("nodes", [])[:30]:
        role = node.get("role", {})
        name = node.get("name", {})
        role_v = role.get("value", "") if isinstance(role, dict) else str(role)
        name_v = name.get("value", "") if isinstance(name, dict) else str(name)
        bid = node.get("browsergym_id", "?")
        print(f"  bid={bid} role={role_v} name=\"{name_v}\"")
else:
    print(str(axtree)[:2000])

env.close()
