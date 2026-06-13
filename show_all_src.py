import pathlib

files = [
    "src/social_rlvr_web/local_app.py",
    "src/social_rlvr_web/browsergym_tasks.py",
    "src/social_rlvr_web/rlvr_policy.py",
    "src/social_rlvr_web/observation.py",
]

for f in files:
    p = pathlib.Path(f)
    if p.exists():
        print(f"\n{'='*60}")
        print(f"FILE: {f}")
        print('='*60)
        print(p.read_text(encoding="utf-8"))
    else:
        print(f"\n[NOT FOUND]: {f}")

# also list all files in src
print("\n\nALL FILES IN SRC:")
for p in pathlib.Path("src").rglob("*.py"):
    print(f"  {p}")
