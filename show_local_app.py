import pathlib

path = pathlib.Path("src/social_rlvr_web/local_app.py")
if path.exists():
    print(path.read_text(encoding="utf-8"))
else:
    # search for it
    for p in pathlib.Path("src").rglob("*.py"):
        print(p)
