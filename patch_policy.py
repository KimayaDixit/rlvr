import re

path = "src/social_rlvr_web/model_policy.py"

with open(path, encoding="utf-8") as f:
    text = f.read()

# Fix 1: increase num_predict
text = re.sub(r"num_predict: int = \d+", "num_predict: int = 512", text)

# Fix 2: remove format:json constraint (causes premature JSON closing)
text = text.replace('"format": "json",', "")
text = text.replace("'format': 'json',", "")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

# Verify
m = re.search(r"num_predict.*", text)
print("num_predict line:", m.group() if m else "NOT FOUND")
print("format:json present:", "format" in text and "json" in text)
print("Patch applied successfully.")
