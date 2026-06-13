import re

path = "src/social_rlvr_web/model_policy.py"

with open(path, encoding="utf-8") as f:
    text = f.read()

old = '''    @staticmethod
    def _valid_bid_lines(axtree: Any) -> str:
        if not isinstance(axtree, dict):
            return ""
        lines = []
        for node in axtree.get("nodes", []):
            if not isinstance(node, dict) or "browsergym_id" not in node:
                continue
            role = node.get("role", {})
            name = node.get("name", {})
            role_value = role.get("value", "") if isinstance(role, dict) else str(role)
            name_value = name.get("value", "") if isinstance(name, dict) else str(name)
            bid = node["browsergym_id"]
            if role_value in {"textbox", "button", "combobox", "menuitem", "link", "checkbox"} or name_value:
                lines.append(f\'- bid="{bid}" role={role_value} name="{name_value}"\')
        return "\\n".join(lines[:80])'''

new = '''    @staticmethod
    def _valid_bid_lines(axtree: Any) -> str:
        if not isinstance(axtree, dict):
            return ""
        INTERACTIVE = {
            "textbox", "button", "combobox", "menuitem", "link",
            "checkbox", "radio", "option", "spinbutton", "switch",
        }
        seen_bids = set()
        lines = []
        for node in axtree.get("nodes", []):
            if not isinstance(node, dict) or "browsergym_id" not in node:
                continue
            role = node.get("role", {})
            name = node.get("name", {})
            role_value = role.get("value", "") if isinstance(role, dict) else str(role)
            name_value = (name.get("value", "") if isinstance(name, dict) else str(name)).strip()
            bid = node["browsergym_id"]
            if bid in seen_bids:
                continue
            # include checked state for checkboxes
            props = node.get("properties", [])
            checked = next(
                (p.get("value", {}).get("value") for p in props if p.get("name") == "checked"),
                None,
            )
            val = node.get("value", {})
            val_v = (val.get("value", "") if isinstance(val, dict) else str(val) if val else "").strip()
            if role_value in INTERACTIVE:
                extra = ""
                if checked is not None:
                    extra += f" checked={checked}"
                if val_v:
                    extra += f" value=\\"{val_v}\\""
                lines.append(f\'- bid="{bid}" role={role_value} name="{name_value}"{extra}\')
                seen_bids.add(bid)
        return "\\n".join(lines[:80])'''

if old in text:
    text = text.replace(old, new)
    print("Replaced _valid_bid_lines via exact match.")
else:
    print("ERROR: exact match not found. Searching for method signature...")
    if "_valid_bid_lines" in text:
        print("Method exists but text didn't match exactly.")
        print("Please manually replace _valid_bid_lines in model_policy.py")
    else:
        print("Method not found at all!")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done.")
