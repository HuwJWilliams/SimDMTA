# %%
import os
import json

def replace_pearson_key_substring(root_dir: str):
    print(f"🔍 Starting scan in: {root_dir}")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"📁 In directory: {dirpath}")
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                print(f"🔧 Found JSON: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    updated = replace_key_substring(data, "Pearson", "pearson")

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(updated, f, indent=4)

                    print(f"✅ Updated: {file_path}")
                except Exception as e:
                    print(f"⚠️  Skipped (error): {file_path}\n    ↳ {e}")

def replace_key_substring(obj, old_substring, new_substring):
    if isinstance(obj, dict):
        return {
            k.replace(old_substring, new_substring): replace_key_substring(v, old_substring, new_substring)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [replace_key_substring(item, old_substring, new_substring) for item in obj]
    else:
        return obj
    
# %%

replace_pearson_key_substring(root_dir="/users/yhb18174/Recreating_DMTA/results/rdkit_desc/complete_archive/")
# %%
