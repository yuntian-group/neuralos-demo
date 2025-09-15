import os

base_dir = "model_generated/train"

for folder in os.listdir(base_dir):
    full_path = os.path.join(base_dir, folder)

    # Only process directories
    if not os.path.isdir(full_path):
        continue

    new_name = folder + "_bak"
    new_path = os.path.join(base_dir, new_name)
    # Check if "ddim" is in the name, but it doesn't end with "_bak"
    if "ddim" in folder and not folder.endswith("_bak") and not os.path.exists(new_path):

        # Rename folder
        print(f"Renaming: {folder} -> {new_name}")
        os.rename(full_path, new_path)

