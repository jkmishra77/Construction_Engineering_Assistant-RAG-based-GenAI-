import os

# Define folder and file structure
structure = {
    "backend": [
        "llm.py",
        "embedding.py",
        "query.py",
        "graph.py",
        "main.py"
    ],
    ".": [
        "frontend.py",
        ".env",
        "requirements.txt"
    ]
}

# Create folders and empty files
for folder, files in structure.items():
    if folder != ".":
        os.makedirs(folder, exist_ok=True)
    for file in files:
        path = os.path.join(folder, file) if folder != "." else file
        with open(path, "w") as f:
            pass  # Create empty file

print(" Folder structure and empty files created.")
