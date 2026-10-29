from pathlib import Path

# current directory
root = Path(".")

#Folders to create

folders = [
    "app/api",
    "app/core",
    "app/services",
    "app/rag",
    "data",
    "frontend",
    "static",
    "uploads",
    "tests"
]

# files to create
files = [
    "app/main.py",
    "app/api/__init__.py",
    "app/core/__init__.py",
    "app/services/__init__.py",
    "app/rag/__init__.py",
    "requirements.txt",
    "run.py",
    ".env"
]

# create folders

for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)
    
# create files
for file in files:
    (root / file).touch(exist_ok=True)
    
print("Project structure created successfully.")