#!/usr/bin/env python3
import os
import re
import shutil

ROOT = "/home/kamau/bsopt"
MAPPINGS = {
    "src.shared/security": "src/auth",
    "src/auth": "src/auth",
    "src/pricing": "src/math_kernel",
    "src.shared/trading": "src/math_kernel",
    "src.shared/wasm": "src/math_kernel",
    "src/ml": "src/ml",
    "src/scrapers": "src/ingestion",
    "src/data": "src/ingestion",
    "src.shared/data": "src/ingestion",
    "src.shared/database": "src/database",
    "src.frontend": "src.frontend",  # wait, we map paths first
    "src/frontend": "src/frontend",
    "src/api": "src/api",
    "src/gateway": "src/api",
    "src/portfolio": "src/portfolio",
    "src.shared/shared": "src/shared",
    "src/shared": "src/shared",
}

IMPORT_MAPPINGS = {
    r"\bcore\.security\b": "src.auth",
    r"\bservices\.auth\b": "src.auth",
    r"\bservices\.pricing\b": "src.math_kernel",
    r"\bcore\.trading\b": "src.math_kernel",
    r"\bcore\.wasm\b": "src.math_kernel",
    r"\bservices\.ml\b": "src.ml",
    r"\bservices\.scrapers\b": "src.ingestion",
    r"\bservices\.data\b": "src.ingestion",
    r"\bcore\.data\b": "src.ingestion",
    r"\bcore\.database\b": "src.database",
    r"\bservices\.frontend\b": "src.frontend",
    r"\bservices\.api\b": "src.api",
    r"\bservices\.gateway\b": "src.api",
    r"\bservices\.portfolio\b": "src.portfolio",
    r"\bcore\.shared\b": "src.shared",
    r"\bservices\.shared\b": "src.shared",
    r"\bcore\b": "src.shared",  # Fallbacks for root level files
    r"\bservices\b": "src",  # Fallbacks
}

def move_files():
    for old_rel, new_rel in MAPPINGS.items():
        if "." in old_rel:
            continue  # skip pure import mappings

        old_path = os.path.join(ROOT, old_rel)
        new_path = os.path.join(ROOT, new_rel)

        if not os.path.exists(old_path):
            continue

        os.makedirs(new_path, exist_ok=True)

        for item in os.listdir(old_path):
            s = os.path.join(old_path, item)
            d = os.path.join(new_path, item)
            if os.path.exists(d):
                if os.path.isdir(s):
                    # merge dir recursively
                    os.system(f"rsync -a {s}/ {d}/")
                    shutil.rmtree(s)
                else:
                    # just overwrite if it's __init__.py or something or keep newer
                    if item == "__init__.py":
                        pass  # keep existing
                    else:
                        shutil.move(s, d)
            else:
                shutil.move(s, d)

        try:
            os.rmdir(old_path)
        except OSError:
            pass

def rewrite_imports():
    patterns = [(re.compile(k), v) for k, v in IMPORT_MAPPINGS.items()]

    # directories to check
    for root, dirs, files in os.walk(ROOT):
        if ".git" in root or ".venv" in root or "node_modules" in root:
            continue

        for file in files:
            if not file.endswith(
                (".py", ".ts", ".tsx", ".js", ".md", ".sh", ".yaml", ".yml", ".rs")
            ):
                continue

            filepath = os.path.join(root, file)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            new_content = content
            for p, v in patterns:
                new_content = p.sub(v, new_content)

            if content != new_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)

if __name__ == "__main__":
    print("Moving files...")
    move_files()
    print("Rewriting imports...")
    rewrite_imports()

    # Final cleanup of empty src.shared and src dirs
    for d in ["src.shared", "src"]:
        p = os.path.join(ROOT, d)
        if os.path.exists(p):
            os.system(f"rm -rf {p}")

    print("Done restructuring and import mapping.")
