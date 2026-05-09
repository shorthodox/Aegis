#!/usr/bin/env python3
"""Debug .env file loading"""
import os
from pathlib import Path

# Read the .env file directly
env_path = Path(".env")
if env_path.exists():
    with open(env_path, "r") as f:
        content = f.read()
    print("=== RAW .env FILE CONTENT (relevant lines) ===")
    for i, line in enumerate(content.split("\n"), 1):
        if "JWT" in line or "CASHFREE" in line or "GOOGLE_CLIENT" in line:
            print(f"Line {i}: {repr(line)}")

# Now try to load it
from dotenv import load_dotenv
load_dotenv(override=True)

print("\n=== LOADED VARIABLES ===")
print(f"JWT_SECRET_KEY: {repr(os.getenv('JWT_SECRET_KEY'))}")
print(f"CASHFREE_APP_ID: {repr(os.getenv('CASHFREE_APP_ID'))}")
print(f"CASHFREE_SECRET_KEY: {repr(os.getenv('CASHFREE_SECRET_KEY'))}")
print(f"GOOGLE_CLIENT_ID: {repr(os.getenv('GOOGLE_CLIENT_ID'))}")
print(f"FIREBASE_PROJECT_ID: {repr(os.getenv('FIREBASE_PROJECT_ID'))}")
