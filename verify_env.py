#!/usr/bin/env python3
"""Quick verification that environment variables load correctly"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("\n" + "="*70)
print("AEGIS-1 ENVIRONMENT CONFIGURATION VERIFICATION")
print("="*70 + "\n")

# Check critical variables
checks = {
    "FIREBASE_PROJECT_ID": ("aegis-d78e1", True),
    "FIREBASE_CREDENTIALS": ("config/serviceAccountKey.json", True),
    "JWT_SECRET_KEY": (None, True),
    "GOOGLE_CLIENT_ID": (None, False),
    "CASHFREE_APP_ID": (None, False),
    "CASHFREE_SECRET_KEY": (None, False),
}

critical_count = 0
optional_count = 0

for var_name, (expected, is_critical) in checks.items():
    value = os.getenv(var_name)
    is_set = bool(value)
    
    if is_critical:
        critical_count += 1
        status = "✅ CRITICAL" if is_set else "❌ CRITICAL"
    else:
        optional_count += 1
        status = "✅ OPTIONAL" if is_set else "⚠️  OPTIONAL"
    
    display_value = value[:20] + "..." if value and len(value) > 20 else value
    print(f"{status:15} | {var_name:25} | {display_value or '(not set)'}")

print("\n" + "-"*70)
print(f"Critical Variables: {'✅ ALL SET' if all(os.getenv(var) for var, (_, critical) in checks.items() if critical) else '❌ MISSING SOME'}")
print("-"*70 + "\n")

# Try to import Firebase Admin SDK
try:
    import firebase_admin
    print("✅ Firebase Admin SDK is installed")
except ImportError:
    print("❌ Firebase Admin SDK is NOT installed - run: pip install firebase-admin")

# Check if serviceAccountKey.json exists
import pathlib
sa_path = pathlib.Path(os.getenv("FIREBASE_CREDENTIALS", "config/serviceAccountKey.json"))
if sa_path.exists():
    print(f"✅ Service Account Key found at: {sa_path}")
else:
    print(f"❌ Service Account Key NOT found at: {sa_path}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("""
1. Copy .env.template to .env (if not already done):
   cp .env.template .env

2. Edit .env and fill in your actual values for:
   - GOOGLE_CLIENT_ID
   - GOOGLE_CLIENT_SECRET  
   - CASHFREE_APP_ID
   - CASHFREE_SECRET_KEY
   - JWT_SECRET_KEY (generate a secure random string)

3. Verify serviceAccountKey.json exists in config/ folder

4. Run the FastAPI server:
   python main.py

5. Test Google Auth at: http://localhost:8000
""")
print("="*70 + "\n")
