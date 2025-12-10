"""
Environment Validation Script
Verifies that all dependencies are correctly installed
"""
import sys
import importlib
from typing import List, Tuple

REQUIRED_PACKAGES = [
    ('fastapi', 'FastAPI'),
    ('pydantic', 'Pydantic'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('google.generativeai', 'Gemini AI'),
    ('uvicorn', 'Uvicorn'),
    ('httpx', 'HTTPX'),
    ('dotenv', 'Python-dotenv'),
]

def validate_python_version() -> bool:
    """Check Python version is 3.12+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (requires 3.12+)")
        return False

def validate_packages() -> Tuple[List[str], List[str]]:
    """Validate all required packages can be imported"""
    success = []
    failed = []
    
    for package, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            success.append(display_name)
        except ImportError:
            print(f"❌ {display_name}: Not installed")
            failed.append(display_name)
    
    return success, failed

def main():
    print("=" * 50)
    print("  VAIA Environment Validation")
    print("=" * 50)
    print()
    
    # Check Python version
    python_ok = validate_python_version()
    print()
    
    # Check packages
    print("Checking required packages...")
    success, failed = validate_packages()
    print()
    
    # Summary
    if python_ok and not failed:
        print("🎉 Environment validation successful!")
        print(f"   {len(success)} packages verified")
        return 0
    else:
        print("⚠️  Environment validation failed!")
        if failed:
            print(f"   Missing packages: {', '.join(failed)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
