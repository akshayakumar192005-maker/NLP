import subprocess
import sys

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# List of required packages
required_packages = ["joblib", "scikit-learn"]

for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"{package} not found. Installing silently...")
        install(package)
        print(f"{package} installed successfully")

print("All required packages are ready to use!")