
import sys
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError

def get_cmd_version(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return output.decode().strip().split('\n')[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Not found"

def print_pkg_version(package_name):
    try:
        v = version(package_name)
        print(f"{package_name:<15} {v}")
    except PackageNotFoundError:
        print(f"{package_name:<15} Not installed")

def main():
    print("="*40)
    print("Environment Report")
    print("="*40)
    print(f"OS:        {platform.system()} {platform.release()}")
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python:    {sys.version.split()[0]}")
    
    print("\n" + "="*40)
    print("External Tools")
    print("="*40)
    print(f"Rust/Cargo:     {get_cmd_version('cargo --version')}")
    print(f"R:              {get_cmd_version('R --version')}")
    
    print("\n" + "="*40)
    print("Python Packages")
    print("="*40)
    
    packages = [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", 
        "polars", "maturin", "scanpy", "anndata"
    ]
    
    for pkg in packages:
        print_pkg_version(pkg)
        
if __name__ == "__main__":
    main()
