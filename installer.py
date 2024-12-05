import os
import subprocess
import sys


def run_command(command):
    """Helper function to execute system commands."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    print("Starting installation...")

    if not os.path.exists("sf3d_env"):
        print("Creating virtual environment...")
        run_command("python -m venv sf3d_env")
    else:
        print("Virtual environment already exists.")

    activate_command = (
        ".\\sf3d_env\\Scripts\\activate" if os.name == "nt" else "source sf3d_env/bin/activate"
    )
    print("Activating virtual environment...")
    run_command(activate_command)

    print("Installing dependencies...")
    run_command("pip install -r requirements.txt")

    print("Installing local modules...")
    run_command("pip install ./uv_unwrapper/")
    run_command("pip install ./texture_baker/")

    print("Verifying installation...")
    run_command("python run.py demo_files/examples/chair1.png --output-dir output/ --device cpu")

    print("Installation complete!")

if __name__ == "__main__":
    main()
