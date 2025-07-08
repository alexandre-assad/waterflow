import subprocess
import sys

def run_command(command: str) -> bool:
    print(f"\nRunning: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Failed: {command}")
        return False
    print(f"Success: {command}")
    return True

def main():
    steps = [
        "uv run ./waterflow/experiment.py",
        "uv run pytest",
        "uv run ./waterflow/ops/set_production_models.py"
    ]

    for command in steps:
        if not run_command(command):
            print("Aborting, next steps skipped.")
            sys.exit(1)

    # All previous steps passed → launch app
    run_command("uv run ./waterflow/app.py")

if __name__ == "__main__":
    main()