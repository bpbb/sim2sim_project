from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="List environments.")
parser.add_argument("--headless", action="store_true", help="Run headless.")
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

# Import the local lab package to trigger registration
import lab.flamingo.tasks

def main():
    table = PrettyTable(["S. No.", "Task Name"])
    table.title = "Available Environments"
    table.align["Task Name"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "Flamingo" in task_spec.id or "Isaac-Velocity" in task_spec.id:
            table.add_row([index + 1, task_spec.id])
            index += 1

    print(table)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
