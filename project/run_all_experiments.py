"""Усі обов'язкові експерименти ЛР1–ЛР5 однією командою."""

from pathlib import Path

import lab1_experiments
import lab2_experiments
import lab3_experiments
import lab4_experiments
import lab5_experiments
from src.preparation import prepare_project


def run_lab3_experiments(project):
    module = lab3_experiments
    print("\nLAB 3 - REQUIRED EXPERIMENTS")
    point = module.initial_point()
    print(f"Initial point = ({point[0]:.6f}; {point[1]:.6f})")
    nm, pw = module.run_base_methods(project.clean, project.noisy)
    best = nm if nm["f"] <= pw["f"] else pw
    module.experiment_simplex_size(project.clean, project.noisy)
    module.experiment_powell_cycle(project.clean, project.noisy)
    module.experiment_powell_tolerance(project.clean, project.noisy, pw)
    module.experiment_one_parameter(project.clean, project.noisy, best)
    print("LAB 3 EXPERIMENTS COMPLETE")


def main():
    Path("outputs").mkdir(exist_ok=True)
    project = prepare_project()
    lab1_experiments.run_experiments(project)
    lab2_experiments.run_experiments(project)
    run_lab3_experiments(project)
    lab4_experiments.run_experiments(project)
    lab5_experiments.run_experiments(project)
    print("\nALL REQUIRED EXPERIMENTS COMPLETE")
    print("Tables and plots: outputs/")


if __name__ == "__main__":
    main()
