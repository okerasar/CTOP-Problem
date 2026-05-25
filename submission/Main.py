"""
Main.py
───────
Validates τα solution_*.txt αρχεία και τα οπτικοποιεί.
Τυπώνει αναλυτικά: profit, valid, fleet usage, και ανά route
(αριθμός κόμβων, load, time, profit).
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from Parser import load_model
from SolutionValidator import validate_solution, parse_solution_file
from SolutionPlotter import plot_ctop_solution
import matplotlib.pyplot as plt


def print_solution(label, filename, routes, model, enforce_mandatory):
    valid, rep = validate_solution(model, routes, enforce_mandatory=enforce_mandatory)
    valid_str = "✓ ΕΓΚΥΡΗ" if valid else "✗ ΜΗ ΕΓΚΥΡΗ"
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print('─' * 64)
    print(f"  Αρχείο : {filename}")
    print(f"  Profit : {rep['total_profit']}")
    print(f"  Valid  : {valid_str}")
    print(f"  Στόλος : {len(routes)}/{model.vehicles} οχήματα σε χρήση")
    for i, (load, cost, profit) in enumerate(zip(
            rep['route_loads'], rep['route_costs'], rep['route_profits'])):
        nodes_count = len(routes[i]) - 2
        print(f"    Route {i}: {nodes_count:3d} κόμβοι | "
              f"load={load:3d}/{model.capacity} | "
              f"time={cost:7.1f}/{model.t_max} | "
              f"profit={profit:4d}")
    if not valid:
        for e in rep['errors']:
            print(f"    ERROR: {e}")


def check_and_plot(label, filename, model, enforce_mandatory):
    path = os.path.join(_DIR, filename)
    if not os.path.exists(path):
        print(f"\n  ⚠  Δεν βρέθηκε το αρχείο: {filename}")
        return None
    routes = parse_solution_file(path)
    print_solution(label, filename, routes, model, enforce_mandatory)
    return routes


if __name__ == "__main__":
    instance = os.path.join(_DIR, "ctop_main_instance.txt")
    model = load_model(instance)

    routes_p1 = check_and_plot("Π1 (no mandatory)",
                               "solution_no_mandatory.txt",
                               model, enforce_mandatory=False)

    routes_p2 = check_and_plot("Π2 (mandatory)",
                               "solution_mandatory.txt",
                               model, enforce_mandatory=True)

    # Plots
    if routes_p1 is not None:
        plot_ctop_solution(model, routes_p1)
    if routes_p2 is not None:
        plot_ctop_solution(model, routes_p2)
