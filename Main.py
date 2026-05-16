from Parser import load_model
from SolutionConstructor import SolutionConstructor, save_solution
from SolutionValidator import validate_solution, parse_solution_file

SEED = 15

def main(instance_file, solution_file, enforce_mandatory=True):
    print(f"Loading Instance: '{instance_file}'...")
    model = load_model(instance_file)

    print("\n--- Model Summary ---")
    print(f"Number of nodes: {model.num_nodes} (including depot)")
    print(f"Vehicle capacity (Q): {model.capacity}")
    print(f"Number of vehicles (K): {model.vehicles}")
    print(f"Time limit per route (T_max): {model.t_max}")
    print(f"Enforce mandatory: {enforce_mandatory}")

    if solution_file:
        print(f"\n--- Constructing greedy solution (seed={SEED}) ---")
        constructor = SolutionConstructor(model, seed=SEED)
        constructed_routes = constructor.construct(enforce_mandatory=enforce_mandatory)
        save_solution(constructed_routes, solution_file)
        print(f"Solution written to '{solution_file}'.")

        print(f"\n--- Validating solution from '{solution_file}' ---")
        routes = parse_solution_file(solution_file)

        print("\nRoutes Proposed:")
        for i, route in enumerate(routes):
            print(f"Route {i}: {route}")

        valid, report = validate_solution(model, routes, enforce_mandatory=enforce_mandatory)

        if valid:
            print("\n✅ Solution is VALID.")
            print(f"🏆 Total Profit Collected: {report['total_profit']}")
            print("\nRoute Details:")
            for i, (load, cost, profit) in enumerate(zip(report['route_loads'], report['route_costs'], report['route_profits'])):
                print(f"Route {i} -> Load = {load}/{model.capacity}, Time = {cost:.2f}/{model.t_max}, Profit = {profit}")
        else:
            print("\n❌ Solution is INVALID.")
            print("Errors Found:")
            for error in report['errors']:
                print(f" - {error}")

        from SolutionPlotter import plot_ctop_solution
        plot_ctop_solution(model, routes)


if __name__ == "__main__":
    main("ctop_main_instance.txt", "solution_mandatory.txt", True)
    main("ctop_main_instance.txt", "solution_no_mandatory.txt", False)
