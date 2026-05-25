def parse_solution_file(file_name):
    routes = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                route = list(map(int, line.split()))
                routes.append(route)
    return routes


def validate_solution(model, routes, enforce_mandatory=True):
    validation_report = {
        "valid": True,
        "total_profit": 0,
        "total_cost": 0,
        "errors": [],
        "route_loads": [],
        "route_costs": [],
        "route_profits": []
    }

    if len(routes) > model.vehicles:
        validation_report["valid"] = False
        validation_report["errors"].append(f"Too many vehicles used: {len(routes)} > {model.vehicles}")

    visited_nodes = set()
    mandatory_nodes = {node.id for node in model.nodes if node.isMandatory}

    for route_idx, route in enumerate(routes):
        route_load = 0
        route_cost = 0
        route_profit = 0

        # Check if route starts and ends at Depot (0)
        if route[0] != 0 or route[-1] != 0:
            validation_report["valid"] = False
            validation_report["errors"].append(f"Route {route_idx} does not start/end at depot (0).")

        for i in range(len(route) - 1):
            curr_node = route[i]
            next_node = route[i + 1]

            if curr_node != 0:
                # Check for multiple visits
                if curr_node in visited_nodes:
                    validation_report["valid"] = False
                    validation_report["errors"].append(f"Node {curr_node} visited multiple times.")
                visited_nodes.add(curr_node)

                route_load += model.nodes[curr_node].demand
                route_profit += model.nodes[curr_node].profit

            route_cost += model.cost_matrix[curr_node][next_node]

        # Capacity Constraint
        if route_load > model.capacity:
            validation_report["valid"] = False
            validation_report["errors"].append(f"Route {route_idx} exceeds capacity: {route_load} > {model.capacity}")

        # Time Limit Constraint (T_max)
        if route_cost > model.t_max:
            validation_report["valid"] = False
            validation_report["errors"].append(
                f"Route {route_idx} exceeds time limit: {route_cost:.2f} > {model.t_max}")

        validation_report["route_loads"].append(route_load)
        validation_report["route_costs"].append(route_cost)
        validation_report["route_profits"].append(route_profit)

        validation_report["total_profit"] += route_profit
        validation_report["total_cost"] += route_cost

    # Check if all mandatory nodes are visited
    if enforce_mandatory:
        missing_mandatory = mandatory_nodes - visited_nodes
        if missing_mandatory:
            validation_report["valid"] = False
            validation_report["errors"].append(f"Missing mandatory nodes: {missing_mandatory}")

    return validation_report["valid"], validation_report