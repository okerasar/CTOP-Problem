import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS


def plot_ctop_solution(model, routes):
    """
    Plots the CTOP solution. Uses MDS to infer 2D coordinates since
    the instance file only provides a precomputed distance matrix.
    """
    print("Inferring 2D coordinates from the distance matrix via MDS...")
    matrix = np.array(model.cost_matrix)

    # Ensure matrix is perfectly symmetric for MDS
    matrix = (matrix + matrix.T) / 2

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(matrix)

    plt.figure(figsize=(12, 10))

    # 1. Gather Assigned vs Unassigned nodes
    assigned_nodes = set()
    for route in routes:
        assigned_nodes.update(route)

    unassigned = [i for i in range(1, model.num_nodes) if i not in assigned_nodes]

    # 1.5 Find mandatory vs optional unassigned
    unassigned_mandatory = [i for i in unassigned if model.nodes[i].isMandatory]
    unassigned_optional = [i for i in unassigned if not model.nodes[i].isMandatory]

    # 2. Plot Unassigned Nodes (Light Gray)
    if unassigned_optional:
        plt.scatter(coords[unassigned_optional, 0], coords[unassigned_optional, 1],
                    c='lightgray', label='Unassigned Optional', s=30, zorder=2)
    if unassigned_mandatory:
        plt.scatter(coords[unassigned_mandatory, 0], coords[unassigned_mandatory, 1],
                    c='red', marker='x', label='Unassigned Mandatory', s=50, zorder=4)

    # 3. Plot Routes
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
    total_profit = 0

    for idx, route in enumerate(routes):
        route_coords = coords[route]
        plt.plot(route_coords[:, 0], route_coords[:, 1],
                 marker='o', color=colors[idx], linewidth=2,
                 label=f'Route {idx}', zorder=3)

        # Highlight mandatory nodes on the route
        route_mandatory_nodes = [n for n in route if model.nodes[n].isMandatory]
        if route_mandatory_nodes:
            plt.scatter(coords[route_mandatory_nodes, 0], coords[route_mandatory_nodes, 1],
                        edgecolors='black', facecolors='none', linewidths=2, s=80, 
                        label='Mandatory Node' if idx == 0 else "", zorder=4)

        # Calculate profit for the title
        total_profit += sum(model.nodes[n].profit for n in route)

    # 4. Plot Depot (Red Square)
    plt.scatter(coords[0, 0], coords[0, 1],
                c='red', marker='s', s=150, label='Depot', edgecolors='black', zorder=5)

    plt.title(f"CTOP Solution Map | Total Profit: {total_profit} | Fleet: {len(routes)}/{model.vehicles}")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
