from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    id: int
    profit: int
    demand: int
    costs: List[float]
    isDepot: bool = False
    isMandatory: bool = False


@dataclass
class Model:
    num_nodes: int = 0
    vehicles: int = 0
    capacity: int = 0
    t_max: float = 0.0
    nodes: List[Node] = None
    cost_matrix: List[List[float]] = None


def load_model(file_name):
    """
    Format:
    1st line: |N| K Q Tmax (num_nodes, vehicles, capacity, max_time)
    2nd line: List of profits (p0, p1, ..., pn) -> p0 is always 0
    3rd line: List of demands (d0, d1, ..., dn) -> d0 is always 0
    4th line: List of mandatory flags (m0, m1, ..., mn) -> 1 = mandatory, 0 = optional
    5th line until end: Cost/Time matrix (cij)
    """
    parsed_model = Model()
    with open(file_name, "r") as f:
        all_lines = f.readlines()

    # 1st line (Parameters)
    params = all_lines[0].split()
    parsed_model.num_nodes = int(params[0])
    parsed_model.vehicles = int(params[1])
    parsed_model.capacity = int(params[2])
    parsed_model.t_max = float(params[3])

    # 2nd line: Profits
    profits = list(map(int, all_lines[1].split()))

    # 3rd line: Demands
    demands = list(map(int, all_lines[2].split()))

    # 4th line: Mandatory
    mandatories = list(map(int, all_lines[3].split()))

    # Cost Matrix
    cost_matrix = []
    for i in range(4, 4 + parsed_model.num_nodes):
        row = list(map(float, all_lines[i].split()))
        cost_matrix.append(row)

    parsed_model.cost_matrix = cost_matrix
    parsed_model.nodes = []

    for i in range(parsed_model.num_nodes):
        node = Node(
            id=i,
            profit=profits[i],
            demand=demands[i],
            costs=cost_matrix[i],
            isDepot=(i == 0),
            isMandatory=bool(mandatories[i])
        )
        parsed_model.nodes.append(node)

    return parsed_model