import random
from typing import List, Optional, Tuple

from Parser import Model


# (route_index, insertion_position, insertion_cost_delta, score)
Insertion = Tuple[int, int, float, float]


class SolutionConstructor:
    """
    GRASP-style constructive heuristic for the Capacitated Team Orienteering Problem.

    Strategy:
      1. Start with K empty routes of the form [0, 0].
      2. If `enforce_mandatory` is True, insert every mandatory customer first
         using cheapest insertion (they must be served regardless of profit).
      3. Iteratively build a Restricted Candidate List (RCL) of unrouted
         customers, ranked by their best profit-to-insertion-cost ratio across
         all feasible (route, position) pairs that respect capacity (Q) and
         time (T_max). One element of the RCL is sampled uniformly at random
         using the seeded RNG, so different seeds yield different initial
         solutions. Stop when no feasible insertion remains.

    The `alpha` parameter controls greediness vs. diversification:
      * alpha = 0.0 -> pure greedy (argmax), deterministic regardless of seed
      * alpha = 1.0 -> uniform random over all feasible insertions
      * 0.1..0.3   -> typical GRASP values, recommended for multi-start

    The output is a `List[List[int]]` (one route per inner list, each starting
    and ending at the depot) so it can be consumed directly by
    `SolutionValidator.validate_solution` and `SolutionPlotter.plot_ctop_solution`.
    """

    EPS = 1e-6

    def __init__(self, model: Model, seed: int = 42, alpha: float = 0.3):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.model = model
        self.alpha = alpha
        self._rng = random.Random(seed)

    def construct(self, enforce_mandatory: bool = False) -> List[List[int]]:
        K = self.model.vehicles
        routes: List[List[int]] = [[0, 0] for _ in range(K)]
        loads: List[int] = [0] * K
        times: List[float] = [0.0] * K
        visited = {0}

        if enforce_mandatory:
            self._insert_mandatory(routes, loads, times, visited)

        self._insert_by_profit(routes, loads, times, visited)
        return routes

    def _insert_mandatory(
        self,
        routes: List[List[int]],
        loads: List[int],
        times: List[float],
        visited: set,
    ) -> None:
        mandatory = [
            n for n in self.model.nodes if n.isMandatory and not n.isDepot
        ]
        mandatory.sort(key=lambda n: (-n.demand, -n.profit))

        for node in mandatory:
            best = self._find_best_insertion(
                node.id, routes, loads, times, ratio=False
            )
            if best is None:
                raise RuntimeError(
                    f"Mandatory node {node.id} cannot be inserted into any "
                    f"feasible route (capacity or time limit exhausted)."
                )
            self._apply(node.id, best, routes, loads, times, visited)

    def _insert_by_profit(
        self,
        routes: List[List[int]],
        loads: List[int],
        times: List[float],
        visited: set,
    ) -> None:
        while True:
            feasible: List[Tuple[int, Insertion]] = []
            for node in self.model.nodes:
                if node.isDepot or node.id in visited:
                    continue
                pick = self._find_best_insertion(
                    node.id, routes, loads, times, ratio=True
                )
                if pick is not None:
                    feasible.append((node.id, pick))

            if not feasible:
                break

            scores = [ins[3] for _, ins in feasible]
            best_score = max(scores)
            worst_score = min(scores)
            threshold = best_score - self.alpha * (best_score - worst_score)
            rcl = [c for c in feasible if c[1][3] >= threshold]

            chosen_id, chosen_pick = self._rng.choice(rcl)
            self._apply(chosen_id, chosen_pick, routes, loads, times, visited)

    def _find_best_insertion(
        self,
        node_id: int,
        routes: List[List[int]],
        loads: List[int],
        times: List[float],
        ratio: bool,
    ) -> Optional[Insertion]:
        """
        Best feasible insertion of `node_id` across all routes/positions.

        `ratio=True`  -> score = profit / (insertion_cost + EPS)   (profit pass)
        `ratio=False` -> score = -insertion_cost                   (mandatory pass)
        """
        node = self.model.nodes[node_id]
        cost = self.model.cost_matrix
        capacity = self.model.capacity
        t_max = self.model.t_max
        best: Optional[Insertion] = None

        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node.demand > capacity:
                continue

            for pos in range(1, len(route)):
                a, b = route[pos - 1], route[pos]
                delta = cost[a][node_id] + cost[node_id][b] - cost[a][b]
                if times[r_idx] + delta > t_max:
                    continue

                score = (node.profit / (delta + self.EPS)) if ratio else -delta
                if best is None or score > best[3]:
                    best = (r_idx, pos, delta, score)

        return best

    def _apply(
        self,
        node_id: int,
        pick: Insertion,
        routes: List[List[int]],
        loads: List[int],
        times: List[float],
        visited: set,
    ) -> None:
        r_idx, pos, delta, _ = pick
        routes[r_idx].insert(pos, node_id)
        loads[r_idx] += self.model.nodes[node_id].demand
        times[r_idx] += delta
        visited.add(node_id)


def save_solution(routes: List[List[int]], file_path: str) -> None:
    """
    Persist routes in the format expected by `SolutionValidator.parse_solution_file`:
    one route per line, node ids space-separated, each route depot-anchored (0 ... 0).
    Empty routes (i.e. [0, 0]) are omitted from the output.
    """
    with open(file_path, "w") as f:
        for route in routes:
            if len(route) <= 2:
                continue
            f.write(" ".join(str(n) for n in route) + "\n")
