"""
CTOP Solver -- Adaptive Large Neighbourhood Search (ALNS)
Handles both Problem 1 (no mandatory) and Problem 2 (mandatory nodes).
Seeds allowed: 4, 8, 15, 16, 23, 42
"""

import random
import math
import time
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from Parser import load_model
from SolutionValidator import validate_solution

# ── Tuning ────────────────────────────────────────────────────────────────────
TIME_LIMIT = 300   # seconds per problem — do not change

# After running tune_alns_separate.py, paste the values from
# alns_best_p1.json here and alns_best_p2.json in P2_PARAMS below.
P1_PARAMS = dict(
    SEGMENT_SIZE = 191,
    REACT        = 0.0798,
    MIN_W        = 0.1401,
    SCORE_BEST   = 11,
    SCORE_BETTER = 10,
    SCORE_ACCEPT = 5,
    T0_FRAC      = 0.0719,
    COOLING      = 0.999379,   # fallback only — actual cooling computed dynamically in alns()
    Q_MIN = 11,
    Q_MAX = 16,
    seed = 23
)

P2_PARAMS = dict(
    SEGMENT_SIZE = 137,
    REACT        = 0.1276,
    MIN_W        = 0.1269,
    SCORE_BEST   = 7,
    SCORE_BETTER = 12,
    SCORE_ACCEPT = 4,
    T0_FRAC      = 0.0559,
    COOLING      = 0.999763,   # fallback only — actual cooling computed dynamically in alns()
    Q_MIN = 14,
    Q_MAX = 22,
    seed = 16
)

# Population seeds — used by solve() to run multiple short restarts and keep the best.
# Each seed gets time_limit // len(seeds) seconds; the winner is polished once.
# Choose from {4, 8, 15, 16, 23, 42}.
P1_SEEDS = [23, 4, 42]   # 3 × 100s = 300s total
P2_SEEDS = [16, 8, 15]   # 3 × 100s = 300s total


# Module-level names used internally — set by _apply_params() before each solve
SEGMENT_SIZE = P1_PARAMS["SEGMENT_SIZE"]
REACT        = P1_PARAMS["REACT"]
MIN_W        = P1_PARAMS["MIN_W"]
SCORE_BEST   = P1_PARAMS["SCORE_BEST"]
SCORE_BETTER = P1_PARAMS["SCORE_BETTER"]
SCORE_ACCEPT = P1_PARAMS["SCORE_ACCEPT"]
T0_FRAC      = P1_PARAMS["T0_FRAC"]
COOLING      = P1_PARAMS["COOLING"]
Q_MIN        = P1_PARAMS["Q_MIN"]
Q_MAX        = P1_PARAMS["Q_MAX"]


# ── Apply per-problem params ──────────────────────────────────────────────────

def _apply_params(p: dict) -> None:
    """Overwrite module-level ALNS constants from a params dict."""
    global SEGMENT_SIZE, REACT, MIN_W, SCORE_BEST, SCORE_BETTER, SCORE_ACCEPT
    global T0_FRAC, COOLING, Q_MIN, Q_MAX
    SEGMENT_SIZE = p["SEGMENT_SIZE"]
    REACT        = p["REACT"]
    MIN_W        = p["MIN_W"]
    SCORE_BEST   = p["SCORE_BEST"]
    SCORE_BETTER = p["SCORE_BETTER"]
    SCORE_ACCEPT = p["SCORE_ACCEPT"]
    T0_FRAC      = p["T0_FRAC"]
    COOLING      = p["COOLING"]
    Q_MIN        = p["Q_MIN"]
    Q_MAX        = p["Q_MAX"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rt(seq, C):
    """Route time for a customer sequence (depot bookends added internally)."""
    full = [0] + seq + [0]
    t = 0.0
    for k in range(len(full) - 1):
        t += C[full[k]][full[k + 1]]
    return t


class Route:
    __slots__ = ('nodes', 'load', 'time')

    def __init__(self, nodes, load, time_val):
        self.nodes = list(nodes)
        self.load  = load
        self.time  = time_val

    def copy(self):
        return Route(list(self.nodes), self.load, self.time)

    def sync(self, nd, C):
        self.load = sum(nd[n].demand for n in self.nodes)
        self.time = _rt(self.nodes, C)


class Solution:
    __slots__ = ('routes', 'model', '_profit')

    def __init__(self, routes, model):
        self.routes  = routes
        self.model   = model
        self._profit = None

    def profit(self):
        if self._profit is None:
            nd = self.model.nodes
            self._profit = sum(nd[n].profit for r in self.routes for n in r.nodes)
        return self._profit

    def invalidate(self):
        self._profit = None

    def copy(self):
        s = Solution.__new__(Solution)
        s.routes  = [r.copy() for r in self.routes]
        s.model   = self.model
        s._profit = self._profit
        return s

    def to_route_lists(self):
        return [[0] + r.nodes + [0] for r in self.routes if r.nodes]

    def visited(self):
        return {n for r in self.routes for n in r.nodes}


# ── Insertion helpers ─────────────────────────────────────────────────────────

def _best_pos(r, u, nd, C, Q, T):
    """Best (score, pos) for inserting u into r. Returns None if infeasible."""
    if r.load + nd[u].demand > Q:
        return None
    slack = T - r.time
    best_sc = -1.0; best_k = None
    seq = [0] + r.nodes + [0]
    for k in range(1, len(seq)):
        a, b  = seq[k - 1], seq[k]
        extra = C[a][u] + C[u][b] - C[a][b]
        if extra > slack + 1e-7:
            continue
        sc = nd[u].profit / max(extra, 1e-9)
        if sc > best_sc:
            best_sc = sc; best_k = k - 1
    return (best_sc, best_k) if best_k is not None else None


def _insert(r, u, pos, nd, C):
    r.nodes.insert(pos, u)
    r.load += nd[u].demand
    r.time  = _rt(r.nodes, C)


def _try_insert(routes, u, nd, C, Q, T):
    """Insert u at best position. Return True on success."""
    best_sc = -1.0; best_ri = best_pos = None
    for ri, r in enumerate(routes):
        res = _best_pos(r, u, nd, C, Q, T)
        if res and res[0] > best_sc:
            best_sc = res[0]; best_ri = ri; best_pos = res[1]
    if best_ri is not None:
        _insert(routes[best_ri], u, best_pos, nd, C)
        return True
    return False


def _repair_time(r, nd, C, T, protect):
    """Remove optional nodes (best time/profit ratio) until route fits T."""
    r.time = _rt(r.nodes, C)
    while r.time > T + 1e-7 and r.nodes:
        best_k = None; best_ratio = -1.0
        seq = [0] + r.nodes + [0]
        for k in range(1, len(seq) - 1):
            if seq[k] in protect:
                continue
            a, mid, b = seq[k - 1], seq[k], seq[k + 1]
            saved = C[a][mid] + C[mid][b] - C[a][b]
            ratio = saved / max(nd[mid].profit, 1)
            if ratio > best_ratio:
                best_ratio = ratio; best_k = k - 1
        if best_k is None:
            break
        r.nodes.pop(best_k)
        r.time = _rt(r.nodes, C)
    r.load = sum(nd[n].demand for n in r.nodes)


# ── Greedy construction ───────────────────────────────────────────────────────

def greedy_no_mandatory(model):
    """Clarke-Wright savings greedy for Problem 1."""
    N = model.num_nodes; K = model.vehicles
    Q = model.capacity;  T = model.t_max
    C = model.cost_matrix; nd = model.nodes

    cands = [i for i in range(1, N) if nd[i].demand <= Q and C[0][i] + C[i][0] <= T]

    rl  = [Route([i], nd[i].demand, C[0][i] + C[i][0]) for i in cands]
    n2r = {i: idx for idx, i in enumerate(cands)}
    act = set(range(len(rl)))

    sav = []
    for i in cands:
        for j in cands:
            if i == j: continue
            s = C[i][0] + C[0][j] - C[i][j]
            sc = (nd[i].profit + nd[j].profit) * s / (nd[i].demand + nd[j].demand + 1)
            sav.append((sc, i, j))
    sav.sort(key=lambda x: -x[0])

    for _, i, j in sav:
        ri_idx = n2r[i]; rj_idx = n2r[j]
        if ri_idx == rj_idx: continue
        ri, rj = rl[ri_idx], rl[rj_idx]
        if ri.nodes[-1] != i or rj.nodes[0] != j: continue
        if ri.load + rj.load > Q: continue
        merged = ri.nodes + rj.nodes
        t = _rt(merged, C)
        if t > T: continue
        new_idx = len(rl)
        rl.append(Route(merged, ri.load + rj.load, t))
        act.discard(ri_idx); act.discard(rj_idx); act.add(new_idx)
        for n in merged: n2r[n] = new_idx

    ranked   = sorted(act, key=lambda idx: -sum(nd[n].profit for n in rl[idx].nodes))
    selected = [rl[idx] for idx in ranked[:K]]

    visited  = {n for r in selected for n in r.nodes}
    unvis    = sorted([n for n in cands if n not in visited], key=lambda n: -nd[n].profit)

    # Cheapest-insertion fill
    changed = True
    while changed:
        changed = False; best_sc = -1.0; best_n = best_ri = best_pos = None
        for u in unvis:
            for ri, r in enumerate(selected):
                res = _best_pos(r, u, nd, C, Q, T)
                if res and res[0] > best_sc:
                    best_sc = res[0]; best_n = u; best_ri = ri; best_pos = res[1]
        if best_n is not None:
            _insert(selected[best_ri], best_n, best_pos, nd, C)
            unvis.remove(best_n); visited.add(best_n); changed = True

    for r in selected: r.sync(nd, C)
    return Solution(selected, model)


def greedy_mandatory(model):
    """
    Mandatory-first greedy for Problem 2:
    1. Build routes around mandatory nodes.
    2. Fill remaining capacity with optional nodes.
    """
    N = model.num_nodes; K = model.vehicles
    Q = model.capacity;  T = model.t_max
    C = model.cost_matrix; nd = model.nodes

    mand_set = frozenset(i for i in range(1, N) if nd[i].isMandatory)
    mand     = sorted(mand_set, key=lambda i: C[0][i] + C[i][0])   # nearest first
    optional = [i for i in range(1, N)
                if i not in mand_set and nd[i].demand <= Q and C[0][i] + C[i][0] <= T]

    routes = [Route([], 0, 0.0) for _ in range(K)]

    # --- Place every mandatory node ---
    for u in mand:
        du = nd[u].demand
        # 1) Try feasible insertion (capacity + time)
        if _try_insert(routes, u, nd, C, Q, T):
            continue
        # 2) Try ignoring time (capacity only)
        best_ex = 1e18; best_ri = best_pos = None
        for ri, r in enumerate(routes):
            if r.load + du > Q: continue
            seq = [0] + r.nodes + [0]
            for k in range(1, len(seq)):
                a, b  = seq[k - 1], seq[k]
                ex = C[a][u] + C[u][b] - C[a][b]
                if ex < best_ex:
                    best_ex = ex; best_ri = ri; best_pos = k - 1
        if best_ri is not None:
            _insert(routes[best_ri], u, best_pos, nd, C)
            _repair_time(routes[best_ri], nd, C, T, mand_set)
            continue
        # 3) Eject cheapest optional node(s) to free capacity, then insert
        ejected = False
        for ri, r in enumerate(routes):
            opt_in_r = [(nd[n].profit, i, n)
                        for i, n in enumerate(r.nodes) if n not in mand_set]
            opt_in_r.sort()  # lowest profit first
            removed = []
            for _, _, n_out in opt_in_r:
                pos = r.nodes.index(n_out)
                r.nodes.pop(pos)
                r.sync(nd, C)
                removed.append(n_out)
                if r.load + du <= Q:
                    break
            if r.load + du <= Q:
                if _try_insert([r], u, nd, C, Q, T):
                    _repair_time(r, nd, C, T, mand_set)
                    ejected = True
                    break
            # Restore ejected nodes if still not possible
            if not ejected:
                for n_out in removed:
                    r.nodes.append(n_out)
                r.sync(nd, C)
        if not ejected:
            # Last resort: force into route 0 (will be repaired)
            r = routes[0]
            r.nodes.append(u)
            r.sync(nd, C)
            _repair_time(r, nd, C, T, mand_set)

    # --- Fill with optional nodes ---
    visited = {n for r in routes for n in r.nodes}
    opt_sorted = sorted([n for n in optional if n not in visited], key=lambda n: -nd[n].profit)
    for u in opt_sorted:
        _try_insert(routes, u, nd, C, Q, T)

    for r in routes: r.sync(nd, C)
    return Solution(routes, model)


# ── Destroy operators ─────────────────────────────────────────────────────────

def _remove(sol, remove_set):
    new_sol = sol.copy()
    C = sol.model.cost_matrix; nd = sol.model.nodes
    for r in new_sol.routes:
        r.nodes = [n for n in r.nodes if n not in remove_set]
        r.sync(nd, C)
    new_sol.invalidate()
    return new_sol, list(remove_set)


def d_random(sol, q, rng, mand):
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return sol.copy(), []
    return _remove(sol, set(rng.sample(pool, min(q, len(pool)))))


def d_worst(sol, q, rng, mand):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    scored = []
    for r in sol.routes:
        seq = [0] + r.nodes + [0]
        for k in range(1, len(seq) - 1):
            node = seq[k]
            if node in mand: continue
            a, b   = seq[k - 1], seq[k + 1]
            tc     = C[a][node] + C[node][b] - C[a][b]
            scored.append((nd[node].profit / max(tc + nd[node].demand * 0.3, 1), node))
    scored.sort()
    pool = scored[:min(len(scored), q * 3)]
    removed = set()
    while len(removed) < min(q, len(pool)) and pool:
        idx  = int(rng.random() ** 3 * len(pool))
        node = pool[idx][1]
        if node not in removed: removed.add(node)
        pool.pop(idx)
    return _remove(sol, removed)


def d_related(sol, q, rng, mand):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return sol.copy(), []
    removed = [rng.choice(pool)]
    while len(removed) < min(q, len(pool)):
        ref   = rng.choice(removed)
        cands = [(C[ref][n] / max(nd[n].profit, 1), n)
                 for n in pool if n not in removed]
        if not cands: break
        cands.sort()
        removed.append(cands[int(rng.random() ** 2 * min(15, len(cands)))][1])
    return _remove(sol, set(removed))


def d_cluster(sol, q, rng, mand):
    C    = sol.model.cost_matrix
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return sol.copy(), []
    seed = rng.choice(pool)
    by_d = sorted(pool, key=lambda n: C[seed][n])
    return _remove(sol, set(by_d[:min(q, len(by_d))]))


def d_low_profit(sol, q, rng, mand):
    nd   = sol.model.nodes
    pool = [(nd[n].profit, n)
            for r in sol.routes for n in r.nodes if n not in mand]
    pool.sort()
    bottom = pool[:min(len(pool), q * 2)]
    rng.shuffle(bottom)
    return _remove(sol, {n for _, n in bottom[:min(q, len(bottom))]})


# ── Repair operators ──────────────────────────────────────────────────────────

def r_greedy(sol, removed, rng, mand, candidates):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    order = sorted(removed, key=lambda n: (n not in mand, -nd[n].profit))
    for u in order:
        best_sc = -1.0; best_ri = best_pos = None
        for ri, r in enumerate(new_sol.routes):
            res = _best_pos(r, u, nd, C, Q, T)
            if res and res[0] > best_sc:
                best_sc = res[0]; best_ri = ri; best_pos = res[1]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
    new_sol.invalidate(); return new_sol


def r_regret2(sol, removed, rng, mand, candidates):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol  = sol.copy()
    to_place = sorted(removed, key=lambda n: (n not in mand, -nd[n].profit))
    while to_place:
        best_reg = -1e18; best_n = best_ri = best_pos = None
        for u in to_place:
            sc_list = []
            for ri, r in enumerate(new_sol.routes):
                res = _best_pos(r, u, nd, C, Q, T)
                if res: sc_list.append((res[0], ri, res[1]))
            if not sc_list: continue
            sc_list.sort(reverse=True)
            reg = sc_list[0][0] - (sc_list[1][0] if len(sc_list) >= 2 else 0)
            if u in mand: reg += 1e6
            if reg > best_reg:
                best_reg = reg; best_n = u
                best_ri = sc_list[0][1]; best_pos = sc_list[0][2]
        if best_n is None: break
        _insert(new_sol.routes[best_ri], best_n, best_pos, nd, C)
        to_place.remove(best_n)
    new_sol.invalidate(); return new_sol


def r_rand_greedy(sol, removed, rng, mand, candidates):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    mpart   = [n for n in removed if n in mand]
    opart   = [n for n in removed if n not in mand]
    rng.shuffle(opart)
    for u in mpart + opart:
        best_sc = -1.0; best_ri = best_pos = None
        for ri, r in enumerate(new_sol.routes):
            res = _best_pos(r, u, nd, C, Q, T)
            if res and res[0] > best_sc:
                best_sc = res[0]; best_ri = ri; best_pos = res[1]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
    new_sol.invalidate(); return new_sol


def r_fill_unvisited(sol, removed, rng, mand, candidates):
    """After re-inserting mandatory nodes, fill with best unvisited candidates."""
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    visited = new_sol.visited()

    # Insert mandatory removed first
    for u in sorted(removed, key=lambda n: (n not in mand, -nd[n].profit)):
        if u in mand:
            best_sc = -1.0; best_ri = best_pos = None
            for ri, r in enumerate(new_sol.routes):
                res = _best_pos(r, u, nd, C, Q, T)
                if res and res[0] > best_sc:
                    best_sc = res[0]; best_ri = ri; best_pos = res[1]
            if best_ri is not None:
                _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
                visited.add(u)

    # Fill with unvisited candidates sorted by profit descending
    unvis = sorted([n for n in candidates if n not in visited], key=lambda n: -nd[n].profit)
    for u in unvis:
        best_sc = -1.0; best_ri = best_pos = None
        for ri, r in enumerate(new_sol.routes):
            res = _best_pos(r, u, nd, C, Q, T)
            if res and res[0] > best_sc:
                best_sc = res[0]; best_ri = ri; best_pos = res[1]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
            visited.add(u)

    new_sol.invalidate(); return new_sol


# ── Local search ──────────────────────────────────────────────────────────────

def ls_2opt(sol):
    """Intra-route 2-opt."""
    C = sol.model.cost_matrix; T = sol.model.t_max
    changed = True
    while changed:
        changed = False
        for r in sol.routes:
            n = len(r.nodes)
            if n < 3: continue
            improved = True
            while improved:
                improved = False
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        seg      = r.nodes[i + 1:j + 1][::-1]
                        new_nodes = r.nodes[:i + 1] + seg + r.nodes[j + 1:]
                        t = _rt(new_nodes, C)
                        if t < r.time - 1e-7 and t <= T + 1e-7:
                            r.nodes = new_nodes; r.time = t; improved = True; changed = True
    sol.invalidate(); return sol


def ls_or_opt(sol):
    """Move single nodes between routes to save time (enabling more inserts)."""
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    changed = True
    while changed:
        changed = False
        for ri, r1 in enumerate(sol.routes):
            if not r1.nodes: continue
            i = 0
            while i < len(r1.nodes):
                u  = r1.nodes[i]
                du = nd[u].demand
                seq1 = [0] + r1.nodes + [0]
                a1, b1 = seq1[i], seq1[i + 2]
                save1  = C[a1][u] + C[u][b1] - C[a1][b1]
                for rj, r2 in enumerate(sol.routes):
                    if ri == rj or r2.load + du > Q: continue
                    slack2 = T - r2.time
                    seq2   = [0] + r2.nodes + [0]
                    for k in range(1, len(seq2)):
                        a2, b2 = seq2[k - 1], seq2[k]
                        ins = C[a2][u] + C[u][b2] - C[a2][b2]
                        if ins > slack2 + 1e-7: continue
                        if save1 - ins > 1e-7:   # time saved
                            r1.nodes.pop(i); r1.sync(nd, C)
                            r2.nodes.insert(k - 1, u); r2.sync(nd, C)
                            sol.invalidate(); changed = True; break
                    if changed: break
                if changed: break
                i += 1
            if changed: break


def ls_profit_swap(sol, all_cands_by_profit, mand):
    """Replace a low-profit visited optional node with a higher-profit unvisited one."""
    C  = sol.model.cost_matrix; nd = sol.model.nodes
    Q  = sol.model.capacity;    T  = sol.model.t_max
    visited = sol.visited()
    unvis   = [u for u in all_cands_by_profit if u not in visited]
    changed = True
    while changed:
        changed = False
        for u in unvis:
            pu = nd[u].profit; du = nd[u].demand
            best_gain = 0; best_ri = best_i = best_v = None
            for ri, r in enumerate(sol.routes):
                seq = [0] + r.nodes + [0]
                for i in range(1, len(seq) - 1):
                    v = seq[i]
                    if v in mand: continue
                    gain = pu - nd[v].profit
                    if gain <= best_gain: continue
                    if r.load - nd[v].demand + du > Q: continue
                    a, b  = seq[i - 1], seq[i + 1]
                    dt    = C[a][u] + C[u][b] - C[a][v] - C[v][b]
                    if r.time + dt <= T + 1e-7:
                        best_gain = gain; best_ri = ri; best_i = i - 1; best_v = v
            if best_ri is not None:
                r = sol.routes[best_ri]
                r.nodes[best_i] = u; r.sync(nd, C)
                visited.discard(best_v); visited.add(u)
                unvis = [n for n in unvis if n != u]
                if best_v not in mand:
                    unvis.append(best_v)
                    unvis.sort(key=lambda n: -nd[n].profit)
                sol.invalidate(); changed = True; break


def ls_fill(sol, all_cands_by_profit):
    """Try to insert unvisited nodes greedily."""
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    visited = sol.visited()
    unvis   = [n for n in all_cands_by_profit if n not in visited]
    changed = True
    while changed:
        changed = False
        still = []
        for u in unvis:
            if _try_insert(sol.routes, u, nd, C, Q, T):
                visited.add(u); sol.invalidate(); changed = True
            else:
                still.append(u)
        unvis = still


def ls_inter_route_swap(sol, all_cands_by_profit, mand):
    """
    Two complementary inter-route moves applied until no further improvement.

    Move A – eject-replace-relocate:
        For each unvisited node v (highest profit first), find an optional node u
        in some route r1 such that v can fill r1 after u is removed.  Then try to
        insert the freed u into a different route r2.
        Accept when net profit gain > 0:
          u relocated → gain = profit[v]            (u survives, v is added free)
          u lost      → gain = profit[v] - profit[u] (only when positive)

    Move B – cross-route exchange:
        Swap optional node u ∈ r1 with optional node v ∈ r2 when both routes stay
        feasible and total travel time strictly decreases.  Shorter routes give
        ls_fill more slack to pack in additional nodes afterwards.
    """
    C  = sol.model.cost_matrix
    nd = sol.model.nodes
    Q  = sol.model.capacity
    T  = sol.model.t_max

    # ── Move A: eject-replace-relocate ───────────────────────────────────────
    visited = sol.visited()
    unvis   = [n for n in all_cands_by_profit if n not in visited]

    improved = True
    while improved:
        improved = False
        for v in list(unvis):
            pv = nd[v].profit
            best_gain  = 0
            best_ri    = best_pos_v = best_u_idx = None
            best_rj    = best_pos_u = None
            eject_only = False

            for ri, r1 in enumerate(sol.routes):
                for u_idx, u in enumerate(r1.nodes):
                    if u in mand:
                        continue
                    pu = nd[u].profit

                    # Hypothetical r1 with u removed
                    tmp_nodes = r1.nodes[:u_idx] + r1.nodes[u_idx + 1:]
                    tmp_r     = Route(tmp_nodes,
                                      r1.load - nd[u].demand,
                                      _rt(tmp_nodes, C))
                    res_v = _best_pos(tmp_r, v, nd, C, Q, T)
                    if res_v is None:
                        continue

                    # v fits — can we save u by relocating it to another route?
                    for rj, r2 in enumerate(sol.routes):
                        if rj == ri:
                            continue
                        res_u = _best_pos(r2, u, nd, C, Q, T)
                        if res_u is not None:
                            gain = pv   # u survives → net gain = profit[v]
                            if gain > best_gain:
                                best_gain  = gain
                                best_ri    = ri;   best_pos_v = res_v[1]
                                best_u_idx = u_idx
                                best_rj    = rj;   best_pos_u = res_u[1]
                                eject_only = False
                            break

                    # Fallback: lose u but gain v if still profitable
                    gain_no_reloc = pv - pu
                    if gain_no_reloc > best_gain:
                        best_gain  = gain_no_reloc
                        best_ri    = ri;   best_pos_v = res_v[1]
                        best_u_idx = u_idx
                        best_rj    = None
                        eject_only = True

            if best_gain > 0 and best_ri is not None:
                r1    = sol.routes[best_ri]
                u_out = r1.nodes[best_u_idx]

                # Remove u from r1 (state now matches tmp_r), then insert v
                r1.nodes.pop(best_u_idx)
                r1.sync(nd, C)
                _insert(r1, v, best_pos_v, nd, C)

                if eject_only:
                    visited.discard(u_out)
                    unvis.append(u_out)
                    unvis.sort(key=lambda n: -nd[n].profit)
                else:
                    _insert(sol.routes[best_rj], u_out, best_pos_u, nd, C)

                visited.add(v)
                unvis.remove(v)
                sol.invalidate()
                improved = True
                break   # restart from highest-profit unvisited node

    # ── Move B: cross-route exchange ─────────────────────────────────────────
    improved = True
    while improved:
        improved = False
        routes = sol.routes
        for ri in range(len(routes)):
            if improved:
                break
            r1 = routes[ri]
            for rj in range(ri + 1, len(routes)):
                if improved:
                    break
                r2 = routes[rj]
                for i, u in enumerate(r1.nodes):
                    if u in mand:
                        continue
                    if improved:
                        break
                    for j, v in enumerate(r2.nodes):
                        if v in mand:
                            continue
                        l1 = r1.load - nd[u].demand + nd[v].demand
                        l2 = r2.load - nd[v].demand + nd[u].demand
                        if l1 > Q or l2 > Q:
                            continue
                        new1 = r1.nodes[:i] + [v] + r1.nodes[i + 1:]
                        new2 = r2.nodes[:j] + [u] + r2.nodes[j + 1:]
                        t1   = _rt(new1, C)
                        t2   = _rt(new2, C)
                        if t1 > T or t2 > T:
                            continue
                        if t1 + t2 < r1.time + r2.time - 1e-7:
                            r1.nodes = new1; r1.load = l1; r1.time = t1
                            r2.nodes = new2; r2.load = l2; r2.time = t2
                            sol.invalidate()
                            improved = True
                            break


# ── ALNS ─────────────────────────────────────────────────────────────────────

def _wc(weights, rng):
    total = sum(weights); r = rng.random() * total
    for i, w in enumerate(weights):
        r -= w
        if r <= 0: return i
    return len(weights) - 1


def alns(init_sol, time_limit, mand, candidates, seed=42, do_polish=True,
         dynamic_cooling=False):
    rng   = random.Random(seed)
    model = init_sol.model
    nd    = model.nodes
    C     = model.cost_matrix
    Q     = model.capacity
    T     = model.t_max

    cands_by_profit = sorted(candidates, key=lambda n: -nd[n].profit)

    destroy_ops = [d_random, d_worst, d_related, d_cluster, d_low_profit]
    repair_ops  = [r_greedy, r_regret2, r_rand_greedy, r_fill_unvisited]

    nd_ = len(destroy_ops); nr_ = len(repair_ops)
    dw  = [1.0] * nd_;  rw  = [1.0] * nr_
    dsc = [0.0] * nd_;  rsc = [0.0] * nr_
    dct = [0]   * nd_;  rct = [0]   * nr_

    current = init_sol.copy()
    best    = init_sol.copy()
    best_p  = best.profit()
    temp    = best_p * T0_FRAC

    # Dynamic cooling for short per-seed budgets: scale so T=1 at 75% of time_limit.
    # Conservative 200 iters/s estimate. Static COOLING used for full single-seed runs.
    if dynamic_cooling and temp > 1.0:
        _target = 0.75 * time_limit * 200
        eff_cooling = math.exp(-math.log(temp) / max(_target, 1))
    else:
        eff_cooling = COOLING

    it = 0; seg = 0; start = time.time()
    print(f"  ALNS start: profit={best_p}  T0={temp:.1f}  cooling={eff_cooling:.6f}")

    while time.time() - start < time_limit:
        di = _wc(dw, rng); ri = _wc(rw, rng)
        q  = rng.randint(Q_MIN, Q_MAX)

        partial, removed = destroy_ops[di](current, q, rng, mand)
        new_sol          = repair_ops[ri](partial, removed, rng, mand, candidates)

        # Periodic local search during ALNS
        if it % 15 == 0:
            ls_2opt(new_sol)
        if it % 50 == 0:
            ls_fill(new_sol, cands_by_profit)
            ls_profit_swap(new_sol, cands_by_profit, mand)
        if it % 200 == 0 and it > 0:
            ls_inter_route_swap(new_sol, cands_by_profit, mand)
            ls_fill(new_sol, cands_by_profit)

        # Reject if mandatory nodes missing
        if mand and not mand.issubset(new_sol.visited()):
            dct[di] += 1; rct[ri] += 1; it += 1; temp *= eff_cooling; continue

        new_p = new_sol.profit(); cur_p = current.profit()
        score = 0; accepted = False

        if new_p > best_p:
            best = new_sol.copy(); best_p = new_p
            score = SCORE_BEST; accepted = True
        elif new_p > cur_p:
            score = SCORE_BETTER; accepted = True
        else:
            delta = cur_p - new_p
            if temp > 1e-8 and rng.random() < math.exp(-delta / temp):
                score = SCORE_ACCEPT; accepted = True

        if accepted: current = new_sol

        dsc[di] += score; dct[di] += 1
        rsc[ri] += score; rct[ri] += 1

        if it > 0 and it % SEGMENT_SIZE == 0:
            seg += 1
            for i in range(nd_):
                if dct[i]:
                    dw[i] = max(MIN_W, (1 - REACT) * dw[i] + REACT * dsc[i] / dct[i])
                dsc[i] = dct[i] = 0
            for i in range(nr_):
                if rct[i]:
                    rw[i] = max(MIN_W, (1 - REACT) * rw[i] + REACT * rsc[i] / rct[i])
                rsc[i] = rct[i] = 0
            el = time.time() - start
            print(f"  seg={seg:3d} | iter={it:7d} | profit={best_p:6d} | "
                  f"T={temp:6.2f} | {el:.0f}s")

        temp *= eff_cooling; it += 1

    if do_polish:
        print(f"  ALNS done ({it} iters) -- polishing...")
        ls_2opt(best)
        ls_or_opt(best)
        ls_fill(best, cands_by_profit)
        ls_profit_swap(best, cands_by_profit, mand)
        ls_inter_route_swap(best, cands_by_profit, mand)
        ls_fill(best, cands_by_profit)
        ls_2opt(best)
        ls_fill(best, cands_by_profit)
        ls_profit_swap(best, cands_by_profit, mand)
        ls_inter_route_swap(best, cands_by_profit, mand)
        ls_fill(best, cands_by_profit)
        print(f"  Final profit: {best.profit()}")
    else:
        print(f"  ALNS done ({it} iters)  profit={best.profit()}")
    return best



# ── Main ──────────────────────────────────────────────────────────────────────

def solve(instance_file, solution_file, mandatory=False, time_limit=TIME_LIMIT,
          seeds=None, seed=42):
    model = load_model(instance_file)
    N  = model.num_nodes; nd = model.nodes
    Q  = model.capacity;  T  = model.t_max; C = model.cost_matrix

    mand = frozenset(i for i in range(1, N) if nd[i].isMandatory) if mandatory else frozenset()
    if mandatory:
        print(f"  Mandatory ({len(mand)}): {sorted(mand)}")

    candidates      = [i for i in range(1, N) if nd[i].demand <= Q and C[0][i] + C[i][0] <= T]
    cands_by_profit = sorted(candidates, key=lambda n: -nd[n].profit)

    print("  Phase 1: greedy...")
    if mandatory:
        init_sol = greedy_mandatory(model)
    else:
        init_sol = greedy_no_mandatory(model)

    ok, rep = validate_solution(model, init_sol.to_route_lists(), enforce_mandatory=mandatory)
    print(f"  Greedy: profit={rep['total_profit']}  valid={ok}")
    if not ok:
        for e in rep['errors']: print(f"    ERR: {e}")

    seed_list    = seeds if seeds is not None else [seed]
    budget       = time_limit // len(seed_list)
    multi_seed   = len(seed_list) > 1   # chained restarts → use dynamic cooling per run

    print(f"  Phase 2: ALNS — {len(seed_list)} run(s) x {budget}s each"
          + (" (chained, dynamic cooling)" if multi_seed else ""))
    best_overall  = init_sol.copy()
    best_profit_o = init_sol.profit()
    current_start = init_sol

    for run, s in enumerate(seed_list):
        print(f"\n  --- Run {run + 1}/{len(seed_list)}  seed={s}  budget={budget}s  "
              f"start_profit={current_start.profit()} ---")
        candidate = alns(current_start.copy(), budget, mand, candidates,
                         seed=s, do_polish=False, dynamic_cooling=multi_seed)
        p = candidate.profit()
        if p > best_profit_o:
            best_profit_o = p
            best_overall  = candidate.copy()
        current_start = best_overall  # next run always continues from global best

    print(f"\n  Best after {len(seed_list)} run(s): profit={best_profit_o} — polishing...")
    best = best_overall
    ls_2opt(best)
    ls_or_opt(best)
    ls_fill(best, cands_by_profit)
    ls_profit_swap(best, cands_by_profit, mand)
    ls_inter_route_swap(best, cands_by_profit, mand)
    ls_fill(best, cands_by_profit)
    ls_2opt(best)
    ls_fill(best, cands_by_profit)
    ls_profit_swap(best, cands_by_profit, mand)
    ls_inter_route_swap(best, cands_by_profit, mand)
    ls_fill(best, cands_by_profit)
    print(f"  After polish: profit={best.profit()}")

    routes = best.to_route_lists()
    ok, rep = validate_solution(model, routes, enforce_mandatory=mandatory)
    print(f"  Final: profit={rep['total_profit']}  valid={ok}")
    if not ok:
        for e in rep['errors']: print(f"    ERR: {e}")
        print("  Falling back to greedy solution")
        routes = init_sol.to_route_lists()
        ok, rep = validate_solution(model, routes, enforce_mandatory=mandatory)
        print(f"  Greedy fallback: valid={ok} profit={rep['total_profit']}")

    with open(solution_file, 'w') as fh:
        for route in routes:
            fh.write(' '.join(map(str, route)) + '\n')
    print(f"  Written: {solution_file}")
    return routes, model


if __name__ == '__main__':
    instance = os.path.join(_DIR, 'ctop_main_instance.txt')

    print("=" * 64)
    print("PROBLEM 1 -- No mandatory nodes")
    print("=" * 64)
    t0 = time.time()
    _apply_params(P1_PARAMS)
    solve(instance, os.path.join(_DIR, 'solution_no_mandatory.txt'),
          mandatory=False, time_limit=TIME_LIMIT, seed=P1_PARAMS["seed"])
    print(f"  Wall time: {time.time() - t0:.1f}s\n")

    print("=" * 64)
    print("PROBLEM 2 -- Mandatory nodes")
    print("=" * 64)
    t0 = time.time()
    _apply_params(P2_PARAMS)
    solve(instance, os.path.join(_DIR, 'solution_mandatory.txt'),
          mandatory=True, time_limit=TIME_LIMIT, seeds=P2_SEEDS)
    print(f"  Wall time: {time.time() - t0:.1f}s\n")
