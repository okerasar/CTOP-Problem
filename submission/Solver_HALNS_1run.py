"""
CTOP Solver -- Hybrid ALNS (HALNS) following Hammami (2024)
5 node selection strategies · 7 removal operators · 5 insertion operators
5 local search procedures · Set Packing Problem (SPP) via Gurobi

Run strategy: 1 single long run on both P1 and P2.
The full time budget is given to a single ALNS run with a fixed seed.
"""

import random
import math
import time
import os
import sys
import argparse

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from Parser import load_model
from SolutionValidator import validate_solution
from Solver_ALNS import (
    Route, Solution,
    greedy_no_mandatory, greedy_mandatory,
    _rt, _best_pos, _insert, _try_insert, _repair_time,
)

# ── Parameters ────────────────────────────────────────────────────────────────
TIME_LIMIT       = 280        # ALNS seconds per problem (SPP uses remaining ~20s)
SEGMENT_SIZE     = 300
REACT            = 0.10
MIN_W            = 0.05
SCORE_BEST       = 8
SCORE_BETTER     = 4
SCORE_ACCEPT     = 1
T0_FRAC          = 0.05
COOLING          = 0.9997
T_MIN            = 0.0001
BETA_MAX_FRAC    = 0.15
SPP_TIME_LIMIT   = 30
POOL_MAX         = 50_000
VERBOSE_INTERVAL = 30.0
VERBOSE          = True

# Single seed for each problem — one long run
P1_SEED = 16   # tuned independently
P2_SEED = 15   # tuned independently

# Problem-specific parameters (tuned independently for each problem)
P1_PARAMS = {
    'COOLING':       0.999188,
    'T0_FRAC':       0.051659,
    'SEGMENT_SIZE':  300,
    'REACT':         0.163320,
    'BETA_MAX_FRAC': 0.192933,
    'SCORE_BEST':    15,
    'SCORE_BETTER':  6,
    'SCORE_ACCEPT':  0,
}

P2_PARAMS = {
    'COOLING':       0.999574,
    'T0_FRAC':       0.094243,
    'SEGMENT_SIZE':  600,
    'REACT':         0.209976,
    'BETA_MAX_FRAC': 0.175359,
    'SCORE_BEST':    11,
    'SCORE_BETTER':  7,
    'SCORE_ACCEPT':  3,
}


# ── Clarke-Wright greedy initialisation (P1) ─────────────────────────────────

def _cheapest_ins(r, u, nd, C, Q, T):
    """Cheapest feasible insertion of u into r; returns (extra_time, pos) or None."""
    if r.load + nd[u].demand > Q:
        return None
    best_extra = None; best_pos = None
    nodes = r.nodes
    for pos in range(len(nodes) + 1):
        prev  = 0 if pos == 0          else nodes[pos - 1]
        nxt   = 0 if pos == len(nodes) else nodes[pos]
        extra = C[prev][u] + C[u][nxt] - C[prev][nxt]
        if r.time + extra <= T:
            if best_extra is None or extra < best_extra:
                best_extra = extra; best_pos = pos
    return None if best_extra is None else (best_extra, best_pos)


def greedy_clark_wright(model):
    """Clarke-Wright Savings greedy for P1 (no mandatory nodes)."""
    nd = model.nodes; Q = model.capacity; T = model.t_max
    C  = model.cost_matrix; K = model.vehicles
    EPS = 1e-9

    cands = [i for i in range(1, model.num_nodes)
             if nd[i].demand <= Q and C[0][i] + C[i][0] <= T - EPS]

    routes = [Route([i], nd[i].demand, C[0][i] + C[i][0]) for i in cands]

    savings = []
    for i in cands:
        for j in cands:
            if i == j: continue
            s     = C[i][0] + C[0][j] - C[i][j]
            score = (nd[i].profit + nd[j].profit) * s / (nd[i].demand + nd[j].demand + 1)
            savings.append((score, i, j))
    savings.sort(key=lambda x: -x[0])

    n2r    = {c: idx for idx, c in enumerate(cands)}
    active = set(range(len(routes)))

    for _, i, j in savings:
        ri_idx, rj_idx = n2r[i], n2r[j]
        if ri_idx == rj_idx: continue
        ri, rj = routes[ri_idx], routes[rj_idx]
        if ri.nodes[-1] != i or rj.nodes[0] != j: continue
        if ri.load + rj.load > Q: continue
        merged = ri.nodes + rj.nodes
        mt     = _rt(merged, C)
        if mt > T - EPS: continue
        new_idx = len(routes)
        routes.append(Route(merged, ri.load + rj.load, mt))
        active.discard(ri_idx); active.discard(rj_idx); active.add(new_idx)
        for n in merged: n2r[n] = new_idx

    ranked   = sorted(active, key=lambda idx: -sum(nd[n].profit for n in routes[idx].nodes))
    selected = [routes[idx] for idx in ranked[:K]]
    visited  = {n for r in selected for n in r.nodes}
    unvisited = [n for n in cands if n not in visited]

    while True:
        best_score = -1.0; best = None
        for u in unvisited:
            for r_idx, r in enumerate(selected):
                res = _cheapest_ins(r, u, nd, C, Q, T)
                if res is None: continue
                extra, pos = res
                score = nd[u].profit / max(extra, 1e-9)
                if score > best_score:
                    best_score = score; best = (u, r_idx, pos)
        if best is None: break
        u, r_idx, pos = best
        r = selected[r_idx]
        r.nodes.insert(pos, u)
        r.sync(nd, C)
        unvisited.remove(u)

    for r in selected:
        _repair_time(r, nd, C, T, frozenset())

    return Solution(selected, model)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _remove(sol, remove_set):
    new_sol = sol.copy()
    C = sol.model.cost_matrix; nd = sol.model.nodes
    for r in new_sol.routes:
        r.nodes = [n for n in r.nodes if n not in remove_set]
        r.sync(nd, C)
    new_sol.invalidate()
    return new_sol, list(remove_set)


def _wc(weights, rng):
    total = sum(weights); rv = rng.random() * total
    for i, w in enumerate(weights):
        rv -= w
        if rv <= 0: return i
    return len(weights) - 1


def _q(sol, rng):
    total = sum(len(r.nodes) for r in sol.routes)
    q_max = max(2, int(BETA_MAX_FRAC * total))
    return rng.randint(2, q_max)


# ── 5 Node Selection Strategies ──────────────────────────────────────────────

def _sel_random(sol, mand, rng):
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    rng.shuffle(pool)
    return pool


def _sel_worst(sol, mand):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    scored = []
    for r in sol.routes:
        seq = [0] + r.nodes + [0]
        for k in range(1, len(seq) - 1):
            node = seq[k]
            if node in mand: continue
            a, b = seq[k - 1], seq[k + 1]
            tc   = C[a][node] + C[node][b] - C[a][b]
            scored.append((nd[node].profit / max(tc + nd[node].demand * 0.3, 1), node))
    scored.sort()
    return [n for _, n in scored]


def _sel_related(sol, mand, rng):
    C  = sol.model.cost_matrix; nd = sol.model.nodes
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return pool
    seed = rng.choice(pool)
    return sorted(pool, key=lambda n: C[seed][n] / max(nd[n].profit, 1))


def _sel_cluster(sol, mand, rng):
    C    = sol.model.cost_matrix
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return pool
    seed = rng.choice(pool)
    return sorted(pool, key=lambda n: C[seed][n])


def _sel_low_profit(sol, mand):
    nd   = sol.model.nodes
    pool = [(nd[n].profit, n) for r in sol.routes for n in r.nodes if n not in mand]
    pool.sort()
    return [n for _, n in pool]


# ── 7 Removal (Destroy) Operators ────────────────────────────────────────────

def d_random(sol, q, mand, rng):
    pool = _sel_random(sol, mand, rng)
    if not pool: return sol.copy(), []
    return _remove(sol, set(pool[:min(q, len(pool))]))


def d_worst(sol, q, mand, rng):
    pool  = _sel_worst(sol, mand)
    cands = pool[:min(len(pool), q * 3)]
    removed = set()
    while len(removed) < min(q, len(cands)) and cands:
        idx = int(rng.random() ** 3 * len(cands))
        removed.add(cands[idx]); cands.pop(idx)
    return _remove(sol, removed)


def d_related(sol, q, mand, rng):
    pool = _sel_related(sol, mand, rng)
    if not pool: return sol.copy(), []
    removed = set(); cands = list(pool)
    while len(removed) < min(q, len(cands)) and cands:
        idx = int(rng.random() ** 2 * min(15, len(cands)))
        removed.add(cands[idx]); cands.pop(idx)
    return _remove(sol, removed)


def d_cluster(sol, q, mand, rng):
    pool = _sel_cluster(sol, mand, rng)
    if not pool: return sol.copy(), []
    return _remove(sol, set(pool[:min(q, len(pool))]))


def d_low_profit(sol, q, mand, rng):
    pool   = _sel_low_profit(sol, mand)
    bottom = pool[:min(len(pool), q * 2)]
    rng.shuffle(bottom)
    return _remove(sol, set(bottom[:min(q, len(bottom))]))


def d_route_strip(sol, q, mand, rng):
    """Remove optional nodes from the route with lowest profit-per-unit-time."""
    nd = sol.model.nodes
    scored = []
    for r in sol.routes:
        opt = [n for n in r.nodes if n not in mand]
        if not opt: continue
        prof = sum(nd[n].profit for n in opt)
        scored.append((prof / max(r.time, 1), opt))
    if not scored: return sol.copy(), []
    scored.sort()
    opt = scored[0][1]
    rng.shuffle(opt)
    return _remove(sol, set(opt[:min(q, len(opt))]))


def d_shaw(sol, q, mand, rng):
    """Shaw removal: group nodes by distance + profit similarity to a seed."""
    C  = sol.model.cost_matrix; nd = sol.model.nodes
    pool = [n for r in sol.routes for n in r.nodes if n not in mand]
    if not pool: return sol.copy(), []
    seed     = rng.choice(pool)
    ps       = nd[seed].profit
    max_p    = max(nd[n].profit for n in pool) or 1
    max_d    = max(C[seed][n] for n in pool) or 1
    route_of = {n: ri for ri, r in enumerate(sol.routes) for n in r.nodes}
    r_seed   = route_of.get(seed, -1)

    def shaw_score(n):
        return (0.5 * C[seed][n] / max_d
                + 0.3 * abs(nd[n].profit - ps) / max_p
                + 0.2 * (0.0 if route_of.get(n) == r_seed else 1.0))

    ordered = sorted(pool, key=shaw_score)
    removed = set(); cands = list(ordered)
    while len(removed) < min(q, len(cands)) and cands:
        idx = int(rng.random() ** 2 * min(15, len(cands)))
        removed.add(cands[idx]); cands.pop(idx)
    return _remove(sol, removed)


_DESTROY = [d_random, d_worst, d_related, d_cluster, d_low_profit, d_route_strip, d_shaw]


# ── 5 Insertion (Repair) Operators ───────────────────────────────────────────

def r_greedy(sol, removed, mand, candidates, rng):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    for u in sorted(removed, key=lambda n: (n not in mand, -nd[n].profit)):
        best_sc = -1.0; best_ri = best_pos = None
        for ri, r in enumerate(new_sol.routes):
            res = _best_pos(r, u, nd, C, Q, T)
            if res and res[0] > best_sc:
                best_sc = res[0]; best_ri = ri; best_pos = res[1]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
    new_sol.invalidate(); return new_sol


def r_regret2(sol, removed, mand, candidates, rng):
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


def r_rand_greedy(sol, removed, mand, candidates, rng):
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


def r_fill_unvisited(sol, removed, mand, candidates, rng):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    visited = new_sol.visited()
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
    for u in sorted([n for n in candidates if n not in visited], key=lambda n: -nd[n].profit):
        best_sc = -1.0; best_ri = best_pos = None
        for ri, r in enumerate(new_sol.routes):
            res = _best_pos(r, u, nd, C, Q, T)
            if res and res[0] > best_sc:
                best_sc = res[0]; best_ri = ri; best_pos = res[1]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
            visited.add(u)
    new_sol.invalidate(); return new_sol


def r_greedy_time(sol, removed, mand, candidates, rng):
    """Insert each node at the position that minimises added travel time."""
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    new_sol = sol.copy()
    for u in sorted(removed, key=lambda n: (n not in mand, -nd[n].profit)):
        best_dt = float('inf'); best_ri = best_pos = None
        du = nd[u].demand
        for ri, r in enumerate(new_sol.routes):
            if r.load + du > Q: continue
            slack = T - r.time
            nodes = r.nodes; n = len(nodes); a = 0
            for k in range(n + 1):
                b  = nodes[k] if k < n else 0
                dt = C[a][u] + C[u][b] - C[a][b]
                if dt <= slack + 1e-7 and dt < best_dt:
                    best_dt = dt; best_ri = ri; best_pos = k
                if k < n: a = nodes[k]
        if best_ri is not None:
            _insert(new_sol.routes[best_ri], u, best_pos, nd, C)
    new_sol.invalidate(); return new_sol


_REPAIR = [r_greedy, r_regret2, r_rand_greedy, r_fill_unvisited, r_greedy_time]


# ── 5 Local Search Procedures ────────────────────────────────────────────────

def ls_2opt(sol):
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
                        seg       = r.nodes[i + 1:j + 1][::-1]
                        new_nodes = r.nodes[:i + 1] + seg + r.nodes[j + 1:]
                        t = _rt(new_nodes, C)
                        if t < r.time - 1e-7 and t <= T + 1e-7:
                            r.nodes = new_nodes; r.time = t
                            improved = True; changed = True
    sol.invalidate(); return sol


def ls_or_opt(sol):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    changed = True
    while changed:
        changed = False
        for ri, r1 in enumerate(sol.routes):
            if not r1.nodes: continue
            i = 0
            while i < len(r1.nodes):
                u = r1.nodes[i]; du = nd[u].demand
                seq1 = [0] + r1.nodes + [0]
                a1, b1 = seq1[i], seq1[i + 2]
                save1  = C[a1][u] + C[u][b1] - C[a1][b1]
                moved = False
                for rj, r2 in enumerate(sol.routes):
                    if ri == rj or r2.load + du > Q: continue
                    slack2 = T - r2.time
                    seq2   = [0] + r2.nodes + [0]
                    for k in range(1, len(seq2)):
                        a2, b2 = seq2[k - 1], seq2[k]
                        ins = C[a2][u] + C[u][b2] - C[a2][b2]
                        if ins > slack2 + 1e-7: continue
                        if save1 - ins > 1e-7:
                            r1.nodes.pop(i); r1.sync(nd, C)
                            r2.nodes.insert(k - 1, u); r2.sync(nd, C)
                            sol.invalidate(); changed = True; moved = True; break
                    if moved: break
                if moved: break
                i += 1
            if changed: break
    return sol


def ls_profit_swap(sol, cands_by_profit, mand):
    C  = sol.model.cost_matrix; nd = sol.model.nodes
    Q  = sol.model.capacity;    T  = sol.model.t_max
    visited = sol.visited()
    unvis   = [u for u in cands_by_profit if u not in visited]
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
                    a, b = seq[i - 1], seq[i + 1]
                    dt   = C[a][u] + C[u][b] - C[a][v] - C[v][b]
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
    return sol


def ls_fill(sol, cands_by_profit):
    C = sol.model.cost_matrix; nd = sol.model.nodes
    Q = sol.model.capacity;    T  = sol.model.t_max
    visited = sol.visited()
    unvis   = [n for n in cands_by_profit if n not in visited]
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
    return sol


def ls_inter_route_swap(sol, cands_by_profit, mand):
    C  = sol.model.cost_matrix; nd = sol.model.nodes
    Q  = sol.model.capacity;    T  = sol.model.t_max

    visited = sol.visited()
    unvis   = [n for n in cands_by_profit if n not in visited]
    improved = True
    while improved:
        improved = False
        for v in list(unvis):
            pv = nd[v].profit
            best_gain = 0
            best_ri = best_pos_v = best_u_idx = best_rj = best_pos_u = None
            eject_only = False
            for ri, r1 in enumerate(sol.routes):
                for u_idx, u in enumerate(r1.nodes):
                    if u in mand: continue
                    pu = nd[u].profit
                    tmp_nodes = r1.nodes[:u_idx] + r1.nodes[u_idx + 1:]
                    tmp_r     = Route(tmp_nodes, r1.load - nd[u].demand,
                                      _rt(tmp_nodes, C))
                    res_v = _best_pos(tmp_r, v, nd, C, Q, T)
                    if res_v is None: continue
                    for rj, r2 in enumerate(sol.routes):
                        if rj == ri: continue
                        res_u = _best_pos(r2, u, nd, C, Q, T)
                        if res_u is not None:
                            if pv > best_gain:
                                best_gain = pv
                                best_ri = ri; best_pos_v = res_v[1]
                                best_u_idx = u_idx
                                best_rj = rj; best_pos_u = res_u[1]
                                eject_only = False
                            break
                    if pv - pu > best_gain:
                        best_gain = pv - pu
                        best_ri = ri; best_pos_v = res_v[1]
                        best_u_idx = u_idx
                        best_rj = None; eject_only = True
            if best_gain > 0 and best_ri is not None:
                r1    = sol.routes[best_ri]
                u_out = r1.nodes[best_u_idx]
                r1.nodes.pop(best_u_idx); r1.sync(nd, C)
                _insert(r1, v, best_pos_v, nd, C)
                if eject_only:
                    visited.discard(u_out)
                    unvis.append(u_out)
                    unvis.sort(key=lambda n: -nd[n].profit)
                else:
                    _insert(sol.routes[best_rj], u_out, best_pos_u, nd, C)
                visited.add(v); unvis.remove(v)
                sol.invalidate(); improved = True; break

    improved = True
    while improved:
        improved = False
        routes = sol.routes
        for ri in range(len(routes)):
            if improved: break
            r1 = routes[ri]
            for rj in range(ri + 1, len(routes)):
                if improved: break
                r2 = routes[rj]
                for i, u in enumerate(r1.nodes):
                    if u in mand: continue
                    if improved: break
                    for j, v in enumerate(r2.nodes):
                        if v in mand: continue
                        l1 = r1.load - nd[u].demand + nd[v].demand
                        l2 = r2.load - nd[v].demand + nd[u].demand
                        if l1 > Q or l2 > Q: continue
                        new1 = r1.nodes[:i] + [v] + r1.nodes[i + 1:]
                        new2 = r2.nodes[:j] + [u] + r2.nodes[j + 1:]
                        t1   = _rt(new1, C)
                        t2   = _rt(new2, C)
                        if t1 > T or t2 > T: continue
                        if t1 + t2 < r1.time + r2.time - 1e-7:
                            r1.nodes = new1; r1.load = l1; r1.time = t1
                            r2.nodes = new2; r2.load = l2; r2.time = t2
                            sol.invalidate(); improved = True; break
    return sol


def _polish(sol, cands_by_profit, mand):
    ls_2opt(sol)
    ls_or_opt(sol)
    ls_fill(sol, cands_by_profit)
    ls_profit_swap(sol, cands_by_profit, mand)
    ls_inter_route_swap(sol, cands_by_profit, mand)
    ls_fill(sol, cands_by_profit)
    ls_2opt(sol)
    ls_fill(sol, cands_by_profit)
    ls_profit_swap(sol, cands_by_profit, mand)
    ls_inter_route_swap(sol, cands_by_profit, mand)
    ls_fill(sol, cands_by_profit)
    return sol


# ── Route Pool ───────────────────────────────────────────────────────────────

def _pool_add(pool, pool_keys, sol, nd):
    """Add all non-empty routes from sol to pool (shared across calls)."""
    for r in sol.routes:
        if not r.nodes: continue
        key = frozenset(r.nodes)
        if key in pool_keys: continue
        pool_keys.add(key)
        pool.append((list(r.nodes), sum(nd[n].profit for n in r.nodes),
                     r.load, r.time))
    while len(pool) > POOL_MAX:
        evicted = pool.pop(0)
        pool_keys.discard(frozenset(evicted[0]))


# ── Set Packing Problem (SPP) via Gurobi ─────────────────────────────────────

_SPP_VAR_LIMIT = 1800

def solve_spp(pool, model, time_limit=SPP_TIME_LIMIT):
    """Select at most K feasible routes from pool maximising profit,
    each customer visited at most once."""
    if not pool: return None
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        if VERBOSE: print("  [SPP] gurobipy not available — skipping")
        return None

    nd = model.nodes; K = model.vehicles; N = model.num_nodes

    work_pool = sorted(pool, key=lambda r: r[1], reverse=True)[:_SPP_VAR_LIMIT]
    n_routes = len(work_pool)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('TimeLimit', time_limit)
    env.start()
    m = gp.Model(env=env)
    x = m.addVars(n_routes, vtype=GRB.BINARY, name='x')

    m.setObjective(gp.quicksum(work_pool[r][1] * x[r] for r in range(n_routes)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(x[r] for r in range(n_routes)) <= K)

    cust_routes = {i: [] for i in range(1, N)}
    for r_idx, (nodes, *_) in enumerate(work_pool):
        for n in nodes:
            if 1 <= n < N:
                cust_routes[n].append(r_idx)
    for i in range(1, N):
        rs = cust_routes[i]
        if len(rs) > 1:
            m.addConstr(gp.quicksum(x[r] for r in rs) <= 1)

    m.optimize()

    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return None
    selected = [r for r in range(n_routes) if x[r].X > 0.5]
    if not selected: return None

    routes = [Route([], 0, 0.0) for _ in range(model.vehicles)]
    for idx, r_idx in enumerate(selected[:model.vehicles]):
        nodes, _, load, t_cost = work_pool[r_idx]
        routes[idx] = Route(list(nodes), load, t_cost)
    return Solution(routes, model)


# ── HALNS ─────────────────────────────────────────────────────────────────────

def halns(init_sol, time_limit, mand, candidates, seed=42, use_spp=True,
          dynamic_cooling=False, pool=None, pool_keys=None):
    rng   = random.Random(seed)
    model = init_sol.model
    nd    = model.nodes
    cands_by_profit = sorted(candidates, key=lambda n: -nd[n].profit)

    if pool is None:      pool      = []
    if pool_keys is None: pool_keys = set()

    nd_ = len(_DESTROY); nr_ = len(_REPAIR)
    dw  = [1.0] * nd_;  rw  = [1.0] * nr_
    dsc = [0.0] * nd_;  rsc = [0.0] * nr_
    dct = [0]   * nd_;  rct = [0]   * nr_

    current = init_sol.copy()
    best    = init_sol.copy()
    best_p  = best.profit()
    temp    = max(best_p * T0_FRAC, 1.0)

    if dynamic_cooling and temp > 1.0:
        _target  = 0.75 * time_limit * 30
        eff_cool = math.exp(-math.log(temp) / max(_target, 1))
    else:
        eff_cool = COOLING

    it = 0; seg = 0; reheated = False
    start      = time.time()
    last_print = start

    if VERBOSE:
        print(f"  HALNS start: profit={best_p}  T0={temp:.1f}  "
              f"cool={eff_cool:.6f}  spp={'on' if use_spp else 'off'}")

    while time.time() - start < time_limit:
        di = _wc(dw, rng); ri = _wc(rw, rng)
        q  = _q(current, rng)

        partial, removed = _DESTROY[di](current, q, mand, rng)
        new_sol          = _REPAIR[ri](partial, removed, mand, candidates, rng)

        if mand and not mand.issubset(new_sol.visited()):
            dct[di] += 1; rct[ri] += 1; it += 1; temp *= eff_cool; continue

        ls_2opt(new_sol)
        ls_fill(new_sol, cands_by_profit)

        new_p = new_sol.profit(); cur_p = current.profit()
        score = 0; accepted = False

        if new_p > best_p:
            ls_profit_swap(new_sol, cands_by_profit, mand)
            new_p = new_sol.profit()
            if new_p > best_p:
                best = new_sol.copy(); best_p = new_p
                if use_spp:
                    _pool_add(pool, pool_keys, new_sol, nd)
            score = SCORE_BEST; accepted = True
        elif new_p > cur_p:
            score = SCORE_BETTER; accepted = True
            if use_spp:
                _pool_add(pool, pool_keys, new_sol, nd)
        else:
            delta = cur_p - new_p
            if temp > 1e-8 and rng.random() < math.exp(-delta / max(temp, 1e-10)):
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
            before = best.profit()
            ls_inter_route_swap(best, cands_by_profit, mand)
            ls_fill(best, cands_by_profit)
            if best.profit() > before:
                best_p = best.profit()

        if use_spp and not reheated and temp < T_MIN:
            reheated = True
            if VERBOSE:
                print(f"  T_MIN at iter={it}, running SPP on {len(pool)} routes...")
            spp_sol = solve_spp(pool, model, SPP_TIME_LIMIT)
            if spp_sol is not None:
                spp_p = spp_sol.profit()
                if VERBOSE: print(f"  SPP reheat profit={spp_p}  best={best_p}")
                if spp_p > best_p:
                    best = spp_sol.copy(); best_p = spp_p
                current = best.copy()
            temp = max(best_p * T0_FRAC * 0.3, 1.0)

        if VERBOSE and time.time() - last_print >= VERBOSE_INTERVAL:
            last_print = time.time()
            el = last_print - start
            print(f"  seg={seg:3d} | iter={it:7d} | profit={best_p:6d} | "
                  f"T={temp:.4f} | pool={len(pool)} | {el:.0f}s")

        temp *= eff_cool; it += 1

    if VERBOSE:
        print(f"  HALNS done ({it} iters, pool={len(pool)})  best={best_p}")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def solve(instance_file, solution_file, mandatory=False, time_limit=TIME_LIMIT,
          seed=42, use_spp=True):
    model = load_model(instance_file)
    N  = model.num_nodes; nd = model.nodes
    C  = model.cost_matrix

    mand = (frozenset(i for i in range(1, N) if nd[i].isMandatory)
            if mandatory else frozenset())
    if mandatory:
        print(f"  Mandatory ({len(mand)}): {sorted(mand)}")

    candidates      = [i for i in range(1, N)
                       if nd[i].demand <= model.capacity
                       and C[0][i] + C[i][0] <= model.t_max]
    cands_by_profit = sorted(candidates, key=lambda n: -nd[n].profit)

    print("  Phase 1: greedy...")
    init_sol = greedy_mandatory(model) if mandatory else greedy_clark_wright(model)
    ok, rep  = validate_solution(model, init_sol.to_route_lists(),
                                  enforce_mandatory=mandatory)
    print(f"  Greedy: profit={rep['total_profit']}  valid={ok}")
    if not ok:
        for e in rep['errors']: print(f"    ERR: {e}")

    # Apply problem-specific tuned parameters
    _override = P1_PARAMS if not mandatory else P2_PARAMS
    _saved = {k: globals()[k] for k in _override}
    globals().update(_override)

    print(f"  Phase 2: HALNS — 1 run x {time_limit}s  seed={seed}")

    pool      = [] if use_spp else None
    pool_keys = set() if use_spp else None

    best_overall = halns(init_sol.copy(), time_limit, mand, candidates,
                         seed=seed, use_spp=use_spp, dynamic_cooling=False,
                         pool=pool, pool_keys=pool_keys)
    best_profit_o = best_overall.profit()

    if use_spp and pool:
        print(f"\n  Final SPP on {len(pool)} pooled routes...")
        spp_sol = solve_spp(pool, model, SPP_TIME_LIMIT)
        if spp_sol is not None:
            spp_p = spp_sol.profit()
            print(f"  SPP final profit={spp_p}  best={best_profit_o}")
            if spp_p > best_profit_o:
                best_overall = spp_sol; best_profit_o = spp_p

    print(f"\n  Best after run: profit={best_profit_o} — polishing...")
    _polish(best_overall, cands_by_profit, mand)
    print(f"  After polish: profit={best_overall.profit()}")

    routes = best_overall.to_route_lists()
    ok, rep = validate_solution(model, routes, enforce_mandatory=mandatory)
    print(f"  Final: profit={rep['total_profit']}  valid={ok}")
    if not ok:
        for e in rep['errors']: print(f"    ERR: {e}")
        print("  Falling back to greedy solution")
        routes = init_sol.to_route_lists()
        ok, rep = validate_solution(model, routes, enforce_mandatory=mandatory)
        print(f"  Greedy fallback: valid={ok} profit={rep['total_profit']}")

    globals().update(_saved)

    with open(solution_file, 'w') as fh:
        for route in routes:
            fh.write(' '.join(map(str, route)) + '\n')
    print(f"  Written: {solution_file}")
    return routes, model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CTOP HALNS Solver — 1 long run')
    ap.add_argument('--no-spp', action='store_true', help='Disable Gurobi SPP phase')
    args = ap.parse_args()
    use_spp = not args.no_spp

    instance = os.path.join(_DIR, 'ctop_main_instance.txt')

    print("=" * 64)
    print("PROBLEM 1 -- No mandatory nodes  [1 long run]")
    print("=" * 64)
    t0 = time.time()
    solve(instance, os.path.join(_DIR, 'solution_no_mandatory.txt'),
          mandatory=False, time_limit=TIME_LIMIT, seed=P1_SEED, use_spp=use_spp)
    print(f"  Wall time: {time.time() - t0:.1f}s\n")

    print("=" * 64)
    print("PROBLEM 2 -- Mandatory nodes  [1 long run]")
    print("=" * 64)
    t0 = time.time()
    solve(instance, os.path.join(_DIR, 'solution_mandatory.txt'),
          mandatory=True, time_limit=TIME_LIMIT, seed=P2_SEED, use_spp=use_spp)
    print(f"  Wall time: {time.time() - t0:.1f}s\n")
