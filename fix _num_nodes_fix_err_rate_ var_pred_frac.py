# good version till now

import networkx as nx
import math, os, random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime as dt

# ---------------------------- CONFIGURATION ----------------------------
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [999999999999999999999999999999999999, 10, 5, 3.33333333333333333333333333, 2.5, 2]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 50
DEBUG                = True
down_link = {}

# ---------------------------- SUBMESH HIERARCHY ----------------------------

# --- NEW: a layout-aware id mapper ---
def xy_to_id_layout(x, y, size, layout=None):
    """
    If layout is None: row-major id = x*size + y (old behavior).
    If layout is a list of length n: returns layout[x*size + y].
    """
    idx = x*size + y
    if layout is None:
        return idx
    return layout[idx]


# --- modify the submesh generators to accept a mapper ---
def generate_type1_submeshes(size, layout=None):
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        b = 2**level
        for i in range(0, size, b):
            for j in range(0, size, b):
                nodes = {
                    xy_to_id_layout(x, y, size, layout)
                    for x in range(i, min(i+b, size))
                    for y in range(j, min(j+b, size))
                }
                hierarchy[(level,2)].append(nodes)
    return hierarchy

def generate_type2_submeshes(size, layout=None):
    levels = int(math.log2(size))
    hierarchy = defaultdict(list)
    for level in range(1, levels):
        b = 2**level
        off = b//2
        for i in range(-off, size, b):
            for j in range(-off, size, b):
                nodes = {
                    xy_to_id_layout(x, y, size, layout)
                    for x in range(i, i+b)
                    for y in range(j, j+b)
                    if 0 <= x < size and 0 <= y < size
                }
                if nodes:
                    hierarchy[(level,1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size, layout=None):
    """
    Optional layout: a permutation list of length n=size*size that
    says which node id occupies each row-major (x,y) cell.
    layout=None keeps old behavior.
    """
    H = generate_type1_submeshes(size, layout=layout)
    H.update(generate_type2_submeshes(size, layout=layout))

    # level-0 equal
    H[(0,1)] = list(H[(0,2)])

    # sanity checks (unchanged)
    levels = sorted({lvl for (lvl,_) in H})
    lo, hi = levels[0], levels[-1]
    if H[(lo,1)] != H[(lo,2)]:
        raise RuntimeError("Level 0 must have identical Type-1/2 clusters")
    for lvl in levels[1:-1]:
        if H[(lvl,1)] == H[(lvl,2)]:
            raise RuntimeError(f"Level {lvl} Type-1 == Type-2; they must differ")

    # root with *all current nodes* (layout or not)
    n = size*size
    all_nodes = set(layout if layout is not None else range(n))
    root_level = hi + 1
    H[(root_level,1)].append(all_nodes)
    H[(root_level,2)].append(all_nodes)
    if H[(root_level,1)] != H[(root_level,2)]:
        raise RuntimeError("Root level must have identical Type-1/2 clusters")

    return H


def make_pred_first_layout(all_nodes, predicted):
    """
    Returns a permutation list L of all_nodes, where elements of `predicted`
    (deduped, in their current order) appear first, followed by the remaining nodes.
    The length of L must be exactly n and contain each node id exactly once.
    """
    seen = set()
    P = [v for v in predicted if v not in seen and not seen.add(v)]
    rest = [v for v in all_nodes if v not in seen]
    return P + rest



def print_clusters(H):
    print("=== Cluster Hierarchy ===")
    levels = sorted(set(lvl for (lvl, _) in H))
    for lvl in levels:
        for t in [1, 2]:
            key = (lvl, t)
            if key in H:
                print(f"Level {lvl} Type-{t}: {len(H[key])} clusters")
                for idx, cl in enumerate(H[key]):
                    print(f"  Cluster {idx}: {sorted(cl)}")
    print("="*40)



def assign_cluster_leaders(H, seed=None, prefer=None):
    """
    Choose one leader per cluster.
    - If `prefer` (an iterable of node ids) is given, we pick the leader
      from (cluster ∩ prefer) when that intersection is non-empty.
    - Otherwise we fall back to uniform random in the cluster.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    prefer = set(prefer) if prefer is not None else None

    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            if prefer:
                cand = [v for v in cl if v in prefer]
                if cand:          # prefer a predicted node inside this cluster
                    leader = rng.choice(cand)
                else:
                    leader = rng.choice(tuple(cl))
            else:
                leader = rng.choice(tuple(cl))
            M[lvl_type].append((leader, cl))
    return M


# ============================ ONLINE (TREE-STYLE) HELPERS ============================
#
# We keep the original MultiBend and PMultiBend implementations intact.
# The helpers below add an *online* PMultiBend variant where predictions are
# revealed over time and leaders are upgraded lazily (only when the spiral path
# actually visits a cluster).
#
# "Tree-style" online (matching the tree experiments) is implemented as:
#   - We start with a fixed list of paired samples (p_i, q_i).
#   - Predictions are revealed in batches over time.
#   - Within each revealed batch, we enforce that the revealed predictions do not
#     contain consecutive duplicates (a small hygiene constraint from the tree code).
#   - When a prediction is revealed, we only *commit* it as a leader for a cluster
#     if that cluster is traversed by a spiral path (lazy update).

import random

def _reorder_no_consecutive(batch, rng, *, key=lambda x: x, max_tries=80):
    """Best-effort reorder to avoid consecutive duplicates of `key(item)`.

    Note: this is purely a *presentation* order helper. In the new supervisor
    model, request serving order can be fixed while prediction *release* order
    can be permuted independently.
    """
    if len(batch) <= 1:
        return list(batch)

    best = list(batch)
    best_bad = sum(1 for i in range(1, len(best)) if key(best[i]) == key(best[i-1]))

    for _ in range(max_tries):
        cand = list(batch)
        rng.shuffle(cand)
        bad = sum(1 for i in range(1, len(cand)) if key(cand[i]) == key(cand[i-1]))
        if bad == 0:
            return cand
        if bad < best_bad:
            best, best_bad = cand, bad

    return best


def serve_requests_remove_by_id(
    vp_and_q,
    rng=None,
    *,
    min_batch_size=1,
    max_batch_size=None,
    max_extract_fraction=0.50,   # kept for backward compatibility (unused)
    max_release_fraction=0.50,   # kept for backward compatibility (unused)
    enforce_no_consecutive_vp=True,
    allow_time0_empty=True,
    time0_empty_prob=0.50,
    debug=False,
):
    """Supervisor-spec online batching with "early next batch" + "return unused".

    Input:
      vp_and_q: list of triples (req_id, p, q) in *true service order*.
        - req_id: stable identifier (usually 0..n-1)
        - p: predicted node
        - q: actual requester node

    Output:
      timeline: list of segments. Each segment is a dict with:
        - 'release': list of (req_id, p, q) items whose predictions are released
        - 'serve'   : list of (req_id, p, q) items whose requests are served

    Rules implemented (per supervisor message):
      1) Time 0 may start with an empty release (no predictions revealed).
      2) Any released batch size is in [1, floor(n/2)], where n = len(vp_and_q)
         (the offline prediction-slot list size), additionally capped by remaining.
      3) After time 0, releases are never empty.
      4) If there will be another release, we choose the release time after
         serving a *proper prefix* of the current batch (so >=1 served but < batch size),
         ensuring at least one prediction from the current batch is outstanding
         right before the next release.
      5) "Return unused" is modeled by *not advancing* past the unserved suffix;
         those items remain eligible to be included in future releases.

    Notes:
      - We preserve the request serving order exactly as vp_and_q order.
      - The "no consecutive duplicates" constraint is applied to the RELEASE order
        (a presentation/order of announcements) and does not affect serving order.
    """
    if rng is None:
        rng = random.Random()

    items = list(vp_and_q)
    n_total = len(items)
    if n_total == 0:
        return []

    # n means the total offline prediction slots (NOT remaining, NOT unique nodes)
    max_global = n_total // 2
    if max_global < 1:
        max_global = 1

    # External clamp knobs (kept) but never allow ranges outside the supervisor spec.
    min_batch_size = max(1, int(min_batch_size))
    if max_batch_size is None:
        max_batch_size = max_global
    max_batch_size = max(1, int(max_batch_size))
    max_batch_size = min(max_batch_size, max_global)
    min_batch_size = min(min_batch_size, max_batch_size)

    timeline = []

    # Time-0 may be empty.
    if allow_time0_empty and rng.random() < float(time0_empty_prob):
        timeline.append({"release": [], "serve": [], "t0": True})
        if debug:
            print("[online] time0: empty release")

    idx = 0
    while idx < n_total:
        remaining = n_total - idx

        # batch size bounds for this release
        eff_max = min(max_batch_size, max_global, remaining)
        eff_min = min_batch_size

        # If there will be another release (i.e., more than 1 request remains),
        # we need a batch size >= 2 so we can serve a proper prefix.
        if remaining > 1:
            eff_min = max(eff_min, 2)
        else:
            eff_min = 1

        if eff_min > eff_max:
            eff_min = eff_max

        batch_size = rng.randint(eff_min, eff_max) if eff_max >= 1 else 1

        # The "current batch" is the next batch_size items in the true service order.
        current_batch = items[idx : idx + batch_size]

        # Choose how many of this batch's requests to serve *before* the next release.
        # If we cannot finish the whole run after this, serve a proper prefix.
        if idx + batch_size >= n_total:
            # last batch: serve all remaining
            serve_count = remaining
        else:
            # non-last: serve 1..batch_size-1 (must leave at least 1 outstanding)
            # If batch_size == 1 (degenerate small-n case), fall back to serving 1.
            serve_count = 1 if batch_size <= 1 else rng.randint(1, batch_size - 1)

        serve_slice = items[idx : idx + serve_count]

        release_for_log = current_batch
        if enforce_no_consecutive_vp:
            release_for_log = _reorder_no_consecutive(
                current_batch,
                rng,
                key=lambda t: t[1],  # compare by predicted node p
            )

        if debug:
            ps = [p for (_i, p, _q) in current_batch]
            print(
                f"[online] release size={len(current_batch)} serve={len(serve_slice)} "
                f"remaining={remaining} p-seq={ps[:8]}{'...' if len(ps)>8 else ''}"
            )

        timeline.append(
            {
                "release": release_for_log,  # reveal predictions in this order
                "serve": serve_slice,        # serve requests in true arrival order
                "t0": False,
            }
        )

        # "Return unused" == do NOT advance past the unserved suffix.
        idx += serve_count

    return timeline



def get_spiral_online(node, leader_map, prefer=None, update=False, verbose=False):
    """Online version of get_spiral().

    If prefer is provided and a visited cluster contains any preferred node(s),
    we can (optionally) *update* that cluster's leader to a preferred node.

    Lazy update rule:
      - Only clusters on the spiral path are eligible to be updated.
      - If update=False, the spiral is computed without modifying leaders.
    """
    prefer = set(prefer) if prefer is not None else None
    path = [node]
    seen = {node}

    levels = sorted({lvl for (lvl, _) in leader_map})
    for lvl in levels:
        for t in (2, 1):
            key = (lvl, t)
            if key not in leader_map:
                continue

            clusters = leader_map[key]
            for idx, (leader, cl) in enumerate(clusters):
                if node not in cl:
                    continue

                chosen = leader
                if prefer is not None:
                    cand = [v for v in cl if v in prefer]
                    if cand:
                        # deterministic choice to reduce extra randomness
                        chosen = min(cand)
                        if update and chosen != leader:
                            clusters[idx] = (chosen, cl)

                if chosen not in seen:
                    path.append(chosen)
                    seen.add(chosen)
                break

    # ensure root leader is appended
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    if verbose:
        print(f"[online spiral] {node} -> {path}")
    return path


def publish_online(owner, leader_map, prefer=None, update=False):
    """Publish using the online spiral (optionally updating leaders lazily)."""
    global down_link
    down_link.clear()
    sp = get_spiral_online(owner, leader_map, prefer=prefer, update=update)
    for i in range(len(sp) - 1, 0, -1):
        down_link[sp[i]] = sp[i - 1]


def _single_request_costs_online(r, owner, leader_map, G, prefer=None, weight=None, trace=False):
    """Same cost decomposition as _single_request_costs(), but with online spirals."""
    publish_online(owner, leader_map, prefer=prefer, update=True)

    sp = get_spiral_online(r, leader_map, prefer=prefer, update=True)
    up_hops = 0
    intersection = None
    for i in range(len(sp) - 1):
        u, v = sp[i], sp[i + 1]
        d = nx.shortest_path_length(G, u, v, weight=weight)
        up_hops += d
        if v in down_link:
            intersection = v
            break
    if intersection is None:
        intersection = sp[-1]

    down_hops = 0
    cur = intersection
    seen = {cur}
    while cur != owner:
        nxt = down_link.get(cur)
        if nxt is None or nxt in seen:
            down_hops += nx.shortest_path_length(G, cur, owner, weight=weight)
            break
        down_hops += nx.shortest_path_length(G, cur, nxt, weight=weight)
        seen.add(nxt)
        cur = nxt

    obj = nx.shortest_path_length(G, owner, r, weight=weight)
    if trace:
        print(f"[online costs] r={r} up={up_hops} down={down_hops} obj={obj}")
    return up_hops, down_hops, obj


def measure_stretch_move_online(requesters, owner, leader_map, G, prefer=None, weight=None, trace=True):
    """MOVE stretch for the online PMultiBend variant."""
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs_online(
            r, owner, leader_map, G, prefer=prefer, weight=weight, trace=trace and DEBUG
        )
        total_alg += (up + down + obj)
        total_opt += obj
        owner = r
        publish_online(owner, leader_map, prefer=prefer, update=True)

    return (total_alg / total_opt) if total_opt > 0 else 1.0



# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------

def get_spiral(node, leader_map, verbose=False):
    path = [node]
    seen = {node}
    # iterate by level, and inside each level visit Type-1 (2) then Type-2 (1)
    levels = sorted({lvl for (lvl, _) in leader_map})
    for lvl in levels:
        for t in (2, 1):  # Type-1, then Type-2
            key = (lvl, t)
            if key not in leader_map: 
                continue
            for leader, cl in leader_map[key]:
                if node in cl:
                    if leader not in seen:
                        path.append(leader)
                        seen.add(leader)
                    break  # there is only one containing cluster per (lvl,t)

    # force-append the root leader so publish always reaches the root
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key  = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    
    if verbose:
        print(f"Spiral upward path for {node}: {path}")
    return path


def publish(owner, leader_map):
    """
    Store downward pointers at every cluster leader the owner belongs to.
    """
    global down_link
    down_link.clear()
    sp = get_spiral(owner, leader_map) # spiral is bottom-up
    # Store a pointer at every leader in the spiral (except the last, which is the owner)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]

# ---------------------------- STRETCH MEASUREMENT ----------------------------


def _single_request_costs(r, owner, leader_map, G, weight=None, trace=False):
    """
    Return (up_hops, down_hops, obj) for a single request r given current owner.
    obj is the shortest-path distance from owner -> r.
    """
    # publish at the current owner
    publish(owner, leader_map)

    # ---- UP: climb r's spiral until it hits a published pointer
    sp = get_spiral(r, leader_map)
    up_hops = 0
    intersection = None
    for i in range(len(sp) - 1):
        u, v = sp[i], sp[i+1]
        d = nx.shortest_path_length(G, u, v, weight=weight)
        up_hops += d
        if v in down_link:
            intersection = v
            break
    if intersection is None:
        intersection = sp[-1]  # root leader

    # ---- DOWN: follow pointers; if missing, jump directly to owner
    down_hops = 0
    cur = intersection
    seen = {cur}
    while cur != owner:
        nxt = down_link.get(cur)
        if nxt is None or nxt in seen:
            down_hops += nx.shortest_path_length(G, cur, owner, weight=weight)
            break
        down_hops += nx.shortest_path_length(G, cur, nxt, weight=weight)
        seen.add(nxt)
        cur = nxt

    # ---- object forwarding hop
    obj = nx.shortest_path_length(G, owner, r, weight=weight)

    if trace:
        print(f"[costs] r={r} up={up_hops} down={down_hops} obj={obj}")

    return up_hops, down_hops, obj



def measure_stretch(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    LOOKUP stretch (your current metric):
      sum(UP+DOWN) / sum(OPT), where OPT = shortest(owner, requester).
    Kept for backward compatibility.
    """
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs(r, owner, leader_map, G, weight, trace and DEBUG)
        total_up_down += (up + down)
        total_opt     += obj
        owner = r  # move ownership for next step (your model does this)
        publish(owner, leader_map)

    return (total_up_down / total_opt) if total_opt > 0 else 1.0



def measure_stretch_move(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    MOVE stretch (what MultiBend reports):
      sum(UP + DOWN + OBJ) / sum(OBJ)  ==  1 + sum(UP+DOWN)/sum(OBJ)
    """
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs(r, owner, leader_map, G, weight, trace and DEBUG)
        total_alg += (up + down + obj)
        total_opt += obj
        owner = r
        publish(owner, leader_map)

    return (total_alg / total_opt) if total_opt > 0 else 1.0




# ---------------------------- GRAPH LOADING ----------------------------

def load_graph(dfile):
    G = nx.read_graphml(os.path.join("graphs","grid",dfile))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- ERRORS & HELPERS ----------------------------

def _choose_vp_pairwise_bounded(G, k, *, divisor=2.0, weight="weight", rng=None, max_tries=80):
    """Pick a length-k prediction list Vp with a pairwise distance bound.

    Supervisor constraint:
      For any two predicted nodes chosen into Vp, dist_G(u,v) <= diameter(G)/c.

    - divisor=c. If divisor is None or <= 1, we treat it as "no restriction".
    - Sampling is with replacement (so duplicates are allowed, matching the
      original code's use of random.choices).

    Strategy (best effort):
      Repeatedly try to build Vp by maintaining the intersection of radius-threshold
      balls around already chosen nodes. This guarantees the pairwise bound.
      If we cannot complete within max_tries, fall back to unconstrained sampling.
    """
    if rng is None:
        rng = random.Random()

    nodes = list(G.nodes())
    if k <= 0:
        return []
    if not nodes:
        return []

    # No restriction requested.
    if divisor is None:
        return [rng.choice(nodes) for _ in range(k)]
    try:
        divisor = float(divisor)
    except Exception:
        divisor = 1.0
    if divisor <= 1.0:
        return [rng.choice(nodes) for _ in range(k)]

    diam = nx.diameter(G, weight=weight)
    threshold = float(diam) / divisor
    if threshold >= float(diam):
        return [rng.choice(nodes) for _ in range(k)]

    # Memoize balls we compute so retries are cheaper.
    ball_cache = {}

    def ball(u):
        if u in ball_cache:
            return ball_cache[u]
        dmap = nx.single_source_dijkstra_path_length(G, u, cutoff=threshold, weight=weight)
        s = set(dmap.keys())
        ball_cache[u] = s
        return s

    for _ in range(max_tries):
        first = rng.choice(nodes)
        vp = [first]
        allowed = set(ball(first))
        if not allowed:
            continue
        for _i in range(1, k):
            # Sampling with replacement: allowed is a set; we don't remove selected.
            if not allowed:
                break
            nxt = rng.choice(tuple(allowed))
            vp.append(nxt)
            allowed &= ball(nxt)
        if len(vp) == k:
            return vp

    # Fall back: keep the experiment running even if the constraint is too strict.
    return [rng.choice(nodes) for _ in range(k)]

def choose_Vp(G, fraction, *, vp_pairwise_divisor=None):
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle the nodes to ensure randomness
    total_nodes = len(nodes)
    vp_size = int(total_nodes * fraction) # Fraction of nodes to be chosen as Vp
    original_Vp = _choose_vp_pairwise_bounded(
        G,
        vp_size,
        divisor=vp_pairwise_divisor,
        rng=random.Random(),
    )
    random.shuffle(original_Vp)  # Shuffle Vp to ensure randomness

    reduced_Vp = set(original_Vp)

    reduced_Vp = list(reduced_Vp)  # Convert back to a list for indexing
    random.shuffle(reduced_Vp)  # Shuffle Vp to ensure randomness

    # Choose an owner node that is not in Vp
    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining))

    # Insert owner to reduced_Vp list at a random position
    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = reduced_Vp.copy()
    S = set(S)  # Convert to a set for uniqueness

    return original_Vp


def count_duplicates(input_list):
    """
    Checks for duplicate elements in a list and returns their counts.

    Args:
        input_list: The list to check for duplicates.

    Returns:
        A dictionary where keys are the duplicate elements and values are their counts.
        Returns an empty dictionary if no duplicates are found.
    """
    counts = Counter(input_list)
    duplicates = {element: count for element, count in counts.items() if count > 1}
    return duplicates


def sample_Q_within_diameter(G, Vp, error_cutoff):
    diam = nx.diameter(G, weight='weight')
    max_iter = 100000  # Maximum number of iterations to avoid infinite loop

    for attempt in range(1, max_iter+1):
        # 1) sample one random reachable node per v
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam/error_cutoff), weight="weight")
            Q.append(random.choice(list(dist_map.keys())))

        # 2) compute overlap
        dup_counts = count_duplicates(Q)
        # extra dups = sum of (count - 1) for each duplicated element
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100

        # 3) check if within tolerance
        if current_overlap <= 100:
            return Q

    random.shuffle(Q)  # Shuffle the list to ensure randomness
    return Q

def sample_actual(G, Vp, error):
    diam = nx.diameter(G)
    act = []
    for v in Vp:
        cutoff = int(diam/error) if error>0 else diam
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
        act.append(random.choice(list(lengths.keys())))
    return act

def calculate_error(Vp, Q, G_example):
    diameter_of_G = nx.diameter(G_example, weight='weight')  # Compute the diameter of the graph G_example
    errors = []
    for req, pred in zip(Q, Vp):
        # Using NetworkX to compute the shortest path length in tree T.
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)
        # print(f"\nDistance between request node {req} and predicted node {pred} is {dist}, error = {error:.4f}")
    
    # print("Diameter of G:", diameter_of_G)
    # print("Diameter of T:", diameter_of_T)
    total_max_error = max(errors) if errors else 0
    total_min_error = min(errors) if errors else 0
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}\nOverall max error (max_i(distance_in_G / diameter_G)) = {total_max_error:.4f}{RESET}")
    print(f"{RED}\nOverall min error (min_i(distance_in_G / diameter_G)) = {total_min_error:.4f}{RESET}")
    return total_max_error


# -----calaulate error stats to get max, min and avg error-------------
def calculate_error_stats(Vp, Q, G):
    diam = nx.diameter(G, weight='weight')
    vals = []
    for req, pred in zip(Q, Vp):
        d = nx.shortest_path_length(G, req, pred, weight='weight')
        vals.append(d / diam)
    if not vals:
        return 0.0, 0.0, 0.0
    return max(vals), min(vals), float(sum(vals)/len(vals))



# ---------------------------- SIMULATION ----------------------------

def simulate(graph_file, use_move_stretch=False):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    measure_fn = measure_stretch_move if use_move_stretch else measure_stretch
    results = []
    for error in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            owner = random.choice(list(G.nodes()))
            publish(owner, leaders)

            for _ in range(NUM_TRIALS):
                pred = choose_Vp(G, frac)
                act  = sample_Q_within_diameter(G, pred, error)
                err  = calculate_error(pred, act, G)

                for req in act:
                    if req == owner:
                        continue
                    stretch = measure_fn([req], owner, leaders, G, trace=False)

                    err_rate = 0.0 if error > 15 else round(1.0 / error, 1)
                    results.append((frac, err_rate, err, stretch))

                    owner = req
                    publish(owner, leaders)
    return results


# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    # use your global list here
    xvals = PREDICTION_FRACTIONS

    plt.figure(figsize=(12,6))

    # ---------------- Error vs Fraction ----------------
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.ErrRate == e ]
        plt.plot(sub.Frac, sub.Err, '-o', label=f"{e:.1f} Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Max)")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    plt.ylim(0, max(ERROR_VALUES_2)*1.1)
    plt.grid(True)
    plt.legend(loc="upper right")

    # ---------------- Stretch vs Fraction ----------------
    plt.subplot(1,2,2)
    # for e in ERROR_VALUES_2:
    #     sub = avg[ avg.ErrRate == e ]
    #     plt.plot(sub.Frac, sub.Str, '-o', label=f"{e:.1f} Stretch")

    # loop over each unique ErrRate in your aggregated frame

    for err_rate, group in avg.groupby("ErrRate"):
        plt.plot(
            group.Frac, 
            group.Str, 
            "-o", 
            label=f"{err_rate:.1f} Stretch"
        )

    


    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    # plt.ylim(0.95, 1.05)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# ---------------------------- HALVING MODE ADD-ONS ----------------------------

def halving_counts(n: int):
    """Return [n/2, n/4, ..., 1] using integer division."""
    counts = []
    k = n // 2
    while k >= 1:
        counts.append(k)
        k //= 2
    return counts

def choose_Vp_halving(G, k: int, *, vp_pairwise_divisor=None):
    """
    Halving-mode version of choose_Vp. Intentionally mirrors the original:
    - uses random.choices (with replacement)
    - dedups to build reduced_Vp
    - selects an owner not in reduced_Vp and inserts it (for parity with original)
    - returns *original_Vp* (same as your original choose_Vp)
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle for randomness
    total_nodes = len(nodes)

    # mirror the original: clamp to [1, n] and use as-is
    vp_size = int(k)
    vp_size = max(1, min(vp_size, total_nodes))

    # with replacement (exactly like your original)
    original_Vp = _choose_vp_pairwise_bounded(
        G,
        vp_size,
        divisor=vp_pairwise_divisor,
        rng=random.Random(),
    )
    random.shuffle(original_Vp)

    # build reduced_Vp & owner (same as original structure)
    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining)) if remaining else random.choice(nodes)

    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = set(reduced_Vp)  # kept for parity; not returned/used (same as original)

    # IMPORTANT: return signature matches your original choose_Vp
    return original_Vp


def simulate_halving(graph_file, use_move_stretch=False, vp_pairwise_divisor=None):
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    measure_fn = measure_stretch_move if use_move_stretch else measure_stretch
    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n
            owner = random.choice(list(G.nodes()))
            publish(owner, leaders)

            for _ in range(NUM_TRIALS):
                P = choose_Vp_halving(G, k, vp_pairwise_divisor=vp_pairwise_divisor)
                Q = sample_Q_within_diameter(G, P, error)
                err = calculate_error(P, Q, G)

                for req in Q:
                    if req == owner:
                        continue
                    stretch = measure_fn([req], owner, leaders, G, trace=False)
                    err_rate = 0.0 if error > 15 else round(1.0 / error, 1)
                    results.append((frac, err_rate, err, stretch))
                    owner = req
                    publish(owner, leaders)
    return results



def plot_results_halving_counts(results, n=None, title_suffix=" (halving mode)", use_log_x=True):
    """
    Same input schema as before. Shows x-axis as |P| counts (1,2,4,...,n/2).
    Left error subplot y-axis fixed to 0..0.5 with ticks at 0.1.
    """
    from matplotlib.ticker import FormatStrFormatter  # local import

    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])

    # infer n if not provided
    if n is None:
        min_frac = df["Frac"].min()
        n = int(round(1.0 / min_frac)) if min_frac > 0 else None
        if not n:
            raise ValueError("Please pass n explicitly to plot_results_halving_counts().")

    # k = |P|
    df["Count"] = (df["Frac"] * n).round().astype(int)
    avg = df.groupby(["Count","ErrRate"]).mean().reset_index()

    # tick positions: 1, 2, 4, 8, ..., floor(n/2) (whatever appeared in results)
    xvals = sorted(df["Count"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True, sharex=True)

    # helper: style both subplots
    def prettify_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("|P| (n/2, n/4, ..., 1)")
        # use log2 x-scale to spread small counts
        if use_log_x:
            try:
                ax.set_xscale("log", base=2)   # mpl >=3.3
            except TypeError:
                ax.set_xscale("log", basex=2)  # older mpl
        # show only our counts as ticks (keeps labels clean)
        ax.xaxis.set_major_locator(FixedLocator(xvals))
        ax.set_xticklabels([str(x) for x in xvals], rotation=0)
        # margins + lighter grid only on Y
        xmin = max(min(xvals), 1)
        ax.set_xlim(xmin * (0.98 if use_log_x else 0.9), max(xvals) * 1.02)
        ax.margins(x=0.02, y=0.08)
        ax.grid(True, axis="y", alpha=0.35)

    # -------- Error vs |P| --------
    ax = axes[0]
    for e in ERROR_VALUES_2:
        sub = avg[avg.ErrRate == e].sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Err, "-o", label=f"{e:.1f} Error")
    prettify_axis(ax, "Error vs |P| (halving)"+title_suffix, "Error")

    # --- lock y-axis to 0..0.5 with ticks every 0.1 ---
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.legend(loc="upper right")

    # -------- Stretch vs |P| --------
    ax = axes[1]
    for err_rate, group in avg.groupby("ErrRate"):
        sub = group.sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Str, "-o", label=f"{err_rate:.1f} Stretch")
    prettify_axis(ax, "Stretch vs |P| (halving)"+title_suffix, "Stretch")
    ax.legend(loc="upper right")

    plt.show()

# -----halving mode add ons complete-------------------------------


# ---------multibend comparison add-ons----------------------------
def multibend_move_sequence_stretch(requesters, owner, leader_map, G, weight=None, trace=False):
    """
    MultiBend *move* stretch for a requester sequence.
    Algorithm cost per request = UP + DOWN + (owner -> requester shortest path).
    OPT lower bound per request = (owner -> requester shortest path).
    Returns (sum alg costs) / (sum OPT costs).
    """
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue

        # publish at current owner (directory path)
        publish(owner, leader_map)

        # ----- UP: climb r's spiral until it hits the published path
        sp = get_spiral(r, leader_map)
        up_hops = 0
        intersection = None
        for i in range(len(sp) - 1):
            u, v = sp[i], sp[i+1]
            d = nx.shortest_path_length(G, u, v, weight=weight)
            up_hops += d
            if v in down_link:
                intersection = v
                break
        if intersection is None:
            intersection = sp[-1]

        # ----- DOWN: follow directory pointers (fallback = direct)
        down_hops = 0
        cur = intersection
        seen = {cur}
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                down_hops += nx.shortest_path_length(G, cur, owner, weight=weight)
                break
            down_hops += nx.shortest_path_length(G, cur, nxt, weight=weight)
            seen.add(nxt)
            cur = nxt

        # ----- object forwarding hop
        obj = nx.shortest_path_length(G, owner, r, weight=weight)

        total_alg += (up_hops + down_hops + obj)
        total_opt += obj

        if trace:
            print(f"[MB seq] r={r} up={up_hops} down={down_hops} obj={obj} "
                  f"⇒ {(up_hops+down_hops+obj)/obj:.3f}")

        # move object to r and continue
        owner = r

    return (total_alg / total_opt) if total_opt > 0 else 1.0



def simulate_halving_compare_multibend(graph_file):
    """
    Compare: Our (prediction-aware leaders) vs MB (baseline random leaders).
    Returns rows: (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr).
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    # Build one SPATIAL hierarchy once (row-major); baseline leaders are random.
    H_base = build_mesh_hierarchy(size)

    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n

            for _ in range(NUM_TRIALS):

                # (optional) independent owner each trial
                owner_start = random.choice(list(G.nodes()))

                # sample predicted set and request sequence
                # new predictions & requests each trial
                P = choose_Vp_halving(G, k)
                Q = sample_Q_within_diameter(G, P, error)

                # REBUILD leaders fresh each trial
                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)

                # OUR directory = same clusters, but leaders biased to P for this batch
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

                # batch error stats (repeated per-row for grouping later)
                err_max, err_min, err_avg = calculate_error_stats(P, Q, G)
                err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

                owner_ours = owner_start
                owner_mb   = owner_start
                publish(owner_ours, leaders_pred)
                publish(owner_mb,   leaders_base)

                for req in Q:
                    if req == owner_ours:
                        owner_ours = req
                        owner_mb   = req
                        publish(owner_ours, leaders_pred)
                        publish(owner_mb,   leaders_base)
                        continue

                    our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                    owner_ours = req
                    publish(owner_ours, leaders_pred)

                    mb_str  = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                    owner_mb = req
                    publish(owner_mb, leaders_base)

                    results.append((frac, err_rate, err_max, err_min, err_avg, our_str, mb_str))

                # align owners for next trial at this (error,k)
                # owner_start = owner_ours

    return results


def simulate_halving_compare_multibend_with_online_tree_style(graph_file,
                                                             min_batch_size=1,
                                                             max_batch_size=None,
                                                             vp_pairwise_divisor=None,
                                                             seed=None):
    """\
    Tree-style ONLINE comparison on a single graph.

    Compares three schemes on the *same served request order*:
      1) MultiBend baseline: random leaders (fixed for the trial)
      2) PMultiBend (offline): leaders biased to the full prediction list P (known at t=0)
      3) Online-PMultiBend (tree-style): predictions are revealed in batches over time.
         Leaders are *lazily upgraded* to predicted nodes only on spiral paths that
         are traversed while serving requests.

    Returns rows:
      (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr, OnlineStr)

    Notes about the online model implemented here:
      - A trial samples a paired list of (predicted, actual) nodes: (P[i], Q[i]).
      - The online process groups these pairs into batches using serve_requests_remove_by_id.
      - Before serving the requests in a batch, all predictions in that batch are
        revealed and added to the known prediction set.
      - During serving, leaders are upgraded only for clusters encountered along
        spiral paths (search and publish), matching the "change only if traversed"
        requirement.
    """

    rng = random.Random(seed) if seed is not None else random.Random()

    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    H_base = build_mesh_hierarchy(size)
    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n

            for _ in range(NUM_TRIALS):
                owner_start = rng.choice(list(G.nodes()))

                # Sample paired predictions and actual requests.
                P = choose_Vp_halving(G, k, vp_pairwise_divisor=vp_pairwise_divisor)
                Q = sample_Q_within_diameter(G, P, error)

                # Error stats for this (P,Q) batch.
                err_max, err_min, err_avg = calculate_error_stats(P, Q, G)
                err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

                # Build leader maps.
                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)
                leaders_online = assign_cluster_leaders(H_base, seed=None, prefer=None)

                # Build online serving batches (tree-style).
                VpAndQ = [(i, P[i], Q[i]) for i in range(len(Q))]
                batches = serve_requests_remove_by_id(
                    VpAndQ,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    rng=rng,
                )

                # Owners are tracked separately per scheme.
                owner_ours = owner_start
                owner_mb = owner_start
                owner_on = owner_start

                # Online revealed prediction set.
                revealed = set()

                # Initialize directory pointers for each scheme.
                publish(owner_ours, leaders_pred)
                publish(owner_mb, leaders_base)
                publish_online(owner_on, leaders_online, revealed)

                for seg in batches:
                    # Reveal this segment's predictions (tree-style release).
                    for _, p, _q in seg["release"]:
                        revealed.add(p)

                    # Serve requests in the true arrival order for all three schemes.
                    for _idx, _p, req in seg["serve"]:
                        # Skip no-op requests (owner already at requester).
                        if req == owner_ours:
                            owner_ours = req
                            owner_mb = req
                            owner_on = req
                            publish(owner_ours, leaders_pred)
                            publish(owner_mb, leaders_base)
                            publish_online(owner_on, leaders_online, revealed)
                            continue

                        our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                        owner_ours = req
                        publish(owner_ours, leaders_pred)

                        mb_str = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                        owner_mb = req
                        publish(owner_mb, leaders_base)

                        on_str = measure_stretch_move_online([req], owner_on, leaders_online, G,
                                                             prefer=revealed, trace=False)
                        owner_on = req
                        publish_online(owner_on, leaders_online, revealed)

                        results.append((frac, err_rate, err_max, err_min, err_avg,
                                        our_str, mb_str, on_str))

    return results




def plot_mb_vs_ours_per_error(results, n, err_levels=None, use_log_x=False, save=False, prefix="mb_vs_ours"):
    """
    Makes 1x2 figures like your reference image, one figure per error cutoff.
    Left:  Error vs |P| (halving)  — Y-axis fixed to 0..0.5 with ticks at 0.1.
    Right: Our stretch vs MultiBend move stretch
    """
    from matplotlib.ticker import FormatStrFormatter  # local import

    if err_levels is None:
        err_levels = ERROR_VALUES_2

    df = pd.DataFrame(results, columns=["Frac","ErrRate","Err","OurStr","MBStr"])
    df["Count"] = (df["Frac"] * n).round().astype(int)

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.groupby("Count").mean().reset_index().sort_values("Count")
        xvals = avg["Count"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

        # -------- Left: Error vs |P| --------
        ax = axes[0]
        ax.plot(avg["Count"], avg["Err"], "-o", label=f"Prediction Error ≤ {e:.1f}")
        ax.set_title("Error vs Fraction of Predicted Nodes")
        ax.set_ylabel("Average of Max Error")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(xvals)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="upper left")

        # --- lock y-axis to 0..0.5 with ticks every 0.1 ---
        ax.set_ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.51, 0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # -------- Right: Stretch comparison --------
        ax = axes[1]
        x = avg["Count"].to_numpy(dtype=float)

        ax.plot(x*0.985, avg["OurStr"], "-o", label=f"Our Stretch ≤ {e:.1f}", zorder=3)
        ax.plot(x*1.015, avg["MBStr"], "--s", label=f"MultiBend Stretch ≤ {e:.1f}", alpha=0.85, zorder=2)

        ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_title("Our Stretch vs Multibend Stretch")
        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try: ax.set_xscale("log", base=2)
            except TypeError: ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")

        if save:
            out = f"{prefix}_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180)
            print("saved:", out)
        else:
            plt.show()



# ---------multibend comparison add-ons complete----------------------------

# ========== Excel helpers (add these) ======================================

def save_compare_results_to_excel(results, n, filename, graph_file=None):
    """
    Save simulate_halving_compare_multibend() results to an .xlsx file.
    Sheets:
      - 'raw': one row per request; includes ErrMax, ErrMin, ErrAvg
      - 'avg': mean aggregated by (Count, ErrRate) for all metrics
      - 'meta': run metadata (n, graph, trials, timestamp, etc.)
    """
    # Support both schemas:
    #   (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr)
    #   (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr, OnlineStr)
    if results and len(results[0]) == 8:
        cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","MBStr","OnlineStr"]
    else:
        cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","MBStr"]
    df   = pd.DataFrame(results, columns=cols)
    df["Count"] = (df["Frac"] * n).round().astype(int)

    # group means for every numeric column except Frac
    group_cols = ["Count","ErrRate"]
    avg = df.groupby(group_cols, as_index=False).mean(numeric_only=True)

    meta = {
        "n": n,
        "graph_file": graph_file or "",
        "num_trials": NUM_TRIALS,
        "prediction_fracs": ",".join(map(str, PREDICTION_FRACTIONS)),
        "error_levels": ",".join(map(str, ERROR_VALUES_2)),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "notes": "ErrMax/ErrMin/ErrAvg are per-batch stats copied to each request row."
    }

    with pd.ExcelWriter(filename, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="raw", index=False)
        avg.to_excel(xw, sheet_name="avg", index=False)
        pd.DataFrame([meta]).to_excel(xw, sheet_name="meta", index=False)
    print(f"saved Excel: {filename}")



# def plot_mb_vs_ours_from_excel(filename, use_avg=True, err_levels=None,
#                                use_log_x=False, save=False, prefix="mb_vs_ours_from_xlsx",
#                                error_metric="ErrAvg"):
#     from matplotlib.ticker import FormatStrFormatter

#     meta = pd.read_excel(filename, sheet_name="meta")
#     n = int(meta["n"].iloc[0])

#     if use_avg:
#         df = pd.read_excel(filename, sheet_name="avg")
#     else:
#         raw = pd.read_excel(filename, sheet_name="raw")
#         if "Count" not in raw.columns:
#             raw["Count"] = (raw["Frac"] * n).round().astype(int)
#         df = raw.groupby(["Count","ErrRate"], as_index=False).mean(numeric_only=True)

#     if err_levels is None:
#         err_levels = ERROR_VALUES_2

#     for e in err_levels:
#         sub = df[df.ErrRate == e].copy()
#         if sub.empty:
#             continue
#         avg = sub.sort_values("Count")

#         # x = avg["Count"].to_numpy(dtype=float)
#         x_labels = avg["Count"].to_numpy()
#         x_pos = np.arange(len(x_labels))

#         cmap = plt.get_cmap('tab20')
#         # fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
#         fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#         fig.tight_layout(pad=1.1)

#         # ---- Stretch only ----
        
#         # ax.plot(x_pos + 0.015, avg["MBStr"], "--s", label="MultiBend", alpha=0.85, zorder=2)
#         # ax.plot(x_pos - 0.015, avg["OurStr"], "-o", label="PMultiBend", zorder=3)
#         # if "OnlineStr" in avg.columns:
#         #     ax.plot(x_pos, avg["OnlineStr"], ":^", label="Online PMultiBend", alpha=0.9, zorder=1)

#         # MultiBend: orange + inverted triangle

#         has_online = "OnlineStr" in avg.columns

#         ax.plot(
#             x_pos, avg["MBStr"],
#             marker="v", linestyle="-.",
#             label="MultiBend",
#             color=cmap(2),
#             linewidth=1,
#             zorder=2
#         )

#         ax.plot(
#             x_pos, avg["OurStr"],
#             marker="^", linestyle=":",
#             label="PMultiBend",
#             color=cmap(4),
#             linewidth=1,
#             zorder=3
#         )

#         if has_online:
#             ax.plot(
#                 x_pos, avg["OnlineStr"],
#                 marker=".", linestyle="-",
#                 label="OPMultiBend",
#                 color=cmap(0),
#                 linewidth=1,
#                 zorder=1
#             )

#         ymax = float(np.nanmax([
#             avg["OurStr"].max(),
#             avg["MBStr"].max(),
#             avg["OnlineStr"].max() if "OnlineStr" in avg.columns else np.nan,
#         ]))
#         ax.set_ylim(0, ymax * 1.05)

#         ax.set_ylabel("Stretch")
#         ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
#         if use_log_x:
#             try: ax.set_xscale("log", base=2)
#             except TypeError: ax.set_xscale("log", basex=2)
        
#         # ax.set_xticks(x)
#         ax.set_xticks(x_pos)
#         ax.set_xticklabels(x_labels)

#         ax.grid(False)
#         ax.legend(loc="lower right")

#         if save:
#             out = f"{prefix}_stretch_only_err_{str(e).replace('.','p')}.png"
#             plt.savefig(out, dpi=180)
#             print("saved:", out)
#         else:
#             plt.show()


def plot_mb_vs_ours_from_excel(
    filename, use_avg=True, err_levels=None,
    use_log_x=False, save=False, prefix="mb_vs_ours_from_xlsx",
    error_metric="ErrAvg"
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    if use_avg:
        df = pd.read_excel(filename, sheet_name="avg")
    else:
        raw = pd.read_excel(filename, sheet_name="raw")
        if "Count" not in raw.columns:
            raw["Count"] = (raw["Frac"] * n).round().astype(int)
        df = raw.groupby(["Count", "ErrRate"], as_index=False).mean(numeric_only=True)

    if err_levels is None:
        err_levels = ERROR_VALUES_2

    # ---- LaTeX-like sizing for 3-up subfigures (smaller ticks/labels) ----
    BASE = 9  # if still big, set to 8

    # Render the plot near its final printed size (so it won't be shrunk a lot in LaTeX)
    FIG_W_IN = 2.35
    FIG_H_IN = FIG_W_IN * (5 / 7)  # keep 7:5 aspect ratio

    rc = {
        "font.size": BASE,
        "axes.labelsize": BASE,
        "xtick.labelsize": BASE - 1,
        "ytick.labelsize": BASE - 1,
        "legend.fontsize": BASE - 1,
        "lines.linewidth": 1.1,
        "lines.markersize": 4.0,
    }

    cmap = plt.get_cmap("tab20")

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.sort_values("Count")

        x_labels = avg["Count"].to_numpy()
        x_pos = np.arange(len(x_labels))

        has_online = "OnlineStr" in avg.columns

        with plt.rc_context(rc):
            fig, ax = plt.subplots(1, 1, figsize=(FIG_W_IN, FIG_H_IN), dpi=300)

            ax.plot(
                x_pos, avg["MBStr"],
                marker="v", linestyle="-.",
                label="MultiBend",
                color=cmap(2),
                zorder=2
            )

            ax.plot(
                x_pos, avg["OurStr"],
                marker="^", linestyle=":",
                label="PMultiBend",
                color=cmap(4),
                zorder=3
            )

            if has_online:
                ax.plot(
                    x_pos, avg["OnlineStr"],
                    marker=".", linestyle="-",
                    label="OPMultiBend",
                    color=cmap(0),
                    zorder=1
                )

            ymax = float(np.nanmax([
                avg["OurStr"].max(),
                avg["MBStr"].max(),
                avg["OnlineStr"].max() if has_online else np.nan,
            ]))
            ax.set_ylim(0, ymax * 1.05)

            # Shorter labels (put "among n" in caption instead if needed)
            ax.set_ylabel("Stretch", labelpad=2)
            ax.set_xlabel("Predicted nodes", labelpad=2)

            if use_log_x:
                try:
                    ax.set_xscale("log", base=2)
                except TypeError:
                    ax.set_xscale("log", basex=2)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in x_labels])

            ax.grid(False)

            # Make ticks less dominant
            ax.tick_params(axis="both", which="major", pad=1, length=3)

            # Compact legend
            ax.legend(
                loc="lower right",
                frameon=True,
                borderpad=0.25,
                labelspacing=0.2,
                handletextpad=0.4,
                handlelength=2.2
            )

            fig.tight_layout(pad=0.05)

            if save:
                out = f"{prefix}_stretch_only_err_{str(e).replace('.','p')}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print("saved:", out)
            else:
                plt.show()




# ========================================================================



# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":


    # halving + multibend comparison + (TREE-STYLE) online PMultiBend + Excel
    # run the comparison once
    res_cmp = simulate_halving_compare_multibend_with_online_tree_style(
        "64grid_diameter14test.edgelist",
        min_batch_size=1,
        max_batch_size=None,
        seed=None,
    )

    # save raw + averaged with three error stats
    save_compare_results_to_excel(
        res_cmp,
        n=64,
        filename="fix64nodefixerrrate_varypredfrac_diamover2.xlsx",
        graph_file="64grid_diameter14test.edgelist",
    )

    # later: plot from Excel; pick which error to show on the left
    # plot_mb_vs_ours_from_excel("mb_compare_256.xlsx", use_avg=True, use_log_x=True, error_metric="ErrAvg")
    # # or
    plot_mb_vs_ours_from_excel("fix64nodefixerrrate_varypredfrac_diamover2.xlsx", use_avg=True, use_log_x=False, error_metric="ErrMax")
    # plot_stretch_minmax_avg_from_excel("esai_test_random_leaders_mb_compare_64.xlsx", use_avg=True, use_log_x=True, error_metric="ErrMax")

    # # or
    # plot_mb_vs_ours_from_excel("mb_compare_256.xlsx", use_avg=True, use_log_x=True, error_metric="ErrMin")

