# good version (grid graphs, halving mode + MB comparison, no cost diff)

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
ERROR_VALUES = [
    999999999999999999999999999999999999,  # effectively "infinite" cutoff
    10, 5, 3.33333333333333333333333333, 2.5, 2
]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 50
DEBUG                = True
down_link = {}

# ---------------------------- SUBMESH HIERARCHY ----------------------------

def xy_to_id_layout(x, y, size, layout=None):
    """
    If layout is None: row-major id = x*size + y (old behavior).
    If layout is a list of length n: returns layout[x*size + y].
    """
    idx = x*size + y
    if layout is None:
        return idx
    return layout[idx]


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
                if cand:
                    leader = rng.choice(cand)
                else:
                    leader = rng.choice(tuple(cl))
            else:
                leader = rng.choice(tuple(cl))
            M[lvl_type].append((leader, cl))
    return M


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
    sp = get_spiral(owner, leader_map)  # spiral is bottom-up
    # Store a pointer at every leader in the spiral (except the last, which is the owner)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]


# ---------------------------- STRETCH MEASUREMENT ----------------------------

def _single_request_costs(r, owner, leader_map, G, weight=None, trace=False):
    """
    Return (up_hops, down_hops, obj) for a single request r given current owner.
    obj is the shortest-path distance from owner -> r (in the graph G).
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

    # ---- object forwarding hop (in G)
    obj = nx.shortest_path_length(G, owner, r, weight=weight)

    if trace:
        print(f"[costs] r={r} up={up_hops} down={down_hops} obj={obj}")

    return up_hops, down_hops, obj


def measure_stretch(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    LOOKUP stretch:
      sum(UP+DOWN) / sum(OPT), where OPT = shortest(owner, requester) in G.
    """
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs(r, owner, leader_map, G, weight, trace and DEBUG)
        total_up_down += (up + down)
        total_opt     += obj
        owner = r  # move ownership for next step
        publish(owner, leader_map)

    return (total_up_down / total_opt) if total_opt > 0 else 1.0


def measure_stretch_move(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    MOVE stretch (MultiBend metric):
      sum(UP + DOWN + OBJ) / sum(OBJ)
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
      If we cannot complete within max_tries, we fall back to unconstrained sampling.
    """
    if rng is None:
        rng = random.Random()

    nodes = list(G.nodes())
    if k <= 0 or not nodes:
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
            if not allowed:
                break
            nxt = rng.choice(tuple(allowed))
            vp.append(nxt)
            allowed &= ball(nxt)
        if len(vp) == k:
            return vp

    # Fall back: keep experiments running even if the constraint is too strict.
    return [rng.choice(nodes) for _ in range(k)]


def choose_Vp(G, fraction, *, vp_pairwise_divisor=2.0):
    """
    Fraction-based prediction set (used by non-halving simulate()).

    Updated per supervisor:
      - Offline Vp generation enforces pairwise distance bound:
          dist(u,v) <= diameter(G) / vp_pairwise_divisor
        with default divisor=2.0, and divisor<=1 or None meaning "no restriction".
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)
    total_nodes = len(nodes)
    vp_size = int(total_nodes * fraction)

    original_Vp = _choose_vp_pairwise_bounded(
        G,
        vp_size,
        divisor=vp_pairwise_divisor,
        rng=random.Random(),
    )
    random.shuffle(original_Vp)

    # Keep the rest of the function structure unchanged (even though S isn't used),
    # to preserve downstream expectations and code flow.
    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining)) if remaining else random.choice(nodes)

    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = set(reduced_Vp)  # not used, but kept for parity

    return original_Vp

def count_duplicates(input_list):
    counts = Counter(input_list)
    duplicates = {element: count for element, count in counts.items() if count > 1}
    return duplicates


def sample_Q_within_diameter(G, Vp, error_cutoff):
    diam = nx.diameter(G, weight='weight')
    max_iter = 100000

    for _ in range(1, max_iter+1):
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(
                G, v, cutoff=float(diam/error_cutoff), weight="weight"
            )
            Q.append(random.choice(list(dist_map.keys())))

        dup_counts = count_duplicates(Q)
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100

        if current_overlap <= 100:
            return Q

    random.shuffle(Q)
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
    diameter_of_G = nx.diameter(G_example, weight='weight')
    errors = []
    for req, pred in zip(Q, Vp):
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)

    total_max_error = max(errors) if errors else 0
    total_min_error = min(errors) if errors else 0
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}\nOverall max error (max_i(distance_in_G / diameter_G)) = {total_max_error:.4f}{RESET}")
    print(f"{RED}\nOverall min error (min_i(distance_in_G / diameter_G)) = {total_min_error:.4f}{RESET}")
    return total_max_error


def calculate_error_stats(Vp, Q, G):
    """
    Per-batch error stats (used in MB comparison).
    """
    diam = nx.diameter(G, weight='weight')
    vals = []
    for req, pred in zip(Q, Vp):
        d = nx.shortest_path_length(G, req, pred, weight='weight')
        vals.append(d / diam)
    if not vals:
        return 0.0, 0.0, 0.0
    return max(vals), min(vals), float(sum(vals)/len(vals))


# ---------------------------- SIMPLE SIMULATION (fraction-based) ----------------------------

def simulate(graph_file, use_move_stretch=False, vp_pairwise_divisor=2.0):
    """
    Original fraction-based simulation (not the halving+MB compare one).
    """
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
                pred = choose_Vp(G, frac, vp_pairwise_divisor=vp_pairwise_divisor)
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


def plot_results(results):
    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    xvals = PREDICTION_FRACTIONS

    plt.figure(figsize=(12,6))

    # Error vs Fraction
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

    # Stretch vs Fraction
    plt.subplot(1,2,2)
    for err_rate, group in avg.groupby("ErrRate"):
        plt.plot(group.Frac, group.Str, "-o", label=f"{err_rate:.1f} Stretch")

    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------------------------- HALVING MODE ----------------------------

def halving_counts(n: int):
    """Return [n/2, n/4, ..., 1] using integer division."""
    counts = []
    k = n // 2
    while k >= 1:
        counts.append(k)
        k //= 2
    return counts


def choose_Vp_halving(G, k: int, *, vp_pairwise_divisor=2.0):
    """
    Halving-mode version of choose_Vp. Uses k predictions, with replacement.

    Updated per supervisor:
      - Offline Vp generation enforces pairwise distance bound:
          dist(u,v) <= diameter(G) / vp_pairwise_divisor
        with default divisor=2.0, and divisor<=1 or None meaning "no restriction".
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)
    total_nodes = len(nodes)

    vp_size = int(k)
    vp_size = max(1, min(vp_size, total_nodes))

    original_Vp = _choose_vp_pairwise_bounded(
        G,
        vp_size,
        divisor=vp_pairwise_divisor,
        rng=random.Random(),
    )
    random.shuffle(original_Vp)

    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining)) if remaining else random.choice(nodes)

    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = set(reduced_Vp)  # not used

    return original_Vp

def simulate_halving(graph_file, use_move_stretch=False, vp_pairwise_divisor=2.0):
    """
    Halving mode without MultiBend comparison (older simple version).
    """
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
    Shows x-axis as |P| counts (1,2,4,...,n/2). Left error, right stretch.
    """
    from matplotlib.ticker import FormatStrFormatter

    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])

    if n is None:
        min_frac = df["Frac"].min()
        n = int(round(1.0 / min_frac)) if min_frac > 0 else None
        if not n:
            raise ValueError("Please pass n explicitly to plot_results_halving_counts().")

    df["Count"] = (df["Frac"] * n).round().astype(int)
    avg = df.groupby(["Count","ErrRate"]).mean().reset_index()

    xvals = sorted(df["Count"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True, sharex=True)

    def prettify_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("|P| (n/2, n/4, ..., 1)")
        if use_log_x:
            try:
                ax.set_xscale("log", base=2)
            except TypeError:
                ax.set_xscale("log", basex=2)
        ax.xaxis.set_major_locator(FixedLocator(xvals))
        ax.set_xticklabels([str(x) for x in xvals], rotation=0)
        xmin = max(min(xvals), 1)
        ax.set_xlim(xmin * (0.98 if use_log_x else 0.9), max(xvals) * 1.02)
        ax.margins(x=0.02, y=0.08)
        ax.grid(True, axis="y", alpha=0.35)

    # Error vs |P|
    ax = axes[0]
    for e in ERROR_VALUES_2:
        sub = avg[avg.ErrRate == e].sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Err, "-o", label=f"{e:.1f} Error")
    prettify_axis(ax, "Error vs |P| (halving)"+title_suffix, "Error")
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc="upper right")

    # Stretch vs |P|
    ax = axes[1]
    for err_rate, group in avg.groupby("ErrRate"):
        sub = group.sort_values("Count", ascending=True)
        ax.plot(sub.Count, sub.Str, "-o", label=f"{err_rate:.1f} Stretch")
    prettify_axis(ax, "Stretch vs |P| (halving)"+title_suffix, "Stretch")
    ax.legend(loc="upper right")

    plt.show()


# --------- MultiBend comparison ----------------------------

def multibend_move_sequence_stretch(requesters, owner, leader_map, G, weight=None, trace=False):
    """
    MultiBend *move* stretch for a requester sequence.
    Algorithm cost per request = UP + DOWN + (owner -> requester shortest path in G).
    OPT per request = (owner -> requester shortest path in G).
    Returns (sum alg costs) / (sum OPT costs).
    """
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue

        publish(owner, leader_map)

        # UP
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

        # DOWN
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

        # OBJ
        obj = nx.shortest_path_length(G, owner, r, weight=weight)

        total_alg += (up_hops + down_hops + obj)
        total_opt += obj

        if trace:
            print(f"[MB seq] r={r} up={up_hops} down={down_hops} obj={obj} "
                  f"⇒ {(up_hops+down_hops+obj)/obj:.3f}")

        owner = r

    return (total_alg / total_opt) if total_opt > 0 else 1.0


def simulate_halving_compare_multibend(graph_file, vp_pairwise_divisor=2.0):
    """
    Compare: Our (prediction-aware leaders) vs MB (baseline random leaders).
    Returns rows:
        (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr)
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    # One spatial hierarchy (grid), reused across trials
    H_base = build_mesh_hierarchy(size)

    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n

            for _ in range(NUM_TRIALS):

                owner_start = random.choice(list(G.nodes()))

                P = choose_Vp_halving(G, k, vp_pairwise_divisor=vp_pairwise_divisor)
                Q = sample_Q_within_diameter(G, P, error)

                # Random leaders (MB baseline)
                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)
                # Prediction-aware leaders (our scheme)
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

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

                    # our stretch for this request
                    our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                    owner_ours = req
                    publish(owner_ours, leaders_pred)

                    # MultiBend baseline stretch for this request
                    mb_str  = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                    owner_mb = req
                    publish(owner_mb, leaders_base)

                    results.append(
                        (frac, err_rate, err_max, err_min, err_avg, our_str, mb_str)
                    )

    return results




# ======================= Tree-styled ONLINE helpers =======================

def _reorder_no_consecutive(batch, rng, *, key=lambda x: x, max_tries=80):
    """Best-effort reorder to avoid consecutive duplicates of `key(item)`.

    In the supervisor model, request serving order can be fixed while prediction
    *release* order can be permuted independently (this helper only affects the
    RELEASE order).
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
    VpAndQ,
    rng=None,
    *,
    min_batch_size=1,
    max_batch_size=None,
    max_extract_fraction=0.5,  # kept for backward compatibility (unused)
    max_release_fraction=0.5,  # kept for backward compatibility (unused)
    enforce_no_consecutive_vp=True,
    allow_time0_empty=True,
    time0_empty_prob=0.50,
    debug=False,
):
    """Supervisor-spec online batching with "early next batch" + "return unused".

    Input:
      VpAndQ: list of triples (req_id, vp, q) in *true service order*.

    Output:
      timeline: list of segments. Each segment is a dict with:
        - 'release': list of (req_id, vp, q) items whose predictions are released
        - 'serve'  : list of (req_id, vp, q) items whose requests are served

    Rules implemented:
      1) Time 0 may start with an empty release.
      2) Any released batch size is in [1, floor(n/2)], where n = len(VpAndQ)
         (the offline prediction-slot list size), additionally capped by remaining.
      3) After time 0, releases are never empty.
      4) If there will be another release, we choose the release time after
         serving a *proper prefix* of the current batch (so >=1 served but < batch size),
         ensuring at least one prediction from the current batch is outstanding
         right before the next release.
      5) "Return unused" is modeled by *not advancing* past the unserved suffix;
         those items remain eligible to be included in future releases.

    Notes:
      - Serving order is preserved exactly as the order in VpAndQ.
      - The "no consecutive duplicates" constraint is applied to the RELEASE order.
    """
    rng = rng or random.Random()

    items = list(VpAndQ)
    n_total = len(items)
    if n_total == 0:
        return []

    max_global = n_total // 2
    if max_global < 1:
        max_global = 1

    # Clamp knobs but never allow ranges outside the supervisor spec.
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

        eff_max = min(max_batch_size, max_global, remaining)
        eff_min = min_batch_size

        # If there will be another release, batch_size must be >=2
        # so we can serve a proper prefix and leave at least one outstanding.
        if remaining > 1:
            eff_min = max(eff_min, 2)
        else:
            eff_min = 1

        if eff_min > eff_max:
            eff_min = eff_max

        batch_size = rng.randint(eff_min, eff_max) if eff_max >= 1 else 1
        current_batch = items[idx : idx + batch_size]

        # How many to serve before the next release?
        if idx + batch_size >= n_total:
            serve_count = remaining  # last segment: serve all
        else:
            serve_count = 1 if batch_size <= 1 else rng.randint(1, batch_size - 1)

        serve_slice = items[idx : idx + serve_count]

        release_for_log = current_batch
        if enforce_no_consecutive_vp:
            release_for_log = _reorder_no_consecutive(
                current_batch,
                rng,
                key=lambda t: t[1],  # compare by predicted node vp
            )

        timeline.append({"release": release_for_log, "serve": serve_slice, "t0": False})

        if debug:
            print(f"[online] release={len(current_batch)} serve={len(serve_slice)} remaining={remaining}")

        # Return-unused: do NOT advance past unserved suffix.
        idx += serve_count

    return timeline


def get_spiral_online(node, leader_map, known_pred, rng=None, verbose=False):
    """Spiral path, but with *lazy* leader upgrades.

    If the current cluster contains any node in known_pred, we switch the leader
    to a random node from (cluster ∩ known_pred) *when this cluster is visited*.

    This matches the intended online behavior: leaders only change if the path
    actually traverses that cluster.
    """
    rng = rng or random.Random()
    known_pred = set(known_pred) if known_pred else set()

    path = [node]
    seen = {node}

    levels = sorted({lvl for (lvl, _) in leader_map})
    for lvl in levels:
        for t in (2, 1):
            key = (lvl, t)
            if key not in leader_map:
                continue
            for idx, (leader, cl) in enumerate(leader_map[key]):
                if node in cl:
                    if known_pred:
                        cand = [v for v in cl if v in known_pred]
                        if cand and leader not in cand:
                            new_leader = rng.choice(cand)
                            leader_map[key][idx] = (new_leader, cl)
                            leader = new_leader
                    if leader not in seen:
                        path.append(leader)
                        seen.add(leader)
                    break

    # ensure root leader is last
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key  = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    if verbose:
        print(f"[online spiral] {node}: {path}")
    return path


def publish_online(owner, leader_map, known_pred, rng=None):
    """Publish using the online spiral (supports lazy leader upgrades)."""
    global down_link
    down_link.clear()
    sp = get_spiral_online(owner, leader_map, known_pred, rng=rng)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]


def _single_request_costs_online(r, owner, leader_map, known_pred, rng, G, weight=None, trace=False):
    """Same cost decomposition as _single_request_costs(), but online-aware."""
    publish_online(owner, leader_map, known_pred, rng=rng)

    # UP
    sp = get_spiral_online(r, leader_map, known_pred, rng=rng)
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

    # DOWN
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

    # OBJ
    obj = nx.shortest_path_length(G, owner, r, weight=weight)

    if trace:
        print(f"[online costs] r={r} up={up_hops} down={down_hops} obj={obj}")

    return up_hops, down_hops, obj


def measure_stretch_move_online(requesters, owner, leader_map, known_pred, G,
                                rng=None, weight=None, trace=True):
    """MOVE stretch for Online-PMultiBend (same formula as PMultiBend)."""
    rng = rng or random.Random()

    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue
        up, down, obj = _single_request_costs_online(
            r, owner, leader_map, known_pred, rng, G, weight,
            trace and DEBUG
        )
        total_alg += (up + down + obj)
        total_opt += obj

        owner = r
        publish_online(owner, leader_map, known_pred, rng=rng)

    return (total_alg / total_opt) if total_opt > 0 else 1.0


def simulate_halving_compare_multibend_online_tree(graph_file,
                                                   min_batch_size=1,
                                                   max_batch_size=None,
                                                   enforce_no_consecutive_vp=True,
                                                   vp_pairwise_divisor=2.0,
                                                   allow_time0_empty=True,
                                                   time0_empty_prob=0.50):
    """Three-way comparison for *tree-styled online* setting.

    Compares:
      - PMultiBend (offline predictions known at start)
      - Online-PMultiBend (predictions revealed over time in batches; lazy leader upgrades)
      - MultiBend (baseline random leaders)

    Returns rows:
      (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, OnlineStr, MBStr)

    Important: In tree-styled online, the *request order* is the flattened batch schedule
    produced by serve_requests_remove_by_id(). We use the same order for all three
    algorithms so the curves are directly comparable.
    """
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
                rng = random.Random()  # per-trial randomness

                owner_start = rng.choice(list(G.nodes()))

                # sample predictions & actuals (paired by index)
                P = choose_Vp_halving(G, k, vp_pairwise_divisor=vp_pairwise_divisor)
                Q = sample_Q_within_diameter(G, P, error)

                VpAndQ = [(i+1, P[i], Q[i]) for i in range(len(P))]
                timeline = serve_requests_remove_by_id(
                    VpAndQ,
                    rng=rng,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    enforce_no_consecutive_vp=enforce_no_consecutive_vp,
                    allow_time0_empty=allow_time0_empty,
                    time0_empty_prob=time0_empty_prob,
                )

                # fixed per-batch error stats
                err_max, err_min, err_avg = calculate_error_stats(P, Q, G)
                err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

                # leader maps
                leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)      # MB
                leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)         # offline PMB
                leaders_on   = assign_cluster_leaders(H_base, seed=None, prefer=None)      # online starts random

                known_pred = set()

                owner_mb   = owner_start
                owner_ours = owner_start
                owner_on   = owner_start

                publish(owner_mb, leaders_base)
                publish(owner_ours, leaders_pred)
                publish_online(owner_on, leaders_on, known_pred, rng=rng)

                # serve in timeline order
                for seg in timeline:
                    # predictions in this release become known at the time of release
                    for (_, vp, _) in seg["release"]:
                        known_pred.add(vp)

                    for (_, vp, req) in seg["serve"]:
                        if req == owner_ours:
                            owner_ours = req
                            owner_mb   = req
                            owner_on   = req
                            publish(owner_ours, leaders_pred)
                            publish(owner_mb,   leaders_base)
                            publish_online(owner_on, leaders_on, known_pred, rng=rng)
                            continue

                        # Offline PMultiBend
                        our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                        owner_ours = req
                        publish(owner_ours, leaders_pred)

                        # Online PMultiBend
                        on_str = measure_stretch_move_online([req], owner_on, leaders_on, known_pred, G,
                                                             rng=rng, trace=False)
                        owner_on = req
                        publish_online(owner_on, leaders_on, known_pred, rng=rng)

                        # MultiBend baseline
                        mb_str = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                        owner_mb = req
                        publish(owner_mb, leaders_base)

                        results.append((frac, err_rate, err_max, err_min, err_avg, our_str, on_str, mb_str))

    return results

# --------- Excel helpers ----------------------------

def save_compare_results_to_excel(results, n, filename, graph_file=None):
    """Save compare results to .xlsx.

    Supports both:
      - 2-way results: (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, MBStr)
      - 3-way results: (Frac, ErrRate, ErrMax, ErrMin, ErrAvg, OurStr, OnlineStr, MBStr)

    Sheets:
      - raw: one row per served request
      - avg: aggregated by (Count, ErrRate)
      - meta: metadata (n, graph, trials, timestamp)
    """
    if not results:
        raise ValueError("No results to save.")

    row_len = len(results[0])
    if row_len == 7:
        cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","MBStr"]
    elif row_len == 8:
        cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","OnlineStr","MBStr"]
    else:
        raise ValueError(f"Unexpected results row length: {row_len}")

    df = pd.DataFrame(results, columns=cols)
    df["Count"] = (df["Frac"] * n).round().astype(int)

    agg = {
        "ErrMax": "max",
        "ErrMin": "min",
        "ErrAvg": "mean",
        "OurStr": "mean",
        "MBStr":  "mean",
    }
    if "OnlineStr" in df.columns:
        agg["OnlineStr"] = "mean"

    avg = df.groupby(["Count", "ErrRate"], as_index=False).agg(agg)

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


def plot_mb_vs_ours_from_excel(filename, use_avg=True, err_levels=None,
                               use_log_x=False, save=False,
                               prefix="mb_vs_ours_from_xlsx",
                               error_metric="ErrAvg"):
    """For each error cutoff, plot stretch vs |P|.

    If the Excel file contains OnlineStr, we plot three curves:
      PMultiBend (offline), Online-PMultiBend, MultiBend.
    """
    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    if use_avg:
        df = pd.read_excel(filename, sheet_name="avg")
    else:
        raw = pd.read_excel(filename, sheet_name="raw")
        if "Count" not in raw.columns:
            raw["Count"] = (raw["Frac"] * n).round().astype(int)
        df = raw.groupby(["Count","ErrRate"], as_index=False).mean(numeric_only=True)

    if err_levels is None:
        err_levels = ERROR_VALUES_2

    has_online = "OnlineStr" in df.columns

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.sort_values("Count")
        x = avg["Count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

        ax.plot(x*0.985, avg["OurStr"], "-o",
                label=f"PMultiBend (err ≤ {e:.1f})", zorder=3)

        if has_online:
            ax.plot(x, avg["OnlineStr"], "-^",
                    label="Online-PMultiBend", alpha=0.9, zorder=2.5)

        ax.plot(x*1.015, avg["MBStr"], "--s",
                label="MultiBend", alpha=0.85, zorder=2)

        ymax = float(np.nanmax([
            avg["OurStr"].max(),
            avg["MBStr"].max(),
            avg["OnlineStr"].max() if has_online else 0
        ]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try:
                ax.set_xscale("log", base=2)
            except TypeError:
                ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(False)
        ax.legend(loc="lower right")
        ax.set_title(f"Stretch vs |P| (err ≤ {e:.1f})")

        if save:
            out = f"{prefix}_stretch_only_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180)
            print("saved:", out)
            plt.close(fig)
        else:
            plt.show()


# --------- NEW: stretch vs error cutoff for fixed |P| -----------------

# def plot_stretch_vs_error_for_fixed_count_from_excel(filename, k,
#                                                      use_log_x=False,
#                                                      save=False,
#                                                      prefix="stretch_vs_error_k"):
#     """Plot stretch vs error cutoff for a fixed |P| = k.

#     If OnlineStr exists in the Excel file, the plot includes Online-PMultiBend.
#     """
#     meta = pd.read_excel(filename, sheet_name="meta")
#     n = int(meta["n"].iloc[0])

#     raw = pd.read_excel(filename, sheet_name="raw")
#     if "Count" not in raw.columns:
#         raw["Count"] = (raw["Frac"] * n).round().astype(int)

#     df_k = raw[raw["Count"] == k].copy()
#     if df_k.empty:
#         print(f"No data rows with |P| = {k} in file {filename}")
#         return

#     avg = (df_k
#            .groupby("ErrRate", as_index=False)
#            .mean(numeric_only=True)
#            .sort_values("ErrRate"))

#     x = avg["ErrRate"].to_numpy(float)
#     has_online = "OnlineStr" in avg.columns

#     cmap = plt.get_cmap('tab20')
#     # fig, ax = plt.subplots(1, 1, figsize=(7,5), constrained_layout=True)
#     fig, ax = plt.subplots(1, 1, figsize=(7,5))
#     fig.tight_layout(pad=1.2)

#     # ax.plot(x, avg["MBStr"], "--s", label="MultiBend", alpha=0.85, zorder=2)
#     # ax.plot(x, avg["OurStr"], "-o", label="PMultiBend", zorder=3)

#     # if has_online:
#     #     ax.plot(x, avg["OnlineStr"], "-^", label="Online-PMultiBend", alpha=0.9, zorder=2.5)


#     # MultiBend: orange dashdot + inverted triangle
#     # ax.plot(
#     #     x, avg["MBStr"],
#     #     linestyle="-.", marker="v",
#     #     color="tab:orange",
#     #     label="MultiBend",
#     #     alpha=0.85, zorder=2,
#     #     markersize=8, linewidth=0.8
#     # )

#     # # PMultiBend: green dotted + triangle
#     # ax.plot(
#     #     x, avg["OurStr"],
#     #     linestyle=":", marker="^",
#     #     color="tab:green",
#     #     label="PMultiBend",
#     #     alpha=0.9, zorder=3,
#     #     markersize=8, linewidth=0.8
#     # )

#     # # Online / OPMultiBend: blue solid + small dot
#     # if has_online:
#     #     ax.plot(
#     #         x, avg["OnlineStr"],
#     #         linestyle="-", marker="o",
#     #         color="tab:blue",
#     #         label="OPMultiBend",      # rename if you want
#     #         alpha=0.9, zorder=2.5,
#     #         markersize=4, linewidth=0.8,
#     #         markeredgewidth=0.6
#     #     )

#     # Friend-consistent styling
#     ax.plot(
#         x, avg["MBStr"],
#         marker="v", linestyle="-.",
#         label="MultiBend",
#         color=cmap(2),
#         linewidth=1,
#         zorder=2
#     )

#     ax.plot(
#         x, avg["OurStr"],
#         marker="^", linestyle=":",
#         label="PMultiBend",
#         color=cmap(4),
#         linewidth=1,
#         zorder=3
#     )

#     if has_online:
#         ax.plot(
#             x, avg["OnlineStr"],
#             marker=".", linestyle="-",
#             label="OPMultiBend",
#             color=cmap(0),
#             linewidth=1,
#             zorder=2.5
#         )



#     ymax = float(np.nanmax([
#         avg["OurStr"].max(),
#         avg["MBStr"].max(),
#         avg["OnlineStr"].max() if has_online else 0
#     ]))
#     ax.set_ylim(0, ymax * 1.05)

#     ax.set_xlabel("Error cutoff")
#     ax.set_ylabel("Stretch")

#     if use_log_x:
#         # (Usually keep linear here since ErrRate includes 0.0)
#         try:
#             ax.set_xscale("log", base=10)
#         except TypeError:
#             ax.set_xscale("log", basex=10)

#     ax.set_xticks(x)
#     ax.set_xticklabels([str(v) for v in x])  #this is also added to look consistent with tree plot
#     ax.grid(False)
#     ax.legend(loc="lower right")

#     if save:
#         out = f"{prefix}_k_{k}.png"
#         plt.savefig(out, dpi=180)
#         print("saved:", out)
#         plt.close(fig)
#     else:
#         plt.show()



def plot_stretch_vs_error_for_fixed_count_from_excel(
    filename, k,
    use_log_x=False,
    save=False,
    prefix="stretch_vs_error_k"
):
    """
    Plot stretch vs error cutoff for a fixed |P| = k.
    If OnlineStr exists in the Excel file, the plot includes OPMultiBend.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    meta = pd.read_excel(filename, sheet_name="meta")
    n = int(meta["n"].iloc[0])

    raw = pd.read_excel(filename, sheet_name="raw")
    if "Count" not in raw.columns:
        raw["Count"] = (raw["Frac"] * n).round().astype(int)

    df_k = raw[raw["Count"] == k].copy()
    if df_k.empty:
        print(f"No data rows with |P| = {k} in file {filename}")
        return

    avg = (df_k
           .groupby("ErrRate", as_index=False)
           .mean(numeric_only=True)
           .sort_values("ErrRate"))

    x = avg["ErrRate"].to_numpy(float)
    has_online = "OnlineStr" in avg.columns

    # ---- LaTeX-like sizing for 3-up subfigures (smaller ticks/labels) ----
    BASE = 9  # if still big, set to 8

    # Render near final printed size (so it won't be shrunk a lot in LaTeX)
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

    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(FIG_W_IN, FIG_H_IN), dpi=300)

        # Friend-consistent styling
        ax.plot(
            x, avg["MBStr"],
            marker="v", linestyle="-.",
            label="MultiBend",
            color=cmap(2),
            zorder=2
        )

        ax.plot(
            x, avg["OurStr"],
            marker="^", linestyle=":",
            label="PMultiBend",
            color=cmap(4),
            zorder=3
        )

        if has_online:
            ax.plot(
                x, avg["OnlineStr"],
                marker=".", linestyle="-",
                label="OPMultiBend",
                color=cmap(0),
                zorder=2.5
            )

        ymax = float(np.nanmax([
            avg["OurStr"].max(),
            avg["MBStr"].max(),
            avg["OnlineStr"].max() if has_online else 0
        ]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_xlabel("Error cutoff", labelpad=2)
        ax.set_ylabel("Stretch", labelpad=2)

        if use_log_x:
            # Usually keep linear since ErrRate includes 0.0
            try:
                ax.set_xscale("log", base=10)
            except TypeError:
                ax.set_xscale("log", basex=10)

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])

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
            out = f"{prefix}_k_{k}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("saved:", out)
        else:
            plt.show()



# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":

    # Example: halving + MultiBend comparison on a 64-node grid graph
    # res_cmp = simulate_halving_compare_multibend("64grid_diameter14test.edgelist")
    # res_cmp = simulate_halving_compare_multibend("144grid_diameter22test.edgelist")
    # res_cmp = simulate_halving_compare_multibend("256grid_diameter30test.edgelist")
    # res_cmp = simulate_halving_compare_multibend("576grid_diameter46test.edgelist")
    # Offline (2-way):
    # res_cmp = simulate_halving_compare_multibend("1024grid_diameter62test.edgelist")

    # Tree-styled ONLINE (3-way):
    res_cmp = simulate_halving_compare_multibend_online_tree(
        "144grid_diameter22test.edgelist",
        min_batch_size=1,
        max_batch_size=None,
        enforce_no_consecutive_vp=True
    )

    # Save raw + aggregated results
    xlsx_name = "fix144node_pred_frac_1by2_vary_errorrate_diamover2.xlsx"
    save_compare_results_to_excel(
        res_cmp, n=144,
        filename=xlsx_name,
        graph_file="144grid_diameter22test.edgelist"
    )

    # # Old view: stretch vs |P|, one figure per error cutoff
    # plot_mb_vs_ours_from_excel(
    #     xlsx_name,
    #     use_avg=True,
    #     use_log_x=True,
    #     error_metric="ErrMax"
    # )

    # NEW view: for a fixed |P|, plot stretch vs error cutoff (ErrRate).
    # Example: |P| = 32 (n/2 when n = 64)
    # Uncomment and adjust k as needed:
    plot_stretch_vs_error_for_fixed_count_from_excel(
        xlsx_name,
        k=36,
        use_log_x=False
    )
