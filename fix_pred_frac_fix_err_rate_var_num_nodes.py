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

def _nx_weight(G):
    """Return the edge-weight attribute name if present, else None (unweighted)."""
    # If edges have a 'weight' attribute, keep using it; otherwise treat as unweighted.
    try:
        for _, _, data in G.edges(data=True):
            if isinstance(data, dict) and "weight" in data:
                return "weight"
            break
    except Exception:
        pass
    return None


def _graph_diameter_safe(G):
    """Compute diameter on (largest) connected component, respecting weights when available."""
    # Many of these experiments use connected grid graphs; handle disconnected graphs defensively.
    if hasattr(nx, "is_connected"):
        try:
            if not nx.is_connected(G):
                comp = max(nx.connected_components(G), key=len)
                G = G.subgraph(comp).copy()
        except Exception:
            pass

    w = _nx_weight(G)
    try:
        return nx.diameter(G, weight=w if w is not None else None)
    except TypeError:
        # Older networkx may not like weight=None
        return nx.diameter(G)


def _pairwise_feasible_set(G, target_unique, max_pair_dist, rng):
    """
    Try to pick up to `target_unique` DISTINCT nodes such that every pair is within max_pair_dist.
    Returns a list (may be shorter than target_unique if not feasible).
    """
    nodes = list(G.nodes())
    if not nodes:
        return []

    if max_pair_dist is None:
        rng.shuffle(nodes)
        return nodes[:target_unique]

    w = _nx_weight(G)
    # Greedy randomized packing
    best = []
    for _ in range(200):
        rng.shuffle(nodes)
        chosen = []
        for v in nodes:
            if not chosen:
                chosen.append(v)
                if len(chosen) == target_unique:
                    return chosen
                continue
            try:
                dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=max_pair_dist, weight=w)
            except TypeError:
                # fallback unweighted
                dist_map = nx.single_source_shortest_path_length(G, v, cutoff=max_pair_dist)

            if all(u in dist_map for u in chosen):
                chosen.append(v)
                if len(chosen) == target_unique:
                    return chosen

        if len(chosen) > len(best):
            best = chosen
            if len(best) == target_unique:
                break

    return best


def _choose_vp_support(G, desired_unique, vp_pairwise_divisor, rng):
    """Choose a support set of distinct nodes to sample Vp from."""
    nodes = list(G.nodes())
    if not nodes or desired_unique <= 0:
        return []

    if vp_pairwise_divisor is None or vp_pairwise_divisor <= 1:
        rng.shuffle(nodes)
        return nodes[:desired_unique]

    diam = float(_graph_diameter_safe(G))
    max_pair_dist = diam / float(vp_pairwise_divisor)

    # Try to get as many distinct nodes as requested; fall back to fewer distinct if needed.
    target = min(desired_unique, len(nodes))
    while target >= 1:
        support = _pairwise_feasible_set(G, target, max_pair_dist, rng)
        if len(support) >= target:
            return support
        target -= 1

    # Always feasible with one node
    return [rng.choice(nodes)]


def _sample_from_multiset(pool, k, rng):
    """Sample k items from list `pool` without replacement, removing them from pool."""
    if k <= 0:
        return []
    k = min(k, len(pool))
    idxs = rng.sample(range(len(pool)), k)
    idxs.sort(reverse=True)
    out = [pool[i] for i in idxs]
    for i in idxs:
        pool.pop(i)
    rng.shuffle(out)
    return out


def _shuffle_avoid_consecutive_dups(seq, rng, max_tries=50):
    """Shuffle seq in-place-ish to avoid consecutive duplicates if possible."""
    if len(seq) <= 1:
        return list(seq)

    best = list(seq)
    def has_consec(xs):
        return any(xs[i] == xs[i-1] for i in range(1, len(xs)))

    for _ in range(max_tries):
        xs = list(seq)
        rng.shuffle(xs)
        if not has_consec(xs):
            return xs
        best = xs

    # If impossible (e.g., all same), return best effort
    return best


def choose_Vp(G, fraction, *, vp_pairwise_divisor=2.0, rng=None):
    """
    Fraction-based prediction set (used by non-halving simulate()).

    Supervisor constraint (optional):
      - If vp_pairwise_divisor is not None and > 1, enforce that for any two DISTINCT nodes
        in the support set we sample Vp from: dist_G(u,v) <= diameter(G) / vp_pairwise_divisor.
      - If vp_pairwise_divisor is None (or <= 1), no restriction (up to full diameter).

    Notes:
      - Vp is sampled WITH replacement from the feasible support set (so duplicates are allowed).
    """
    if rng is None:
        rng = random

    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    total_nodes = len(nodes)

    vp_size = int(total_nodes * fraction)
    if vp_size <= 0:
        return []
    vp_size = min(vp_size, total_nodes)

    # Choose a feasible distinct support set, then sample with replacement to fill slots.
    support = _choose_vp_support(G, desired_unique=min(vp_size, total_nodes), vp_pairwise_divisor=vp_pairwise_divisor, rng=rng)
    if not support:
        support = nodes[:1] if nodes else []

    original_Vp = list(rng.choices(support, k=vp_size))
    rng.shuffle(original_Vp)

    # (legacy leftover structure — kept to preserve original flow)
    reduced_Vp = list(set(original_Vp))
    rng.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = rng.choice(list(remaining)) if remaining else rng.choice(nodes)

    insert_position = rng.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    _ = set(reduced_Vp)  # not used

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

def simulate(graph_file, use_move_stretch=False):
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


def choose_Vp_halving(G, k: int, *, vp_pairwise_divisor=2.0, rng=None):
    """
    Halving-mode version of choose_Vp. Uses k prediction slots, sampled WITH replacement.

    Supervisor constraint (optional):
      - If vp_pairwise_divisor is not None and > 1, enforce pairwise distance bounded support:
        dist_G(u,v) <= diameter(G) / vp_pairwise_divisor for any two DISTINCT nodes in the support.
      - If vp_pairwise_divisor is None (or <= 1), no restriction.
    """
    if rng is None:
        rng = random

    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    total_nodes = len(nodes)

    vp_size = int(k)
    vp_size = max(1, min(vp_size, total_nodes))

    support = _choose_vp_support(G, desired_unique=min(vp_size, total_nodes), vp_pairwise_divisor=vp_pairwise_divisor, rng=rng)
    if not support:
        support = nodes[:1] if nodes else []

    original_Vp = list(rng.choices(support, k=vp_size))
    rng.shuffle(original_Vp)

    # (legacy leftover structure — kept to preserve original flow)
    reduced_Vp = list(set(original_Vp))
    rng.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = rng.choice(list(remaining)) if remaining else rng.choice(nodes)

    insert_position = rng.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    _ = set(reduced_Vp)  # not used

    return original_Vp

def simulate_halving(graph_file, use_move_stretch=False):
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
                P = choose_Vp_halving(G, k)
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


# --------- MultiBend comparison (all |P| in halving mode) ----------------------------

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




# ---------------------------- ONLINE PMULTIBEND (TREE-STYLED) ----------------------------
# In this variant, predictions are not all known at time 0. Instead, they arrive in batches.
# At the start of each batch, we add the batch's predicted nodes to `known_pred`.
# Cluster leaders are updated lazily: we only switch a cluster's leader to a predicted node
# when a traversal actually visits that cluster.

def serve_requests_remove_by_id(vp_and_q, rng=None, *,
                               min_batch_size=1,
                               max_batch_size=None,
                               max_extract_fraction=0.50,   # kept for backward compat (unused)
                               max_release_fraction=0.50,   # kept for backward compat (unused)
                               enforce_no_consecutive_vp=True,
                               allow_time0_empty=True,
                               time0_empty_prob=0.50,
                               debug=False):
    """
    Supervisor-compliant tree-styled batching of online predictions.

    Input:
        vp_and_q = list of triples (id, vp, q).
          - We interpret the vp values as the *offline prediction pool* Vp (slots/multiset),
            and q values as the *request stream* Q in order.
          - The initial vp in vp_and_q are treated as the available prediction slots.

    Output:
        timeline = list of segments, each:
            {"release": [vp1, vp2, ...], "serve": [q1, q2, ...]}

    Required constraints implemented:
      1) Time 0 may or may not have predictions.
         - If time0 is empty, we SERVE a non-empty prefix of requests before the first non-empty release
           (Option A).
      2) Batch size range: any integer from 1 up to floor(n/2),
         where n is the TOTAL offline Vp slot count (len(vp_and_q)).
      3) No empty batches after time 0.
      4) Outstanding constraint + "return unused predictions":
         - If another batch will be released later, we only serve a PROPER PREFIX of the current batch's
           assigned requests so >=1 predictions remain outstanding at the moment of the next release.
         - The unused prediction suffix is returned to the pool BEFORE sampling the next batch, and the
           unserved requests remain to be re-predicted.

    Notes:
      - enforce_no_consecutive_vp applies to the release order only.
      - max_extract_fraction / max_release_fraction are retained for API compatibility but not used
        under this supervisor spec.
    """
    if rng is None:
        rng = random.Random()

    pred_pool = [vp for _, vp, _ in vp_and_q]  # multiset of available prediction slots
    req_queue = [q for _, _, q in vp_and_q]    # requests in order

    n_total = len(pred_pool)
    max_global = (n_total // 2) if n_total > 0 else 0
    max_global = max(1, max_global) if n_total > 0 else 0

    timeline = []

    # Optional time-0 empty release (Option A): serve some prefix before first non-empty release.
    if allow_time0_empty and req_queue and (rng.random() < float(time0_empty_prob)):
        serve_cap = len(req_queue)
        # serve at least one request; serving all ends the process (no releases at all)
        serve_count = rng.randint(1, serve_cap)
        timeline.append({"release": [], "serve": req_queue[:serve_count]})
        req_queue = req_queue[serve_count:]
        # predictions are NOT consumed in the time-0 empty segment

        if debug:
            print(f"[time0-empty] served {serve_count}, remaining req={len(req_queue)}, pool={len(pred_pool)}")

    # Main loop: repeat release -> serve(prefix) until all requests served.
    while req_queue:
        if not pred_pool:
            # Defensive fallback: if pool is empty, we cannot release predictions; serve remaining.
            timeline.append({"release": [], "serve": list(req_queue)})
            break

        # Determine per-release batch size bounds
        max_b = min(max_global, len(pred_pool), len(req_queue))
        if max_batch_size is not None:
            max_b = min(max_b, int(max_batch_size))

        min_b = max(1, int(min_batch_size))
        min_b = min(min_b, max_b)

        if max_b < 1:
            timeline.append({"release": [], "serve": list(req_queue)})
            break

        B = rng.randint(min_b, max_b)

        # Sample predictions without replacement from pool
        batch_preds = _sample_from_multiset(pred_pool, B, rng)

        # Enforce no consecutive duplicates in the release order (best effort)
        if enforce_no_consecutive_vp and B > 1:
            batch_preds = _shuffle_avoid_consecutive_dups(batch_preds, rng)

        # Assign predictions to the next B requests
        window_reqs = req_queue[:B]

        # Choose how many of these B requests to actually serve before next release
        if len(req_queue) <= B:
            # last segment: no next release, safe to serve all
            serve_count = B
        else:
            # must leave at least 1 outstanding prediction/request from this batch
            serve_count = rng.randint(1, B - 1) if B > 1 else 1

        timeline.append({"release": list(batch_preds), "serve": list(window_reqs[:serve_count])})

        # Consume served requests from the queue
        req_queue = req_queue[serve_count:]

        # Return unused prediction suffix to the pool (before next sampling)
        unused_preds = batch_preds[serve_count:]
        pred_pool.extend(unused_preds)

        if debug:
            print(f"[release] B={B}, served={serve_count}, returned={len(unused_preds)}, remaining req={len(req_queue)}, pool={len(pred_pool)}")

    return timeline

def get_spiral_online(node, leader_map, known_pred, rng=None, verbose=False):
    """\
    Same as get_spiral(), but with *lazy leader upgrades*.

    When we visit the cluster containing `node` at some (lvl,t), if that cluster contains
    any known predicted node, we switch its leader to one of those predicted nodes.

    This models "online leader election" where predictions are revealed over time and
    we only update leaders along paths that are actually traversed.
    """
    if rng is None:
        rng = random.Random()

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
                    # lazy upgrade if this cluster has any known predictions
                    if known_pred:
                        cand = [v for v in cl if v in known_pred]
                        if cand and leader not in known_pred:
                            new_leader = rng.choice(cand)
                            leader_map[key][idx] = (new_leader, cl)
                            leader = new_leader

                    if leader not in seen:
                        path.append(leader)
                        seen.add(leader)
                    break

    # force-append the root leader
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    if verbose:
        print(f"[online spiral] node={node} path={path}")

    return path


def publish_online(owner, leader_map, known_pred, rng=None):
    """Publish for online PMultiBend (uses get_spiral_online)."""
    global down_link
    down_link.clear()
    sp = get_spiral_online(owner, leader_map, known_pred, rng=rng)
    for i in range(len(sp) - 1, 0, -1):
        down_link[sp[i]] = sp[i-1]


def _single_request_costs_online(r, owner, leader_map, known_pred, G, weight=None, rng=None, trace=False):
    """Return (up_hops, down_hops, obj) under online leader upgrades."""
    publish_online(owner, leader_map, known_pred, rng=rng)

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


def measure_stretch_move_online(requesters, owner, leader_map, known_pred, G, weight=None, rng=None, trace=True):
    """MOVE stretch under tree-styled online predictions + lazy leader upgrades."""
    total_alg = 0
    total_opt = 0

    for r in requesters:
        if r == owner:
            continue

        up, down, obj = _single_request_costs_online(
            r, owner, leader_map, known_pred, G, weight=weight, rng=rng, trace=(trace and DEBUG)
        )
        total_alg += (up + down + obj)
        total_opt += obj

        owner = r
        publish_online(owner, leader_map, known_pred, rng=rng)

    return (total_alg / total_opt) if total_opt > 0 else 1.0

def simulate_halving_compare_multibend(graph_file):
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

                P = choose_Vp_halving(G, k)
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


# --------- Excel helpers for halving+MB ----------------------------

def save_compare_results_to_excel(results, n, filename, graph_file=None):
    """
    Save simulate_halving_compare_multibend() results to an .xlsx file.
    Sheets:
      - 'raw': one row per request
      - 'avg': aggregated by (Count, ErrRate)
      - 'meta': metadata (n, graph, trials, timestamp, etc.)
    """
    cols = ["Frac","ErrRate","ErrMax","ErrMin","ErrAvg","OurStr","MBStr"]
    df   = pd.DataFrame(results, columns=cols)
    df["Count"] = (df["Frac"] * n).round().astype(int)

    group_cols = ["Count", "ErrRate"]
    avg = df.groupby(group_cols, as_index=False).agg({
        "ErrMax": "max",
        "ErrMin": "min",
        "ErrAvg": "mean",
        "OurStr": "mean",
        "MBStr": "mean"
    })

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
    """
    For each error cutoff, plot Our stretch vs MB stretch as a function of |P|.
    Uses data from the Excel file.
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

    for e in err_levels:
        sub = df[df.ErrRate == e].copy()
        if sub.empty:
            continue
        avg = sub.sort_values("Count")
        x = avg["Count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

        ax.plot(x*0.985, avg["OurStr"], "-o",
                label=f"PMultiBend (err ≤ {e:.1f})", zorder=3)
        ax.plot(x*1.015, avg["MBStr"], "--s",
                label="MultiBend", alpha=0.85, zorder=2)

        ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
        ax.set_ylim(0, ymax * 1.05)

        ax.set_ylabel("Stretch")
        ax.set_xlabel(f"Number of predicted nodes among {n} nodes")
        if use_log_x:
            try:
                ax.set_xscale("log", base=2)
            except TypeError:
                ax.set_xscale("log", basex=2)
        ax.set_xticks(x)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="best")
        ax.set_title(f"Stretch vs |P| (err ≤ {e:.1f})")

        if save:
            out = f"{prefix}_stretch_only_err_{str(e).replace('.','p')}.png"
            plt.savefig(out, dpi=180)
            print("saved:", out)
            plt.close(fig)
        else:
            plt.show()


# --------- Stretch vs error cutoff for one fixed |P| (from Excel) ---------

def plot_stretch_vs_error_for_fixed_count_from_excel(filename, k,
                                                     use_log_x=False,
                                                     save=False,
                                                     prefix="stretch_vs_error_k"):
    """
    For a fixed |P| = k and single graph (Excel file from halving+MB pipeline):
    X-axis: ErrRate (0.0, 0.1, ..., 0.5)
    Y-axis: Our stretch and MB stretch.
    """
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

    fig, ax = plt.subplots(1, 1, figsize=(7,5), constrained_layout=True)

    ax.plot(x, avg["OurStr"], "-o", label="PMultiBend", zorder=3)
    ax.plot(x, avg["MBStr"], "--s", label="MultiBend", alpha=0.85, zorder=2)

    ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
    ax.set_ylim(0, ymax * 1.05)

    ax.set_xlabel("Error cutoff (ErrRate)")
    ax.set_ylabel("Stretch")
    title = f"Stretch vs error cutoff for |P| = {k} (n = {n})"
    ax.set_title(title)

    if use_log_x:
        # Usually keep this linear; ErrRate includes 0.0.
        try:
            ax.set_xscale("log", base=10)
        except TypeError:
            ax.set_xscale("log", basex=10)

    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best")

    if save:
        out = f"{prefix}_k_{k}.png"
        plt.savefig(out, dpi=180)
        print("saved:", out)
        plt.close(fig)
    else:
        plt.show()


# ==================== NEW: fixed-|P| simulator & cross-n plots ====================

def simulate_halving_compare_multibend_fixedP(graph_file, k):
    """
    Like simulate_halving_compare_multibend, but for a single fixed |P| = k.
    Returns rows: (ErrRate, OurStr, MBStr) for that graph.
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    H_base = build_mesh_hierarchy(size)
    results = []

    for error in ERROR_VALUES:
        err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

        for _ in range(NUM_TRIALS):
            owner_start = random.choice(list(G.nodes()))

            P = choose_Vp_halving(G, k)
            Q = sample_Q_within_diameter(G, P, error)

            # leaders for MB baseline and prediction-aware scheme
            leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)
            leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

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

                mb_str = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                owner_mb = req
                publish(owner_mb, leaders_base)

                results.append((err_rate, our_str, mb_str))

    return results

def simulate_halving_compare_multibend_fixedP_tree_online(
    graph_file,
    k,
    *,
    min_batch_size=1,
    max_batch_size=None,
    max_extract_fraction=0.50,
    max_release_fraction=0.50,
    enforce_no_consecutive_vp=True,
    seed=None,
    debug_batches=False,
):
    """
    Tree-styled Online PMultiBend for a single fixed |P| = k.

    What "tree-styled" means here:
      - We start with *no* predictions known.
      - Predictions arrive in batches over time.
      - When a batch arrives, we reveal a subset of predicted nodes from P.
      - Only after that batch is revealed do we serve the corresponding actual requests.
      - Leader upgrades (switching a cluster leader to a known predicted node) are done lazily,
        only along spiral paths that are actually traversed.

    Returns rows: (ErrRate, OurStr, MBStr, OnlineStr)
      - OurStr: offline PMultiBend (all predictions known at time 0)
      - MBStr: MultiBend baseline (uniform random leaders)
      - OnlineStr: online PMultiBend (predictions revealed over time)
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))

    H_base = build_mesh_hierarchy(size)
    results = []

    # base seed for reproducibility (optional)
    base_seed = seed if seed is not None else random.randrange(1 << 30)

    for error in ERROR_VALUES:
        err_rate = 0.0 if error > 15 else round(1.0 / error, 1)

        for t in range(NUM_TRIALS):
            owner_start = random.choice(list(G.nodes()))

            # sample predictions + actual requests
            P = choose_Vp_halving(G, k)
            Q = sample_Q_within_diameter(G, P, error)

            # leaders for MB baseline and offline PMultiBend
            leaders_base = assign_cluster_leaders(H_base, seed=None, prefer=None)
            leaders_pred = assign_cluster_leaders(H_base, seed=None, prefer=P)

            # leaders for online PMultiBend start as baseline (no predictions known yet)
            leaders_online = assign_cluster_leaders(H_base, seed=None, prefer=None)
            known_pred = set()

            # build (id, vp, q) pairs and create tree-styled batches
            pairs = [(i, P[i], Q[i]) for i in range(min(len(P), len(Q)))]
            trial_seed = (base_seed + 1000003 * t + 97 * int(err_rate * 10) + 17 * k) & 0xFFFFFFFF
            rng = random.Random(trial_seed)
            timeline = serve_requests_remove_by_id(
                pairs,
                rng=rng,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                max_extract_fraction=max_extract_fraction,
                max_release_fraction=max_release_fraction,
                enforce_no_consecutive_vp=enforce_no_consecutive_vp,
                debug=debug_batches,
            )

            owner_ours = owner_start
            owner_mb = owner_start
            owner_on = owner_start
            publish(owner_ours, leaders_pred)
            publish(owner_mb, leaders_base)
            publish_online(owner_on, leaders_online, known_pred, rng=rng)

            # serve in batch order
            for seg in timeline:
                # reveal this segment's predictions first
                for vp in seg.get('release', []):
                    known_pred.add(vp)

                # now serve the corresponding actual requests
                for req in seg.get('serve', []):
                    if req == owner_ours:
                        owner_ours = req
                        owner_mb = req
                        owner_on = req
                        publish(owner_ours, leaders_pred)
                        publish(owner_mb, leaders_base)
                        publish_online(owner_on, leaders_online, known_pred, rng=rng)
                        continue

                    our_str = measure_stretch_move([req], owner_ours, leaders_pred, G, trace=False)
                    owner_ours = req
                    publish(owner_ours, leaders_pred)

                    mb_str = multibend_move_sequence_stretch([req], owner_mb, leaders_base, G)
                    owner_mb = req
                    publish(owner_mb, leaders_base)

                    on_str = measure_stretch_move_online([req], owner_on, leaders_online, known_pred, G, rng=rng, trace=False)
                    owner_on = req
                    publish_online(owner_on, leaders_online, known_pred, rng=rng)

                    results.append((err_rate, our_str, mb_str, on_str))

    return results



def plot_stretch_vs_error_from_results(results, n, k):
    """
    Convenience: for one graph & fixed |P| = k, plot stretch vs error cutoff
    without going through Excel.
    """
    df = pd.DataFrame(results, columns=["ErrRate", "OurStr", "MBStr"])
    avg = df.groupby("ErrRate", as_index=False).mean(numeric_only=True).sort_values("ErrRate")

    x = avg["ErrRate"].to_numpy(float)

    fig, ax = plt.subplots(1, 1, figsize=(7,5), constrained_layout=True)

    ax.plot(x, avg["OurStr"], "-o", label="PMultiBend", zorder=3)
    ax.plot(x, avg["MBStr"], "--s", label="MultiBend", alpha=0.85, zorder=2)

    ymax = float(np.nanmax([avg["OurStr"].max(), avg["MBStr"].max()]))
    ax.set_ylim(0, ymax * 1.05)

    ax.set_xlabel("Error cutoff (ErrRate)")
    ax.set_ylabel("Stretch")
    ax.set_title(f"Stretch vs error cutoff for |P| = {k} (n = {n})")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best")
    plt.show()


def run_across_graphs_fixed_fraction(graph_files, frac):
    """
    For a fixed prediction fraction 'frac' (e.g. 0.5 for n/2),
    run PMultiBend vs MultiBend on each graph in graph_files.

    graph_files: list of (filename, n) pairs.

    Returns a DataFrame with columns:
        n, ErrRate, OurStr, MBStr, OnlineStr
    averaged over NUM_TRIALS per graph.
    """
    rows = []

    for graph_file, n in graph_files:
        k = int(round(frac * n))
        print(f"Running {graph_file} with n={n}, |P|={k} (frac={frac})")
        results = simulate_halving_compare_multibend_fixedP_tree_online(graph_file, k)

        for (err_rate, our_str, mb_str, online_str) in results:
            rows.append((n, err_rate, our_str, mb_str, online_str))

    df = pd.DataFrame(rows, columns=["n", "ErrRate", "OurStr", "MBStr", "OnlineStr"])
    avg = df.groupby(["n", "ErrRate"], as_index=False).mean(numeric_only=True)
    return avg


# def plot_stretch_vs_n_for_fixed_fraction(avg_df, frac, use_log_x=False):
#     """
#     New plot your supervisor asked for:

#       - fixed prediction fraction 'frac' (|P| = frac * n)
#       - x-axis: number of nodes n (different graphs)
#       - one figure per ErrRate in ERROR_VALUES_2
#       - curves: PMultiBend (offline) vs PMultiBend (online) vs MultiBend
#     """
#     for e in ERROR_VALUES_2:
#         sub = avg_df[avg_df["ErrRate"] == e].sort_values("n")
#         if sub.empty:
#             continue

#         x = sub["n"].to_numpy(dtype=float)

#         cmap = plt.get_cmap('tab20')
#         plt.figure(figsize=(7,5))
#         # plt.plot(x, sub["OurStr"], "-o", label="PMultiBend (offline)")
#         # plt.plot(x, sub["OnlineStr"], "-.^", label="PMultiBend (online)")
#         # plt.plot(x, sub["MBStr"], "--s", label="MultiBend")

#         # MultiBend: orange dashdot + inverted triangle
#         # plt.plot(
#         #     x, sub["MBStr"],
#         #     linestyle="-.", marker="v",
#         #     color="tab:orange",
#         #     label="MultiBend",
#         #     alpha=0.85,
#         #     markersize=8, linewidth=0.8
#         # )

#         # # PMultiBend (offline): green dotted + triangle
#         # plt.plot(
#         #     x, sub["OurStr"],
#         #     linestyle=":", marker="^",
#         #     color="tab:green",
#         #     label="PMultiBend",
#         #     alpha=0.9,
#         #     markersize=8, linewidth=0.8
#         # )

#         # # PMultiBend (online): blue solid + small dot
#         # plt.plot(
#         #     x, sub["OnlineStr"],
#         #     linestyle="-", marker="o",
#         #     color="tab:blue",
#         #     label="OPMultiBend",
#         #     alpha=0.9,
#         #     markersize=4, linewidth=0.8,
#         #     markeredgewidth=0.6
#         # )

#         plt.plot(
#             x, sub["MBStr"],
#             marker="v", linestyle="-.",
#             label="MultiBend",
#             color=cmap(2),
#             linewidth=1
#         )

#         plt.plot(
#             x, sub["OurStr"],
#             marker="^", linestyle=":",
#             label="PMultiBend",
#             color=cmap(4),
#             linewidth=1
#         )

#         plt.plot(
#             x, sub["OnlineStr"],
#             marker=".", linestyle="-",
#             label="OPMultiBend",
#             color=cmap(0),
#             linewidth=1
#         )

#         if use_log_x:
#             try:
#                 plt.xscale("log", base=2)
#             except TypeError:
#                 plt.xscale("log", basex=2)

#         plt.xticks(x, [str(int(v)) for v in x])
#         plt.grid(False)
#         plt.xticks(x, [str(int(v)) for v in x])
#         plt.ylabel("Stretch")
#         plt.xlabel("Number of nodes n")
#         # plt.title(f"Stretch vs n for |P| = {frac}·n  (ErrRate ≤ {e:.1f})")
#         plt.legend(loc="lower right")
#         plt.tight_layout()
#         plt.show()


# def plot_stretch_vs_n_for_fixed_fraction(avg_df, frac, use_log_x=False):
#     """
#     Plot your supervisor asked for:

#       - fixed prediction fraction 'frac' (|P| = frac * n)
#       - x-axis: number of nodes n (different graphs)
#       - one figure per ErrRate in ERROR_VALUES_2
#       - curves: PMultiBend (offline) vs PMultiBend (online) vs MultiBend

#     UPDATED:
#       - x-axis is now equally spaced (categorical positions) and labeled by n.
#       - use_log_x is ignored in this mode (log scale conflicts with equal spacing).
#     """
#     for e in ERROR_VALUES_2:
#         sub = avg_df[avg_df["ErrRate"] == e].sort_values("n")
#         if sub.empty:
#             continue

#         # ---- Equal-spaced x axis ----
#         x_labels = sub["n"].to_numpy(dtype=int)   # actual n values shown on ticks
#         x_pos = np.arange(len(x_labels))          # equally spaced positions

#         cmap = plt.get_cmap("tab20")
#         plt.figure(figsize=(7, 5))

#         # Friend-consistent styling
#         plt.plot(
#             x_pos, sub["MBStr"],
#             marker="v", linestyle="-.",
#             label="MultiBend",
#             color=cmap(2),
#             linewidth=1
#         )
#         plt.plot(
#             x_pos, sub["OurStr"],
#             marker="^", linestyle=":",
#             label="PMultiBend",
#             color=cmap(4),
#             linewidth=1
#         )
#         plt.plot(
#             x_pos, sub["OnlineStr"],
#             marker=".", linestyle="-",
#             label="OPMultiBend",
#             color=cmap(0),
#             linewidth=1
#         )

#         # Ticks show the real n values, but spacing is uniform
#         plt.xticks(x_pos, [str(v) for v in x_labels])

#         plt.grid(False)
#         plt.ylabel("Stretch")
#         plt.xlabel("Number of nodes n")
#         plt.legend(loc="lower right")
#         plt.tight_layout()
#         plt.show()

def plot_stretch_vs_n_for_fixed_fraction(avg_df, frac, use_log_x=False, save=False, prefix="stretch_vs_n_frac"):
    """
    Plot (LaTeX-like sizing for 3-up subfigures):

      - fixed prediction fraction 'frac' (|P| = frac * n)
      - x-axis: number of nodes n (different graphs), equally spaced positions
      - one figure per ErrRate in ERROR_VALUES_2
      - curves: MultiBend vs PMultiBend vs OPMultiBend

    Note:
      - use_log_x is ignored (equal spacing conflicts with log scale).
    """
    import numpy as np
    import matplotlib.pyplot as plt

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

    for e in ERROR_VALUES_2:
        sub = avg_df[avg_df["ErrRate"] == e].sort_values("n")
        if sub.empty:
            continue

        # ---- Equal-spaced x axis ----
        x_labels = sub["n"].to_numpy(dtype=int)   # real n values shown on ticks
        x_pos = np.arange(len(x_labels))          # equally spaced positions

        has_online = "OnlineStr" in sub.columns

        with plt.rc_context(rc):
            fig, ax = plt.subplots(1, 1, figsize=(FIG_W_IN, FIG_H_IN), dpi=300)

            # Friend-consistent styling
            ax.plot(
                x_pos, sub["MBStr"],
                marker="v", linestyle="-.",
                label="MultiBend",
                color=cmap(2),
                zorder=2
            )
            ax.plot(
                x_pos, sub["OurStr"],
                marker="^", linestyle=":",
                label="PMultiBend",
                color=cmap(4),
                zorder=3
            )
            if has_online:
                ax.plot(
                    x_pos, sub["OnlineStr"],
                    marker=".", linestyle="-",
                    label="OPMultiBend",
                    color=cmap(0),
                    zorder=1
                )

            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in x_labels])

            ax.grid(False)
            ax.set_ylabel("Stretch", labelpad=2)
            ax.set_xlabel("Number of nodes $n$", labelpad=2)

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
                out = f"{prefix}_frac_{str(frac).replace('.','p')}_err_{str(e).replace('.','p')}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print("saved:", out)
            else:
                plt.show()


# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":

    # ================== B) NEW: cross-n experiment (x-axis = number of nodes) ===============

    # Replace the filenames below with the actual files you have in graphs/grid/
    GRAPH_FILES = [
        ("64grid_diameter14test.edgelist", 64),
        ("144grid_diameter22test.edgelist", 144),
        ("256grid_diameter30test.edgelist", 256),
        ("576grid_diameter46test.edgelist", 576),
        ("1024grid_diameter62test.edgelist", 1024),
    ]

    # Fixed prediction fraction: e.g. 0.5 for |P| = n/2, 0.25 for n/4, etc.
    frac = 0.5

    avg_across = run_across_graphs_fixed_fraction(GRAPH_FILES, frac)
    plot_stretch_vs_n_for_fixed_fraction(avg_across, frac, use_log_x=False)