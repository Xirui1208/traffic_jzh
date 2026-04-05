import os
import heapq
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

import matplotlib.font_manager as _fm
for _fn in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
    try:
        _fm.findfont(_fn, fallback_to_default=False)
        rcParams['font.sans-serif'] = [_fn]
        rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        pass

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

DATA_DIR = "data"
CITY_FILES = {
    "Shenyang":  "Shenyang_Edgelist.csv",
    "Dalian":    "Dalian_Edgelist.csv",
    "Zhengzhou": "Zhengzhou_Edgelist.csv",
    "Qingdao":   "Qingdao_Edgelist.csv",
    "Harbin":    "Harbin_Edgelist.csv",
    "Dongguan":  "Dongguan_Edgelist.csv",
    "Chengdu":   "Chengdu_Edgelist.csv",
    "Quanzhou":  "Quanzhou_Edgelist.csv",
}

def load_graph(city_name):
    path = os.path.join(DATA_DIR, CITY_FILES[city_name])
    df = pd.read_csv(path)
    G = nx.Graph()
    node_coords = {}
    for _, row in df.iterrows():
        u, v = int(row['START_NODE']), int(row['END_NODE'])
        G.add_edge(u, v, weight=float(row['LENGTH']))
        node_coords[u] = (float(row['XCoord']), float(row['YCoord']))
    for n, (x, y) in node_coords.items():
        if n in G:
            G.nodes[n]['x'] = x
            G.nodes[n]['y'] = y
    return G

def problem1_basic_stats():
    print("\n" + "="*70)
    print("问题1：网络基本特征统计 & 度分布建模")
    print("="*70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    summary = []

    for idx, city in enumerate(CITY_FILES):
        G = load_graph(city)
        N = G.number_of_nodes()
        M = G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        avg_deg = np.mean(degrees)
        max_deg = max(degrees)
        avg_clust = nx.average_clustering(G)

        Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        random.seed(42)
        n_sample = min(80, Gcc.number_of_nodes())
        sample_nodes = random.sample(list(Gcc.nodes()), n_sample)
        paths = []
        for src in sample_nodes:
            lens = nx.single_source_shortest_path_length(Gcc, src)
            paths.extend(lens.values())
        avg_path = round(float(np.mean(paths)), 2) if paths else -1
        diameter_est = int(np.percentile(paths, 99)) if paths else -1

        summary.append({
            "城市": city, "节点数N": N, "边数M": M,
            "平均度<k>": round(avg_deg, 3),
            "最大度k_max": max_deg,
            "平均聚类系数C": round(avg_clust, 4),
            "直径(99%估计)": diameter_est,
            "平均路径长(估计)": avg_path
        })

        deg_vals = np.array(degrees)
        k_min_v, k_max_v = int(deg_vals.min()), int(deg_vals.max())
        bins = np.arange(k_min_v, k_max_v + 2) - 0.5
        counts, edges = np.histogram(deg_vals, bins=bins, density=True)
        k_centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        kpos = k_centers[mask]
        cpos = counts[mask]

        k_fit = np.linspace(max(float(kpos.min()), 0.5), float(kpos.max()), 300)

        ax = axes[idx]
        ax.bar(kpos, cpos, width=0.8, alpha=0.55, color='steelblue', label='实测')

        try:
            popt_pl, _ = curve_fit(
                lambda k, C, g: C * k ** (-g),
                kpos, cpos, p0=[1.0, 2.0], maxfev=5000,
                bounds=([0, 0.1], [np.inf, 10]))
            ax.plot(k_fit, popt_pl[0] * k_fit ** (-popt_pl[1]),
                    'r-', lw=2, label=f'幂律 γ={popt_pl[1]:.2f}')
        except Exception:
            pass

        try:
            popt_exp, _ = curve_fit(
                lambda k, C, lam: C * np.exp(-lam * k),
                kpos, cpos, p0=[1.0, 0.5], maxfev=5000,
                bounds=([0, 0.001], [np.inf, 20]))
            ax.plot(k_fit, popt_exp[0] * np.exp(-popt_exp[1] * k_fit),
                    'g--', lw=2, label=f'指数 λ={popt_exp[1]:.2f}')
        except Exception:
            pass

        ax.set_title(f'{city}\nN={N}, <k>={avg_deg:.2f}', fontsize=9)
        ax.set_xlabel('度 k')
        ax.set_ylabel('P(k)')
        ax.legend(fontsize=7)
        ax.set_yscale('log')
        ax.set_xscale('log')

    plt.suptitle('8城市度分布及拟合（双对数坐标，道路网以指数拟合为主）', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/Q1_degree_distributions.png', dpi=150)
    plt.close()

    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    df_sum.to_csv('outputs/Q1_basic_stats.csv', index=False, encoding='utf-8-sig')
    print("\n[已保存] Q1_basic_stats.csv  &  Q1_degree_distributions.png")
    return df_sum

def compute_robustness_curve(G, removal_order, step_frac=0.01):
    N0 = G.number_of_nodes()
    step = max(1, int(N0 * step_frac))

    Gwork = G.copy()
    fracs = [0.0]
    lcc_ratios = [1.0]

    removed_so_far = 0
    batch = []
    for node in removal_order:
        if node not in Gwork:
            continue
        batch.append(node)
        if len(batch) >= step:
            Gwork.remove_nodes_from(batch)
            removed_so_far += len(batch)
            batch = []
            frac = removed_so_far / N0
            if Gwork.number_of_nodes() == 0:
                lcc = 0
            else:
                lcc = max(len(c) for c in nx.connected_components(Gwork))
            fracs.append(frac)
            lcc_ratios.append(lcc / N0)
            if lcc / N0 < 0.005:
                break

    R = _trapz(lcc_ratios, fracs)
    return np.array(fracs), np.array(lcc_ratios), R

def problem2_random_failure(n_trials=10):
    print("\n" + "="*70)
    print("问题2：随机节点故障健壮性")
    print("="*70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    results = []

    for idx, city in enumerate(CITY_FILES):
        G = load_graph(city)
        N0 = G.number_of_nodes()
        nodes = list(G.nodes())

        all_fracs_list, all_lcc_list, R_list = [], [], []

        for trial in range(n_trials):
            np.random.seed(trial)
            order = np.random.permutation(nodes).tolist()
            fracs, lcc_ratios, R = compute_robustness_curve(G, order)
            all_fracs_list.append(fracs)
            all_lcc_list.append(lcc_ratios)
            R_list.append(float(R))

        x_max = max(f[-1] for f in all_fracs_list)
        x_common = np.linspace(0, x_max, 400)
        y_interp = [np.interp(x_common, f, l)
                    for f, l in zip(all_fracs_list, all_lcc_list)]
        y_mean = np.mean(y_interp, axis=0)
        y_std = np.std(y_interp, axis=0)
        R_mean = float(np.mean(R_list))
        R_std = float(np.std(R_list))

        idx_half = np.argmax(y_mean <= 0.5)
        frac_critical = float(x_common[idx_half]) if idx_half > 0 else float(x_common[-1])

        _n = len(x_common)
        _m = max(3, int(_n * 0.05))
        _dy = np.gradient(y_mean[_m:-_m], x_common[_m:-_m])
        frac_peak = float(x_common[_m + int(np.argmin(_dy))])

        results.append({
            "城市": city, "节点数": N0,
            "健壮性R均值": round(R_mean, 4),
            "健壮性R标准差": round(R_std, 5),
            "p(LCC=0.5)": round(frac_critical, 3),
            "p(最大下降速率)": round(frac_peak, 3)
        })

        ax = axes[idx]
        ax.fill_between(x_common, y_mean - y_std, y_mean + y_std,
                        alpha=0.25, color='steelblue')
        ax.plot(x_common, y_mean, 'b-', lw=2,
                label=f'均值 R={R_mean:.3f}±{R_std:.3f}')
        ax.axvline(frac_critical, color='r', ls='--', lw=1.5,
                   label=f'p₀.₅={frac_critical:.2f}')
        ax.axvline(frac_peak, color='orange', ls=':', lw=1.5,
                   label=f'p*={frac_peak:.2f}')
        ax.axhline(0.5, color='gray', ls=':', lw=1)
        ax.set_title(f'{city}', fontsize=10)
        ax.set_xlabel('故障节点比例 p')
        ax.set_ylabel('LCC/N₀')
        ax.legend(fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

    plt.suptitle(f'问题2：随机故障性能曲线（{n_trials}次平均）', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/Q2_random_failure.png', dpi=150)
    plt.close()

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('outputs/Q2_random_failure.csv', index=False, encoding='utf-8-sig')
    print("\n[已保存] Q2_random_failure.csv  &  Q2_random_failure.png")
    return df

def _approx_k(N):
    return max(30, min(150, N // 60))

def targeted_attack_order(G, recalc_steps=25):
    N = G.number_of_nodes()
    recalc_interval = max(1, N // recalc_steps)
    k_approx = _approx_k(N)

    Gwork = G.copy()
    order = []
    pending = []
    removed_since_recalc = 0

    while Gwork.number_of_nodes() > 0:
        if not pending or removed_since_recalc >= recalc_interval:
            if Gwork.number_of_nodes() == 0:
                break
            k = min(k_approx, Gwork.number_of_nodes())
            bc = nx.betweenness_centrality(Gwork, k=k, normalized=True, seed=0)
            pending = sorted(bc, key=bc.get, reverse=True)
            removed_since_recalc = 0

        target = pending.pop(0)
        if target not in Gwork:
            continue
        order.append(target)
        Gwork.remove_node(target)
        removed_since_recalc += 1

    return order

def problem3_optimal_attack():
    print("\n" + "="*70)
    print("问题3：最优蓄意攻击（自适应近似 HBA，25次介数重算）")
    print("="*70)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    results = []

    for idx, city in enumerate(CITY_FILES):
        G = load_graph(city)
        N0 = G.number_of_nodes()
        print(f"  处理 {city} (N={N0}, k_approx={_approx_k(N0)})...")

        attack_order = targeted_attack_order(G, recalc_steps=25)
        fracs_att, lcc_att, _ = compute_robustness_curve(G, attack_order)

        idx_01 = np.argmax(lcc_att <= 0.01)
        if idx_01 == 0:
            idx_01 = len(lcc_att) - 1
        frac_01 = float(fracs_att[idx_01])
        R_attack = float(_trapz(lcc_att[:idx_01+1], fracs_att[:idx_01+1]))

        np.random.seed(42)
        rand_order = np.random.permutation(list(G.nodes())).tolist()
        fracs_rand, lcc_rand, R_rand = compute_robustness_curve(G, rand_order)

        results.append({
            "城市": city, "节点数": N0,
            "蓄意攻击R": round(R_attack, 4),
            "随机攻击R": round(float(R_rand), 4),
            "攻击效力(ΔR)": round(float(R_rand) - R_attack, 4),
            "LCC=1%时故障比例": round(frac_01, 3)
        })

        ax = axes[idx]
        ax.plot(fracs_att, lcc_att, 'r-', lw=2,
                label=f'蓄意 R={R_attack:.3f}')
        ax.plot(fracs_rand, lcc_rand, 'b--', lw=1.5,
                label=f'随机 R={R_rand:.3f}')
        ax.axhline(0.01, color='gray', ls=':', lw=1, label='LCC=1%')
        ax.set_title(f'{city}', fontsize=10)
        ax.set_xlabel('移除节点比例 p')
        ax.set_ylabel('LCC/N₀')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

    plt.suptitle('问题3：自适应近似HBA蓄意攻击 vs 随机攻击', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/Q3_targeted_attack.png', dpi=150)
    plt.close()

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    best_city = df.loc[df['蓄意攻击R'].idxmax(), '城市']
    print(f"\n>>> 问题3结论：蓄意攻击下健壮性最大城市 = {best_city}")

    df.to_csv('outputs/Q3_targeted_attack.csv', index=False, encoding='utf-8-sig')
    print("[已保存] Q3_targeted_attack.csv  &  Q3_targeted_attack.png")
    return df, best_city

def spatial_attack_order_with_radius(G, radius):
    Gwork = G.copy()
    all_nodes = np.array(list(G.nodes()))
    all_coords = np.array([[G.nodes[n].get('x', 0), G.nodes[n].get('y', 0)]
                            for n in all_nodes])
    tree = cKDTree(all_coords)

    deg_dict = dict(Gwork.degree())
    heap = [(-d, n) for n, d in deg_dict.items()]
    heapq.heapify(heap)

    alive = set(Gwork.nodes())
    removal_batches = []

    while alive:
        if len(alive) < 3:
            removal_batches.append(list(alive))
            break

        target = None
        while heap:
            neg_d, node = heapq.heappop(heap)
            if node not in alive:
                continue
            actual_deg = Gwork.degree(node) if node in Gwork else 0
            if actual_deg == -neg_d:
                target = node
                break
            if node in alive:
                heapq.heappush(heap, (-actual_deg, node))

        if target is None:
            target = next(iter(alive))

        cx = G.nodes[target].get('x', 0)
        cy = G.nodes[target].get('y', 0)
        if radius > 0:
            idxs = tree.query_ball_point([cx, cy], radius)
            batch = [all_nodes[i] for i in idxs if all_nodes[i] in alive]
        else:
            batch = [target]

        if not batch:
            batch = [target]

        removal_batches.append(batch)

        batch_set = set(batch)
        for n in batch:
            if n in Gwork:
                for nbr in Gwork.neighbors(n):
                    if nbr in alive and nbr not in batch_set:
                        new_d = deg_dict.get(nbr, 0) - 1
                        deg_dict[nbr] = new_d

        Gwork.remove_nodes_from(batch)
        alive -= batch_set

    return removal_batches

def compute_robustness_spatial(G, removal_batches):
    N0 = G.number_of_nodes()
    Gwork = G.copy()
    fracs = [0.0]
    lcc_ratios = [1.0]
    removed = 0

    for batch in removal_batches:
        valid = [n for n in batch if n in Gwork]
        Gwork.remove_nodes_from(valid)
        removed += len(valid)
        frac = removed / N0
        if Gwork.number_of_nodes() == 0:
            lcc = 0
        else:
            lcc = max(len(c) for c in nx.connected_components(Gwork))
        fracs.append(min(frac, 1.0))
        lcc_ratios.append(lcc / N0)
        if lcc / N0 < 0.005:
            break

    R = _trapz(lcc_ratios, fracs)
    return np.array(fracs), np.array(lcc_ratios), R

def problem4_spatial_failure(radii=None):
    if radii is None:
        radii = [0, 500, 1000, 2000, 5000]

    print("\n" + "="*70)
    print("问题4：空间级联故障（懒惰堆优化波及半径攻击）")
    print("="*70)

    all_results = []
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(radii)))

    for idx, city in enumerate(CITY_FILES):
        G = load_graph(city)
        N0 = G.number_of_nodes()
        print(f"  处理 {city} (N={N0})...")
        ax = axes[idx]

        for r_idx, radius in enumerate(radii):
            batches = spatial_attack_order_with_radius(G, radius)
            fracs, lcc_ratios, _ = compute_robustness_spatial(G, batches)

            idx_01 = np.argmax(lcc_ratios <= 0.01)
            if idx_01 == 0:
                idx_01 = len(lcc_ratios) - 1
            R_01 = float(_trapz(lcc_ratios[:idx_01+1], fracs[:idx_01+1]))

            ax.plot(fracs, lcc_ratios, color=colors[r_idx], lw=1.8,
                    label=f'r={radius}m R={R_01:.3f}')
            all_results.append({
                "城市": city, "半径(m)": radius,
                "健壮性R": round(R_01, 4), "节点数": N0
            })

        ax.axhline(0.01, color='gray', ls=':', lw=1)
        ax.set_title(f'{city}', fontsize=10)
        ax.set_xlabel('攻击（批次）节点比例')
        ax.set_ylabel('LCC/N₀')
        ax.legend(fontsize=5.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

    plt.suptitle('问题4：不同波及半径下空间攻击性能曲线', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/Q4_spatial_attack.png', dpi=150)
    plt.close()

    df_all = pd.DataFrame(all_results)
    for radius in radii:
        sub = df_all[df_all['半径(m)'] == radius]
        best = sub.loc[sub['健壮性R'].idxmax(), '城市']
        print(f"  radius={radius}m → 最健壮城市: {best} "
              f"(R={sub['健壮性R'].max():.4f})")

    df_all.to_csv('outputs/Q4_spatial_attack.csv',
                  index=False, encoding='utf-8-sig')
    print("[已保存] Q4_spatial_attack.csv  &  Q4_spatial_attack.png")
    return df_all

def edge_cost(G, u, v):
    x1, y1 = G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)
    x2, y2 = G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)
    return float(np.hypot(x1 - x2, y1 - y2))

def find_bypass_candidates(G, n_top_bc=None, n_pairs=3):
    N = G.number_of_nodes()
    if n_top_bc is None:
        n_top_bc = max(20, N // 100)
    k_approx = _approx_k(N)
    bc = nx.betweenness_centrality(G, k=k_approx, normalized=True, seed=0)

    top_nodes = sorted(bc, key=bc.get, reverse=True)[:n_top_bc]
    existing_keys = set()
    for u, v in G.edges():
        existing_keys.add((min(u, v), max(u, v)))
    candidates = []

    for v in top_nodes:
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 2:
            continue
        vx = G.nodes[v].get('x', 0)
        vy = G.nodes[v].get('y', 0)

        angles = {}
        for nb in neighbors:
            nx_ = G.nodes[nb].get('x', 0)
            ny_ = G.nodes[nb].get('y', 0)
            angles[nb] = float(np.arctan2(ny_ - vy, nx_ - vx))

        sorted_nbrs = sorted(neighbors, key=lambda n: angles[n])
        seen_pairs = set()
        added = 0

        for u in sorted_nbrs:
            if added >= n_pairs:
                break
            target_ang = angles[u] + np.pi
            if target_ang > np.pi:
                target_ang -= 2 * np.pi
            best_w = min(
                [nb for nb in sorted_nbrs if nb != u],
                key=lambda n: abs(((angles[n] - target_ang + np.pi)
                                   % (2 * np.pi)) - np.pi)
            )
            pair_key = (min(u, best_w), max(u, best_w))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if pair_key not in existing_keys:
                c = edge_cost(G, u, best_w)
                val = bc[v] / (c + 1e-9)
                candidates.append((val, c, u, best_w, v))
                existing_keys.add(pair_key)
                added += 1

    candidates.sort(reverse=True)
    return candidates

def greedy_edge_addition(G, target_R, max_edges=150, eval_interval=8):
    Gnew = G.copy()
    added_edges = []
    total_cost = 0.0
    added_keys = set()

    def quick_R(g):
        N = g.number_of_nodes()
        k = min(50, _approx_k(N))
        bc = nx.betweenness_centrality(g, k=k, normalized=True, seed=0)
        order = sorted(g.nodes(), key=lambda n: -bc[n])
        _, _, R = compute_robustness_curve(g, order, step_frac=0.02)
        return float(R)

    current_R = quick_R(Gnew)
    print(f"    初始R={current_R:.4f}, 目标R={target_R:.4f}")

    if current_R >= target_R:
        return added_edges, total_cost, current_R

    cand_queue = find_bypass_candidates(Gnew)

    for iteration in range(max_edges):
        if current_R >= target_R:
            break

        chosen = None
        while cand_queue:
            val, c, u, w, via = cand_queue.pop(0)
            key = (min(u, w), max(u, w))
            if key not in added_keys and not Gnew.has_edge(u, w):
                chosen = (u, w, c)
                break

        if chosen is None:
            top30 = sorted(Gnew.nodes(), key=lambda n: -Gnew.degree(n))[:30]
            extra = []
            for i in range(len(top30)):
                for j in range(i + 1, len(top30)):
                    u2, w2 = top30[i], top30[j]
                    key2 = (min(u2, w2), max(u2, w2))
                    if key2 not in added_keys and not Gnew.has_edge(u2, w2):
                        c2 = edge_cost(Gnew, u2, w2)
                        extra.append((Gnew.degree(u2) + Gnew.degree(w2), c2, u2, w2, -1))
            extra.sort(reverse=True)
            cand_queue = extra[:100]
            if not cand_queue:
                break
            continue

        u, w, c = chosen
        Gnew.add_edge(u, w, weight=c)
        added_keys.add((min(u, w), max(u, w)))
        added_edges.append((u, w, round(c, 2)))
        total_cost += c

        if (iteration + 1) % eval_interval == 0:
            current_R = quick_R(Gnew)
            print(f"    iter={iteration+1:3d}, +{len(added_edges)}条边, "
                  f"cost={total_cost/1000:.1f}km, R={current_R:.4f}")

            if current_R < target_R:
                new_cands = find_bypass_candidates(Gnew)
                cand_queue = [(v, c2, u2, w2, via) for v, c2, u2, w2, via in new_cands
                              if (min(u2, w2), max(u2, w2)) not in added_keys][:200]

    final_R = quick_R(Gnew)
    return added_edges, total_cost, final_R

def problem5_edge_addition(n_edges_budget=120):
    print("\n" + "="*70)
    print("问题5：最优加边提升健壮性（Betweenness Hardening，与Q3指标一致）")
    print("="*70)

    print("  计算各城市基线健壮性（近似介数攻击，k=30，与Q3一致）...")
    city_R = {}
    for city in CITY_FILES:
        G = load_graph(city)
        N = G.number_of_nodes()
        k = min(50, _approx_k(N))
        bc = nx.betweenness_centrality(G, k=k, normalized=True, seed=0)
        order = sorted(G.nodes(), key=lambda n: -bc[n])
        _, _, R = compute_robustness_curve(G, order, step_frac=0.02)
        city_R[city] = float(R)
        print(f"    {city}: R={R:.4f}")

    best_city = max(city_R, key=city_R.get)
    target_R = city_R[best_city]
    print(f"\n  最健壮城市: {best_city}, R_target={target_R:.4f}")

    results = []
    for city in CITY_FILES:
        if city == best_city:
            results.append({
                "城市": city, "初始R": round(city_R[city], 4),
                "目标R": round(target_R, 4), "新增边数": 0,
                "总代价(km)": 0.0, "最终R": round(city_R[city], 4),
                "备注": "目标城市"
            })
            continue

        if city_R[city] >= target_R:
            results.append({
                "城市": city, "初始R": round(city_R[city], 4),
                "目标R": round(target_R, 4), "新增边数": 0,
                "总代价(km)": 0.0, "最终R": round(city_R[city], 4),
                "备注": "已达目标"
            })
            continue

        print(f"\n  处理 {city} (初始R={city_R[city]:.4f})...")
        G = load_graph(city)
        added, cost, final_R = greedy_edge_addition(
            G, target_R, max_edges=n_edges_budget)

        results.append({
            "城市": city, "初始R": round(city_R[city], 4),
            "目标R": round(target_R, 4), "新增边数": len(added),
            "总代价(km)": round(cost / 1000, 2), "最终R": round(final_R, 4),
            "备注": "已达目标" if final_R >= target_R - 5e-4 else "预算不足"
        })

        if added:
            pd.DataFrame(added, columns=['节点U', '节点V', '长度(m)']).to_csv(
                f'outputs/Q5_{city}_new_edges.csv', index=False,
                encoding='utf-8-sig')

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    df.to_csv('outputs/Q5_edge_addition.csv', index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(13, 6))
    cities = df['城市'].tolist()
    x = np.arange(len(cities))
    w = 0.35
    ax.bar(x - w / 2, df['初始R'], w, label='初始R', color='steelblue', alpha=0.85)
    ax.bar(x + w / 2, df['最终R'], w, label='加边后R', color='coral', alpha=0.85)
    ax.axhline(target_R, color='green', ls='--', lw=2,
               label=f'目标R={target_R:.3f} ({best_city})')
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha='right')
    ax.set_ylabel('健壮性 R')
    ax.set_title('问题5：Betweenness Hardening 加边前后健壮性对比')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/Q5_edge_addition.png', dpi=150)
    plt.close()

    print("[已保存] Q5_edge_addition.csv  &  Q5_edge_addition.png")
    return df

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   北京大学江泽涵杯 B题：交通网络健壮性 — 优化版            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df1 = problem1_basic_stats()
    df2 = problem2_random_failure(n_trials=10)
    df3, best_city_q3 = problem3_optimal_attack()
    df4 = problem4_spatial_failure(radii=[0, 500, 1000, 2000, 5000])
    df5 = problem5_edge_addition(n_edges_budget=200)

    print("\n\n✅ 所有问题计算完成！输出文件列表：")
    for f in sorted(os.listdir('outputs')):
        print(f"   outputs/{f}")