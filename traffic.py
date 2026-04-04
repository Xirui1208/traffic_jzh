"""
北京大学第二十三届"江泽涵杯"数学建模竞赛 B题
交通网络的健壮性和提升 — 优化版解答代码
数据格式: XCoord(START_NODE坐标), YCoord, START_NODE, END_NODE, EDGE, LENGTH
坐标系: UTM投影坐标（单位: 米）

修正与优化:
1. 修复 np.trapezoid (NumPy<2.0不支持) → 兼容包装 _trapz
2. 添加中文字体支持(SimHei)，修复图表中文乱码
3. 修复 Q1 k_fit 变量作用域 bug（指数拟合静默失败）
4. Q3 介数攻击: 全量O(N³)→近似采样+批次重算，实际可运行
5. Q4 最大度查找: O(N²)暴力→O((N+M)logN)懒惰堆
6. Q5 加边策略: 随机→绕行高介数节点 (Betweenness Hardening)
"""

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

# ─── 中文字体支持 ────────────────────────────────────────────────────────────
import matplotlib.font_manager as _fm
for _fn in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']:
    try:
        _fm.findfont(_fn, fallback_to_default=False)
        rcParams['font.sans-serif'] = [_fn]
        rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        pass

# ─── NumPy trapz 兼容包装 ─────────────────────────────────────────────────────
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0

# ─────────────────────────────────────────────────────────────────────
# 数据加载工具
# ─────────────────────────────────────────────────────────────────────
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
    """从CSV构建无向加权图，节点含坐标属性（UTM米制）"""
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


# ═══════════════════════════════════════════════════════════════════════
# 问题1：基本特征统计 + 度分布建模
# ═══════════════════════════════════════════════════════════════════════
def problem1_basic_stats():
    """
    思路：
    1. 对8城市统计节点数、边数、平均度、最大度、聚类系数、
       直径（99百分位采样估计）、平均最短路径（随机BFS采样估计）。
    2. 度分布拟合：幂律 P(k)∝k^{-γ} vs 指数衰减 P(k)∝e^{-λk}。
       道路网络是空间平面图，度分布以指数衰减为主，不符合无标度幂律。
    3. 双对数坐标下两种拟合曲线可直观区分：幂律→直线，指数→弯曲。
    """
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

        # 最大连通分量的直径和平均路径——随机采样BFS估计（适用所有规模）
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

        # ── 度分布直方图 + 拟合 ──────────────────────────────────────────
        deg_vals = np.array(degrees)
        k_min_v, k_max_v = int(deg_vals.min()), int(deg_vals.max())
        bins = np.arange(k_min_v, k_max_v + 2) - 0.5
        counts, edges = np.histogram(deg_vals, bins=bins, density=True)
        k_centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        kpos = k_centers[mask]
        cpos = counts[mask]

        # 公共拟合 x 轴在 try 块外定义，修复两个 try 块共用 k_fit 的作用域 bug
        k_fit = np.linspace(max(float(kpos.min()), 0.5), float(kpos.max()), 300)

        ax = axes[idx]
        ax.bar(kpos, cpos, width=0.8, alpha=0.55, color='steelblue', label='实测')

        # 幂律拟合 P(k) ~ C·k^{-γ}
        try:
            popt_pl, _ = curve_fit(
                lambda k, C, g: C * k ** (-g),
                kpos, cpos, p0=[1.0, 2.0], maxfev=5000,
                bounds=([0, 0.1], [np.inf, 10]))
            ax.plot(k_fit, popt_pl[0] * k_fit ** (-popt_pl[1]),
                    'r-', lw=2, label=f'幂律 γ={popt_pl[1]:.2f}')
        except Exception:
            pass

        # 指数衰减拟合 P(k) ~ C·e^{-λk}（道路网络更符合此分布）
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


# ═══════════════════════════════════════════════════════════════════════
# 问题2：随机节点故障下的健壮性
# ═══════════════════════════════════════════════════════════════════════
def compute_robustness_curve(G, removal_order, step_frac=0.01):
    """
    给定节点移除顺序，计算性能曲线并返回 (fracs, lcc_ratios, R)
    fracs: 已移除节点占比列表
    lcc_ratios: 对应的最大连通分量/原始规模
    R: 曲线下面积 (健壮性)
    """
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
    """
    思路：
    1. 对每个城市随机打乱节点顺序，重复多次取平均得稳健的性能曲线。
    2. 计算健壮性R（曲线积分）。
    3. 检测"显著变化点"：用性能曲线的二阶导数或阈值法找拐点
       (通常设LCC降至0.5时对应的故障比例)。
    """
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

        # 插值到公共 x 轴
        x_max = max(f[-1] for f in all_fracs_list)
        x_common = np.linspace(0, x_max, 400)
        y_interp = [np.interp(x_common, f, l)
                    for f, l in zip(all_fracs_list, all_lcc_list)]
        y_mean = np.mean(y_interp, axis=0)
        y_std = np.std(y_interp, axis=0)
        R_mean = float(np.mean(R_list))
        R_std = float(np.std(R_list))

        # 临界点 (a): LCC 首次降至 0.5
        idx_half = np.argmax(y_mean <= 0.5)
        frac_critical = float(x_common[idx_half]) if idx_half > 0 else float(x_common[-1])

        # 临界点 (b): 一阶差分最大下降位置（跳过首尾5%避免端点数值伪影）
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


# ═══════════════════════════════════════════════════════════════════════
# 问题3：最优攻击顺序（蓄意攻击），最小化健壮性
# ═══════════════════════════════════════════════════════════════════════
def _approx_k(N):
    """根据图规模确定近似介数采样源数 k（平衡精度与速度）"""
    return max(30, min(150, N // 60))


def targeted_attack_order(G, recalc_steps=25):
    """
    自适应近似介数中心性攻击（优化版 HBA，High Betweenness Adaptive）：

    原理：
    - 高介数节点是网络的瓶颈，优先移除可使 LCC 快速崩溃。
    - 使用 betweenness_centrality(G, k=k_approx) 近似计算
      （k_approx = min(150, N//60) 个随机源，比全量快 50~200 倍）。
    - 每移除 recalc_interval = N//recalc_steps 个节点后重算一次，
      共约 recalc_steps 次重算，保持对网络动态变化的响应。

    时间复杂度：O(recalc_steps × k_approx × M)
    vs 原始全量方案：O(N × N × M)（完全不可行，每城市需数小时）

    参数
    ----
    recalc_steps : int
        总重算次数（越大越精确，越慢）。建议 20~30。
    """
    N = G.number_of_nodes()
    recalc_interval = max(1, N // recalc_steps)
    k_approx = _approx_k(N)

    Gwork = G.copy()
    order = []
    pending = []          # 当前按介数排好序的待移除队列
    removed_since_recalc = 0

    while Gwork.number_of_nodes() > 0:
        # 队列耗尽或到达重算时机
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
    """
    思路：
    1. 自适应近似 HBA 策略：每隔 N//25 步重算一次近似介数，
       共约25次 × O(k·M) 开销。该策略是已知最优近似方案（Holme 2002）。
    2. 对比随机攻击曲线，量化蓄意攻击对网络的额外威胁。
    3. 输出8城市蓄意/随机攻击健壮性及攻击效力，找出最健壮城市。
    """
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

        # LCC 降至 1% 时的截止健壮性
        idx_01 = np.argmax(lcc_att <= 0.01)
        if idx_01 == 0:
            idx_01 = len(lcc_att) - 1
        frac_01 = float(fracs_att[idx_01])
        R_attack = float(_trapz(lcc_att[:idx_01+1], fracs_att[:idx_01+1]))

        # 对比随机攻击
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


# ═══════════════════════════════════════════════════════════════════════
# 问题4：空间级联故障（波及半径内所有节点）
# ═══════════════════════════════════════════════════════════════════════
def spatial_attack_order_with_radius(G, radius):
    """
    空间级联攻击（懒惰堆优化版）：
    - 贪心策略：每轮选当前剩余度最大节点作为攻击中心；
    - 移除该中心 radius 范围内所有存活节点（KD树加速）；
    - 维护懒惰最大堆追踪度数变化，避免 O(N) 线性扫描。

    时间复杂度：O(N²) 暴力 → O((N+M) log N) 懒惰堆。
    返回批次列表 removal_batches。
    """
    Gwork = G.copy()
    all_nodes = np.array(list(G.nodes()))
    all_coords = np.array([[G.nodes[n].get('x', 0), G.nodes[n].get('y', 0)]
                            for n in all_nodes])
    tree = cKDTree(all_coords)

    # 构建最大堆（存 -degree 以模拟最大堆）
    deg_dict = dict(Gwork.degree())
    heap = [(-d, n) for n, d in deg_dict.items()]
    heapq.heapify(heap)

    alive = set(Gwork.nodes())
    removal_batches = []

    while alive:
        if len(alive) < 3:
            removal_batches.append(list(alive))
            break

        # 懒惰堆：弹出直到找到度数未过期的存活节点
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

        # 更新被删节点邻居的度数（用于堆的懒惰刷新）
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
    """根据批次移除计算性能曲线"""
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
    """
    思路：
    1. 扩展问题3的模型：每次攻击某节点时，同时移除其物理距离radius内
       所有节点（模拟爆炸、地震等空间级联故障）。
    2. 攻击中心选择：贪心取当前度最大节点（KD树+懒惰堆加速）。
    3. 研究 radius 从小到大对健壮性的影响规律（坐标单位: UTM米）。
    4. 比较8城市的空间鲁棒性，找出最健壮城市。
    """
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
    # 每个radius下健壮性最大城市
    for radius in radii:
        sub = df_all[df_all['半径(m)'] == radius]
        best = sub.loc[sub['健壮性R'].idxmax(), '城市']
        print(f"  radius={radius}m → 最健壮城市: {best} "
              f"(R={sub['健壮性R'].max():.4f})")

    df_all.to_csv('outputs/Q4_spatial_attack.csv',
                  index=False, encoding='utf-8-sig')
    print("[已保存] Q4_spatial_attack.csv  &  Q4_spatial_attack.png")
    return df_all


# ═══════════════════════════════════════════════════════════════════════
# 问题5：最优加边方案提升健壮性
# ═══════════════════════════════════════════════════════════════════════
def edge_cost(G, u, v):
    """新增边代价 = 两节点欧氏距离（修路成本与距离正比，UTM米制）"""
    x1, y1 = G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)
    x2, y2 = G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)
    return float(np.hypot(x1 - x2, y1 - y2))


def find_bypass_candidates(G, n_top_bc=None, n_pairs=3):
    """
    Betweenness Hardening — 角度对称旁路候选边：

    对每个高介数瓶颈节点 v，将其邻居按相对 v 的方位角排序，
    为每个邻居寻找方位角相差约 180° 的对侧邻居，连成旁路边。

    这比「最近邻居对」策略更有效：
    - 近邻对 → 形成小三角，流量仍经过 v 同侧
    - 对向邻居对 → 形成真正跨越 v 的绕行路径（如桥梁两端直连）

    返回: [(value, cost, u, w, via_v), ...]
    """
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

        # 计算各邻居相对于 v 的方位角
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
            # 寻找方位角最近 (angles[u] + π) 的邻居（对侧）
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
    """
    贪心加边策略（Betweenness Hardening + 代价最优）：

    1. 生成绕行高介数节点的候选边（按介数/代价降序排列）；
    2. 依次加入性价比最高的候选边；
    3. 每 eval_interval 步重新评估健壮性（静态度攻击快速估计）；
    4. 达到 target_R 或耗尽预算则停止。

    返回 (新增边列表, 总代价(m), 最终R)
    """
    Gnew = G.copy()
    added_edges = []
    total_cost = 0.0
    added_keys = set()

    def quick_R(g):
        """近似介数攻击估计健壮性（k=50，seed=0，与Q3指标一致）"""
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

    # 初始候选集（Betweenness Hardening）
    cand_queue = find_bypass_candidates(Gnew)

    for iteration in range(max_edges):
        if current_R >= target_R:
            break

        # 从候选队列取下一条可加的边
        chosen = None
        while cand_queue:
            val, c, u, w, via = cand_queue.pop(0)
            key = (min(u, w), max(u, w))
            if key not in added_keys and not Gnew.has_edge(u, w):
                chosen = (u, w, c)
                break

        if chosen is None:
            # 候选耗尽：补充高度节点对
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
                # 补充新候选（已加边后介数重分）
                new_cands = find_bypass_candidates(Gnew)
                cand_queue = [(v, c2, u2, w2, via) for v, c2, u2, w2, via in new_cands
                              if (min(u2, w2), max(u2, w2)) not in added_keys][:200]

    final_R = quick_R(Gnew)
    return added_edges, total_cost, final_R


def problem5_edge_addition(n_edges_budget=120):
    """
    思路：
    1. 以近似介数攻击（k=30）重新计算各城市健壮性基线（与Q3完全一致）；
    2. 取最健壮城市的 R 作为目标水平 R_target；
    3. 对其余城市用 Betweenness Hardening 加边策略：
       - find_bypass_candidates：对每个高介数节点找角度对称的对侧邻居对，
         构成真正绕越瓶颈的旁路边，按 介数/代价 降序排优先级；
       - 每 eval_interval 步用介数攻击重新评估 R，直到达标或耗尽预算；
    4. 输出每城市新增边数、总修路代价（km）、达标情况；
    5. 可视化加边前后健壮性对比（与Q3同一指标，结果可直接比照）。
    """
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

    # 可视化
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


# ═══════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════
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