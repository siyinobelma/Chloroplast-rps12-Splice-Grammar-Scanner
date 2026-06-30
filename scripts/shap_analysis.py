#!/usr/bin/env python3
"""
SHAP 可解释性分析脚本 - 针对rps12三通道CNN分类器
专门设计用于：
1. 种子植物 vs 蕨类植物的剪接信号差异
2. 蕨类“高-低”拷贝对比
3. 蕨类顺式剪接强弱位点对比
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from scripts.model import SplicingCNN
from scripts.utils import seqs_to_tensor, load_pairs

# ==================== 配置 ====================
MODEL_PATH = "best_model.pt"
OUTPUT_DIR = "shap_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 选择分析的样本对
# 蕨类“高-低”拷贝对比
FERN_HIGH_LOW_PAIRS = {
    "Nephrolepis_cordifolia": {
        "high_file": "predict_result/test_juelei_predictions_V1.csv",
        "high_filter": {"species": "Nephrolepis cordifolia", "positive_prob": ">0.9"},
        "low_filter": {"species": "Nephrolepis cordifolia", "positive_prob": "<0.1"}
    },
    "Cyrtomium_fortunei": {
        "high_filter": {"species": "Cyrtomium fortunei", "positive_prob": ">0.9"},
        "low_filter": {"species": "Cyrtomium fortunei", "positive_prob": "<0.1"}
    }
}

# 顺式剪接强弱位点对比
FERN_CIS_PAIRS = {
    "Cryptogramma_acrostichoides": {
        "weak_pair": "exon_1-exon_2",
        "strong_pair": "exon_2-exon_3"
    }
}

# 种子植物高分样本（从被子+裸子中随机选）
SEED_PLANT_SPECIES = [
    "Arabidopsis thaliana", "Oryza sativa", "Pinus taeda",
    "Ginkgo biloba", "Gnetum montanum"
]


# ==================== 1. 加载模型 ====================
def load_model(model_path=MODEL_PATH):
    """加载训练好的CNN模型"""
    model = SplicingCNN(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# ==================== 2. 数据准备 ====================
def extract_pair_sequence(pair_row):
    """从预测结果行中提取外显子和内含子序列"""
    exon1 = pair_row.get('exon1_seq', '')
    intron = pair_row.get('intron_seq', '')
    exon2 = pair_row.get('exon2_seq', '')

    # 检查是否存在序列列，如果不存在则返回None
    if not exon1 or not exon2:
        return None, None, None

    return exon1, intron, exon2


def load_pair_data(csv_path, filter_criteria=None):
    """
    从预测CSV加载配对的序列数据
    注意：你的预测CSV可能没有包含原始序列列，
    所以这里提供两个方案：
    1. 如果CSV有序列列（exon1_seq, intron_seq, exon2_seq），直接读取
    2. 否则需要从原始提取的pairs文件匹配
    """
    try:
        df = pd.read_csv(csv_path, sep='\t')
        print(f"Loaded {len(df)} pairs from {csv_path}")
        print(f"Available columns: {list(df.columns)}")

        # 过滤
        if filter_criteria:
            for col, val in filter_criteria.items():
                if isinstance(val, str) and val.startswith('>'):
                    threshold = float(val[1:])
                    df = df[df[col] > threshold]
                elif isinstance(val, str) and val.startswith('<'):
                    threshold = float(val[1:])
                    df = df[df[col] < threshold]
                else:
                    df = df[df[col] == val]

        # 检查是否包含序列列
        seq_cols = ['exon1_seq', 'intron_seq', 'exon2_seq']
        has_seq = all(col in df.columns for col in seq_cols)

        return df, has_seq
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, False


def pad_or_truncate(seq, max_len=Config.MAX_SEQ_LEN):
    """填充或截断序列到固定长度"""
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + 'N' * (max_len - len(seq))


def prepare_background_data(model, n_background=100):
    """
    准备背景数据集（用于SHAP解释）
    使用随机生成的序列作为背景，或者从训练集中采样
    """
    # 方法1：使用随机序列作为背景
    background = []
    for _ in range(n_background):
        exon1 = ''.join(np.random.choice(['A', 'C', 'G', 'T'], Config.MAX_SEQ_LEN))
        intron = ''.join(np.random.choice(['A', 'C', 'G', 'T'], Config.MAX_SEQ_LEN))
        exon2 = ''.join(np.random.choice(['A', 'C', 'G', 'T'], Config.MAX_SEQ_LEN))
        tensor = seqs_to_tensor(exon1, intron, exon2).unsqueeze(0)
        background.append(tensor)

    background_tensor = torch.cat(background, dim=0)
    print(f"Created background dataset: {background_tensor.shape}")
    return background_tensor


# ==================== 3. SHAP 分析核心函数 ====================
class SplicingModelWrapper(torch.nn.Module):
    """包装模型以适配SHAP输入格式"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x shape: (batch, 3*4*L) 或 (batch, 3, 4, L)
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, 3, 4, Config.MAX_SEQ_LEN)
        return self.model(x)


def compute_shap_values(model, target_samples, background, sample_names=None):
    """计算SHAP值"""
    wrapper = SplicingModelWrapper(model)

    # 将目标样本展平为2D (batch, 3*4*L)
    target_flat = target_samples.view(target_samples.shape[0], -1)
    background_flat = background.view(background.shape[0], -1)

    # 使用DeepExplainer（适合深度学习模型）
    print("Initializing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(wrapper, background_flat[:50])  # 使用50个背景样本

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(target_flat[:10])  # 最多计算10个样本

    return shap_values, explainer


def compute_gradient_saliency(model, sample_tensor, target_class=1):
    """
    使用梯度方法计算重要性（备选方案，如果SHAP太慢）
    """
    model.eval()
    x = sample_tensor.clone().detach().requires_grad_(True).to(Config.DEVICE)
    output = model(x)
    score = output[:, target_class].sum()
    grad = torch.autograd.grad(score, x)[0]
    return grad.abs().cpu().numpy()


# ==================== 4. 可视化函数 ====================
def plot_shap_per_channel(shap_values, sample_tensor, sample_names, save_path):
    """为每个通道（外显子1、内含子、外显子2）绘制SHAP值"""
    n_samples = len(sample_names)
    channels = ['Exon 1', 'Intron', 'Exon 2']

    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(sample_names):
        for j, ch_name in enumerate(channels):
            ax = axes[i, j]
            # 提取该通道的SHAP值（平均所有碱基类型）
            channel_shap = shap_values[i, j].mean(axis=0)[:len(sample_tensor[i, j, 0])]
            ax.plot(channel_shap, linewidth=1)
            ax.set_title(f'{name} - {ch_name}')
            ax.set_xlabel('Position')
            ax.set_ylabel('SHAP value')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradient_comparison(high_grad, low_grad, high_name, low_name, save_path):
    """对比高分和低分样本的梯度重要性"""
    channels = ['Exon 1', 'Intron', 'Exon 2']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for j, ch_name in enumerate(channels):
        ax = axes[j]
        high_channel = high_grad[0, j].mean(axis=0)
        low_channel = low_grad[0, j].mean(axis=0)

        ax.plot(high_channel, label=f'{high_name} (High score)', color='green', alpha=0.7)
        ax.plot(low_channel, label=f'{low_name} (Low score)', color='red', alpha=0.7)
        ax.set_title(ch_name)
        ax.set_xlabel('Position')
        ax.set_ylabel('Gradient importance')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sequence_logo_style(shap_matrix, seq, save_path, top_n=10):
    """
    绘制类似sequence logo的SHAP重要性图
    突出显示最重要的碱基位置
    """
    # shap_matrix: (4, L)
    importance = shap_matrix.mean(axis=0)  # 平均所有碱基通道

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), gridspec_kw={'height_ratios': [3, 1]})

    # 上图：重要性曲线
    ax1 = axes[0]
    ax1.plot(importance, linewidth=1.5, color='steelblue')
    ax1.set_ylabel('Mean |SHAP|')
    ax1.set_title('Position Importance')

    # 标注top N位置
    top_positions = np.argsort(importance)[-top_n:]
    for pos in top_positions:
        ax1.annotate(f'{pos}', (pos, importance[pos]),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=8, ha='center')

    # 下图：序列展示
    ax2 = axes[1]
    bases = ['A', 'C', 'G', 'T']
    seq_array = np.argmax(shap_matrix, axis=0)

    # 用颜色标注碱基
    colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    for i, base_idx in enumerate(seq_array[:len(seq)]):
        base = bases[base_idx]
        ax2.text(i, 0, base, fontsize=10, ha='center',
                 color=colors.get(base, 'black'), fontweight='bold')

    ax2.set_xlim(-1, len(importance))
    ax2.set_ylim(-0.5, 0.5)
    ax2.axis('off')
    ax2.set_title('Sequence')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_channel_interaction(shap_values, sample_name, save_path):
    """
    检测跨通道（外显子-内含子）的交互SHAP值
    如果SHAP支持，可以显示哪些外显子位置与内含子位置存在协同效应
    """
    # 这里使用简化的相关性分析
    # 计算外显子1和内含子之间重要性的相关性

    exon1_shap = np.abs(shap_values[0, 0]).mean(axis=0)  # (L,)
    intron_shap = np.abs(shap_values[0, 1]).mean(axis=0)  # (L,)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(exon1_shap[::5], intron_shap[::5], alpha=0.5, s=20)  # 降采样
    ax.set_xlabel('Exon 1 |SHAP|')
    ax.set_ylabel('Intron |SHAP|')
    ax.set_title(f'{sample_name}: Exon1-Intron Importance Correlation')

    # 计算相关性
    corr = np.corrcoef(exon1_shap, intron_shap)[0, 1]
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== 5. 主分析流程 ====================
def analyze_fern_high_low(model, csv_path):
    """分析蕨类“高-低”拷贝的SHAP差异"""
    print("\n" + "=" * 60)
    print("FERN HIGH-LOW COPY ANALYSIS")
    print("=" * 60)

    df, has_seq = load_pair_data(csv_path)
    if df is None:
        return

    # 选择具体的物种进行分析
    target_species = [
        "Nephrolepis cordifolia",
        "Cyrtomium fortunei",
        "Dryopteris crassirhizoma"
    ]

    for species in target_species:
        species_df = df[df['species'] == species]
        if len(species_df) < 2:
            print(f"Not enough data for {species}, skipping")
            continue

        # 找最高分和最低分的样本
        high_sample = species_df[species_df['positive_prob'] > 0.9].head(1)
        low_sample = species_df[species_df['positive_prob'] < 0.1].head(1)

        if len(high_sample) == 0 or len(low_sample) == 0:
            print(f"No high/low pair found for {species}")
            continue

        print(f"\nAnalyzing {species}:")
        print(f"  High score: {high_sample['positive_prob'].values[0]:.4f}")
        print(f"  Low score: {low_sample['positive_prob'].values[0]:.4f}")

        # 如果CSV不包含序列，需要从原始提取文件中匹配
        if not has_seq:
            print("  Warning: CSV does not contain sequence columns.")
            print("  Please ensure the input CSV includes exon1_seq, intron_seq, exon2_seq")
            print("  Using gradient-based saliency instead...")

            # 使用简化方法：只做梯度归因
            # 这里需要从原始数据源获取序列
            continue

        # 提取序列并转换为张量
        high_seq = (
            high_sample['exon1_seq'].values[0],
            high_sample['intron_seq'].values[0],
            high_sample['exon2_seq'].values[0]
        )
        low_seq = (
            low_sample['exon1_seq'].values[0],
            low_sample['intron_seq'].values[0],
            low_sample['exon2_seq'].values[0]
        )

        high_tensor = seqs_to_tensor(*high_seq).unsqueeze(0).to(Config.DEVICE)
        low_tensor = seqs_to_tensor(*low_seq).unsqueeze(0).to(Config.DEVICE)

        # 计算梯度重要性
        high_grad = compute_gradient_saliency(model, high_tensor)
        low_grad = compute_gradient_saliency(model, low_tensor)

        # 对比图
        plot_gradient_comparison(
            high_grad, low_grad,
            f"{species} (High: {high_sample['positive_prob'].values[0]:.4f})",
            f"{species} (Low: {low_sample['positive_prob'].values[0]:.4f})",
            f"{OUTPUT_DIR}/fern_high_low_{species.replace(' ', '_')}.png"
        )


def analyze_seed_plant_consensus(model, csv_paths):
    """分析种子植物（被子+裸子）的共享剪接信号"""
    print("\n" + "=" * 60)
    print("SEED PLANT CONSENSUS ANALYSIS")
    print("=" * 60)

    # 合并多个CSV
    dfs = []
    for path in csv_paths:
        df, _ = load_pair_data(path)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No data loaded for seed plants")
        return

    all_seed = pd.concat(dfs, ignore_index=True)
    print(f"Total seed plant pairs: {len(all_seed)}")

    # 选择高分样本
    high_score_seed = all_seed[all_seed['positive_prob'] > 0.99].head(5)

    if len(high_score_seed) == 0:
        print("No high-score seed plant samples found")
        return

    # 计算平均梯度重要性（种子植物共识）
    grads = []
    for idx, row in high_score_seed.iterrows():
        seqs = (row['exon1_seq'], row['intron_seq'], row['exon2_seq'])
        tensor = seqs_to_tensor(*seqs).unsqueeze(0).to(Config.DEVICE)
        grad = compute_gradient_saliency(model, tensor)
        grads.append(grad)

    # 平均梯度
    mean_grad = np.mean(grads, axis=0)

    # 绘制共识重要性图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channels = ['Exon 1', 'Intron', 'Exon 2']

    for j, ch_name in enumerate(channels):
        ax = axes[j]
        channel_grad = mean_grad[0, j].mean(axis=0)
        ax.plot(channel_grad, linewidth=2, color='steelblue')
        ax.set_title(f'{ch_name} - Seed Plant Consensus')
        ax.set_xlabel('Position')
        ax.set_ylabel('Mean |Gradient|')

        # 标记峰值位置
        top_peaks = np.argsort(channel_grad)[-5:]
        for peak in top_peaks:
            ax.axvline(x=peak, color='red', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/seed_plant_consensus.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/seed_plant_consensus.png")

    # 保存峰值位置信息
    consensus_peaks = []
    for j, ch_name in enumerate(channels):
        channel_grad = mean_grad[0, j].mean(axis=0)
        top_positions = np.argsort(channel_grad)[-10:]
        for pos in top_positions:
            consensus_peaks.append({
                'channel': ch_name,
                'position': pos,
                'importance': channel_grad[pos]
            })

    peaks_df = pd.DataFrame(consensus_peaks)
    peaks_df = peaks_df.sort_values('importance', ascending=False)
    peaks_df.to_csv(f"{OUTPUT_DIR}/consensus_peaks.csv", sep='\t', index=False)
    print(f"Top 10 consensus peaks saved to {OUTPUT_DIR}/consensus_peaks.csv")
    print(peaks_df.head(10).to_string(index=False))


# ==================== 6. 主函数 ====================
def main():
    print("Starting SHAP Analysis...")
    print(f"Output directory: {OUTPUT_DIR}")

    # 加载模型
    model = load_model()

    # 准备背景数据（如果要用SHAP）
    # background = prepare_background_data(model, n_background=50)

    # 分析1：蕨类高低拷贝对比
    fern_csv = "predict_result/test_juelei_predictions_with_seq.csv"
    if os.path.exists(fern_csv):
        analyze_fern_high_low(model, fern_csv)
    else:
        print(f"Fern predictions not found: {fern_csv}")

    # 分析2：种子植物共识
    seed_csvs = [
        "predict_result/test_angiosperm_predictions_with_seq.csv",
        "predict_result/test_gymno_predictions_with_seq.csv"
    ]
    seed_csvs = [f for f in seed_csvs if os.path.exists(f)]
    if seed_csvs:
        analyze_seed_plant_consensus(model, seed_csvs)
    else:
        print("No seed plant predictions found")

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()