import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from config import Config
from scripts.model import SplicingCNN
import matplotlib.pyplot as plt
from scripts.utils import seqs_to_tensor

def get_gradient_importance(model, seqs, target_class=1):
    """计算输入序列的梯度重要性（saliency map）"""
    model.eval()
    x = seqs.clone().detach().requires_grad_(True).to(Config.DEVICE)
    output = model(x)
    score = output[:, target_class].sum()      # 一个标量
    grad = torch.autograd.grad(score, x, create_graph=False)[0]  # 直接求导
    return grad.abs().cpu().numpy()

def plot_saliency(saliency, save_path="saliency_plot.png"):
    s = saliency[0]  # (3,4,L)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    titles = ['Exon 1', 'Intron', 'Exon 2']
    for i in range(3):
        ax = axes[i]
        avg = s[i].mean(axis=0)
        ax.plot(avg)
        ax.set_title(titles[i])
        ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saliency plot saved to {save_path}")

def mutation_scan(model, exon1, intron, exon2, pos_list, device):
    """对指定位置进行饱和突变，返回每个突变的影响矩阵"""
    import copy
    model.eval()
    original_tensor = seqs_to_tensor(exon1, intron, exon2).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_prob = torch.softmax(model(original_tensor), dim=1)[0, 1].item()

    effects = {}
    for pos, region in pos_list:  # region: 0 for exon1, 1 for intron, 2 for exon2
        for base_idx, base in enumerate(['A','C','G','T']):
            mutated = original_tensor.clone()
            # 将指定位置的one-hot向量改为新碱基
            mutated[0, region, :, pos] = 0
            mutated[0, region, base_idx, pos] = 1.0
            with torch.no_grad():
                prob = torch.softmax(model(mutated), dim=1)[0, 1].item()
            effects[(region, pos, base)] = orig_prob - prob
    return effects

def main():
    # 安全加载（消除 FutureWarning）
    test_data, test_labels = torch.load(Config.TEST_FILE, weights_only=True)
    model = SplicingCNN(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    model.eval()

    # 预测
    with torch.no_grad():
        outputs = model(test_data.to(Config.DEVICE))
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
    true = test_labels.numpy()

    auc = roc_auc_score(true, probs[:,1].cpu().numpy())
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(true, preds, target_names=['Negative', 'Positive']))

    # 梯度重要性（取正样本）
    pos_idx = np.where(true == 1)[0][:Config.BATCH_SIZE]
    pos_samples = test_data[pos_idx]
    saliency = get_gradient_importance(model, pos_samples, target_class=1)
    plot_saliency(saliency, "saliency_rps12.png")

    np.save("saliency_map.npy", saliency)
    print("Saliency map saved as saliency_map.npy")

if __name__ == '__main__':
    main()
