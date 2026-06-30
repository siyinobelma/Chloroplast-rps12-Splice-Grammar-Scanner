import random
import csv
from config import Config
from scripts.utils import load_pairs

def swap_exon1(pos, all_exons1):
    """随机替换外显子1为其他物种的同一类型外显子"""
    new_e1 = random.choice(all_exons1)
    pos['exon1_seq'] = new_e1['exon1_seq']
    return pos

def swap_exon2(pos, all_exons2):
    """随机替换外显子2"""
    new_e2 = random.choice(all_exons2)
    pos['exon2_seq'] = new_e2['exon2_seq']
    return pos

def reverse_intron(pos):
    """反转内含子序列"""
    pos['intron_seq'] = pos['intron_seq'][::-1]
    return pos

def shuffle_intron(pos):
    """随机打乱内含子"""
    seq = list(pos['intron_seq'])
    random.shuffle(seq)
    pos['intron_seq'] = ''.join(seq)
    return pos

def main():
    random.seed(Config.SEED)
    pos_pairs = load_pairs(Config.RAW_DATA)
    # 按类型分组以保持负样本的生物学背景
    groups = {}
    for p in pos_pairs:
        t = p['type']
        groups.setdefault(t, []).append(p)

    negatives = []
    for t, items in groups.items():
        all_e1 = [{'exon1_seq': it['exon1_seq']} for it in items]
        all_e2 = [{'exon2_seq': it['exon2_seq']} for it in items]
        for i, item in enumerate(items):
            neg = item.copy()
            # 随机选一种破坏方法
            method = random.choice(Config.NEGATIVE_METHODS)
            if method == "swap_exon1":
                neg = swap_exon1(neg, all_e1)
            elif method == "swap_exon2":
                neg = swap_exon2(neg, all_e2)
            elif method == "reverse_intron":
                neg = reverse_intron(neg)
            elif method == "shuffle_intron":
                neg = shuffle_intron(neg)
            neg['label'] = 0
            negatives.append(neg)
            item['label'] = 1

    all_data = pos_pairs + negatives
    random.shuffle(all_data)

    out_path = Config.RAW_DATA.replace('.csv', '_with_neg.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_data[0].keys()), delimiter='\t')
        writer.writeheader()
        writer.writerows(all_data)
    print(f"Generated {len(negatives)} negatives, total {len(all_data)}. Saved to {out_path}")

if __name__ == '__main__':
    main()