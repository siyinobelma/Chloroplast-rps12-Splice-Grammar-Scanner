import torch
import csv
import sys
import argparse
from config import Config
from scripts.model import SplicingCNN
from scripts.utils import seqs_to_tensor


def predict_pairs(model, pairs_csv, output_csv, device):
    """对给定的配对CSV进行预测，输出概率、标签和基因组坐标"""
    model.eval()

    # 读取配对文件
    with open(pairs_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)

    results = []
    for row in rows:
        try:
            exon1 = row['exon1_seq']
            intron = row['intron_seq']
            exon2 = row['exon2_seq']

            # 转换为张量
            tensor = seqs_to_tensor(exon1, intron, exon2).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()

            # 提取基因组坐标（如果存在）
            result = {
                'species': row.get('species', ''),
                'accession': row.get('accession', ''),
                'exon1_id': row.get('exon1_id', ''),
                'exon2_id': row.get('exon2_id', ''),
                'type': row.get('type', ''),
                'positive_prob': prob,
                'predicted_label': 1 if prob >= 0.5 else 0
            }

            # 添加基因组坐标（如果存在）
            coord_fields = [
                'exon1_start', 'exon1_end', 'exon1_strand',
                'intron_start', 'intron_end', 'intron_strand',
                'exon2_start', 'exon2_end', 'exon2_strand'
            ]
            for field in coord_fields:
                if field in row:
                    result[field] = row[field]

            results.append(result)

        except Exception as e:
            print(f"Skipping row due to error: {e}")

    # 写入输出
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(results)

        print(f"Predictions saved to {output_csv}")
        # 打印简要统计
        pos_count = sum(1 for r in results if r['predicted_label'] == 1)
        print(f"Total pairs: {len(results)}, predicted positive: {pos_count}")
    else:
        print("No valid pairs found in input.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True, help='Input pairs CSV from extract_rps12_pairs.py')
    parser.add_argument('--output', default='predictions.csv', help='Output predictions CSV')
    parser.add_argument('--model', default='best_model.pt', help='Path to trained model checkpoint')
    parser.add_argument('--include-coords', action='store_true', help='Include genomic coordinates in output')
    args = parser.parse_args()

    model = SplicingCNN(Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=Config.DEVICE, weights_only=True))

    # 调用修改后的 predict_pairs 函数
    predict_pairs(model, args.pairs, args.output, Config.DEVICE)

if __name__ == '__main__':
    main()