import csv
import argparse

def longest_common_subsequence_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def character_accuracy(gt: str, pred: str) -> float:
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    lcs_len = longest_common_subsequence_length(gt, pred)
    return lcs_len / len(gt)

def main():
    model_name = "BaseCNN_epoch25_hidden128_channel3_lr0.001_batchsize16"
    csv_file = f"test_results/test_results_{model_name}.csv"
    global_lcs = 0
    total_gt_chars = 0
    row_accuracies = []
    total_rows = 0
    whole_string_correct = 0
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)[:-1]
        total_rows = len(rows)
        with open(f"test_results/test_summary_{model_name}.txt", "w", encoding="utf-8") as outfile:
            outfile.write("Per-row character accuracy (based on longest common subsequence):\n")
            print("Per-row character accuracy (based on longest common subsequence):")
            for row in rows:
                gt = row["Ground Truth"]
                pred = row["Prediction"]
                acc = character_accuracy(gt, pred)
                row_accuracies.append(acc)
                total_gt_chars += len(gt)
                lcs = longest_common_subsequence_length(gt, pred)
                global_lcs += lcs
                if gt == pred:
                    whole_string_correct += 1
                line = f"GT: {gt:20s} | Pred: {pred:20s} | Accuracy: {acc * 100:6.2f}%"
                print(line)
                outfile.write(line + "\n")
            global_accuracy = global_lcs / total_gt_chars if total_gt_chars > 0 else 1.0
            whole_string_accuracy = whole_string_correct / total_rows if total_rows > 0 else 1.0
            summary_lines = [
                "\nSummary:",
                f"Global character accuracy: {global_accuracy * 100:.2f}%",
                f"Whole string accuracy: {whole_string_accuracy * 100:.2f}%"
            ]
            for s in summary_lines:
                print(s)
                outfile.write(s + "\n")

if __name__ == "__main__":
    main()

