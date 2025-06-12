def evaluate_accuracy(predicted_path, gold_path):
    total, correct = 0, 0
    with open(predicted_path, "r", encoding="utf-8") as pred_f, \
         open(gold_path, "r", encoding="utf-8") as gold_f:
        for pred_line, gold_line in zip(pred_f, gold_f):
            _, pred_tags = pred_line.strip().split("\t")
            _, gold_tags = gold_line.strip().split("\t")
            pred_tags = pred_tags.split()
            gold_tags = gold_tags.split()
            for p, g in zip(pred_tags, gold_tags):
                if p == g:
                    correct += 1
                total += 1
    acc = correct / total if total > 0 else 0
    print(f"🎯 Accuracy: {acc:.2%}")
