from collections import defaultdict
import json

def build_tag_stats(tagged_path, tag_counts_path, transition_counts_path):
    """Etiketli çıktılardan tag ve transition count json'larını üretir."""
    # Tag counts
    tag_counts = defaultdict(int)
    with open(tagged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    _, tag = line.split("\t")
                    tag_counts[tag] += 1
                except ValueError:
                    pass
    with open(tag_counts_path, "w", encoding="utf-8") as f:
        json.dump(dict(tag_counts), f, ensure_ascii=False, indent=2)

    # Transition counts
    transition_counts = defaultdict(lambda: defaultdict(int))
    with open(tagged_path, "r", encoding="utf-8") as f:
        current_tags = []
        for line in f:
            line = line.strip()
            if line == "":
                for i in range(1, len(current_tags)):
                    prev_tag = current_tags[i - 1]
                    curr_tag = current_tags[i]
                    transition_counts[prev_tag][curr_tag] += 1
                current_tags = []
            else:
                try:
                    _, tag = line.split("\t")
                    current_tags.append(tag)
                except ValueError:
                    pass
        # Dosya sonunda son cümle
        for i in range(1, len(current_tags)):
            prev_tag = current_tags[i - 1]
            curr_tag = current_tags[i]
            transition_counts[prev_tag][curr_tag] += 1
    transition_counts_dict = {prev: dict(nexts) for prev, nexts in transition_counts.items()}
    with open(transition_counts_path, "w", encoding="utf-8") as f:
        json.dump(transition_counts_dict, f, ensure_ascii=False, indent=2)
