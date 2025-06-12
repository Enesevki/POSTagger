def build_gold_files(tagged_path, sentences_path, gold_path):
    sent_words = []
    sent_tags = []
    with open(tagged_path, "r", encoding="utf-8") as f, \
         open(sentences_path, "w", encoding="utf-8") as f_sent, \
         open(gold_path, "w", encoding="utf-8") as f_gold:
        for line in f:
            line = line.strip()
            if line == "":
                if sent_words:
                    f_sent.write(" ".join(sent_words) + "\n")
                    f_gold.write(" ".join(sent_words) + "\t" + " ".join(sent_tags) + "\n")
                    sent_words = []
                    sent_tags = []
            else:
                try:
                    word, tag = line.split("\t")
                    sent_words.append(word)
                    sent_tags.append(tag)
                except ValueError:
                    pass
        if sent_words:
            f_sent.write(" ".join(sent_words) + "\n")
            f_gold.write(" ".join(sent_words) + "\t" + " ".join(sent_tags) + "\n")
