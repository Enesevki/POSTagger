import stanza

def tag_sentences(input_path, output_path):
    """Tokenli txt dosyasını Stanza ile POS-tagleyip yeni dosya olarak kaydeder."""
    stanza.download("tr")  # Sadece ilk çalıştırmada
    nlp = stanza.Pipeline("tr", processors="tokenize,pos")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    for line in lines:
        line = line.strip()
        if line:
            word = line.split()[0]
            current_sentence.append(word)
        else:
            if current_sentence:
                sentences.append(" ".join(current_sentence))
                current_sentence = []

    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            doc = nlp(sent)
            for sentence in doc.sentences:
                for word in sentence.words:
                    f.write(f"{word.text}\t{word.upos}\n")
            f.write("\n")
