import yaml

from scripts.sentence_tokenizer import tokenize_sentences
from scripts.stanza_tagger import tag_sentences
from scripts.build_tag_stats import build_tag_stats
from scripts.build_probs import build_transition_probs, build_emission_probs
from scripts.build_gold import build_gold_files
from scripts.viterbi import run_viterbi
from scripts.evaluate import evaluate_accuracy

def main(config):
    print(">>> [1] Cümleleri tokenize et")
    tokenize_sentences(config["raw_excel_path"], config["processed_tokenized_path"])
    
    print(">>> [2] Stanza ile POS tagle")
    tag_sentences(config["processed_tokenized_path"], config["processed_tagged_path"])
    
    print(">>> [3] Tag ve transition count'ları oluştur")
    build_tag_stats(
        config["processed_tagged_path"],
        config["tag_counts_path"],
        config["transition_counts_path"]
    )

    print(">>> [4] Transition ve emission olasılıklarını oluştur")
    build_transition_probs(
        config["transition_counts_path"],
        config["tag_counts_path"],
        config["transition_probs_path"]
    )
    build_emission_probs(
        config["processed_tagged_path"],
        config["emission_counts_path"],
        config["emission_probs_path"],
        config["tag_counts_path"]
    )

    print(">>> [5] Gold (referans) dosyalarını hazırla")
    build_gold_files(
        config["processed_tagged_path"],
        config["sentences_path"],
        config["gold_path"]
    )

    print(">>> [6] Viterbi ile cümleleri etiketle")
    run_viterbi(
        config["sentences_path"],
        config["viterbi_output_path"],
        config["transition_probs_path"],
        config["emission_probs_path"],
        config["tag_counts_path"]
    )

    print(">>> [7] Doğruluk (accuracy) hesapla")
    evaluate_accuracy(
        config["viterbi_output_path"],
        config["gold_path"]
    )

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    main(config)
