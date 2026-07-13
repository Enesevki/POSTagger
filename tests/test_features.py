from src.features import get_word_shape, sent2features, sent2labels, sent2tokens


def test_word_shape_compresses_character_classes():
    assert get_word_shape("Ankara123") == "Xxd"


def test_sentence_helpers_preserve_tokens_and_labels():
    sentence = [("Ankara", "NOUN"), ("güzel", "ADJ")]

    assert sent2tokens(sentence) == ["Ankara", "güzel"]
    assert sent2labels(sentence) == ["NOUN", "ADJ"]

    features = sent2features(sentence)
    assert features[0]["BOS"] is True
    assert features[-1]["EOS"] is True
    assert features[0]["+1:word.lower"] == "güzel"

