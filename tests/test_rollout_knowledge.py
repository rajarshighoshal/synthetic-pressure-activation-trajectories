from experiments.rollout_sycon import classify_knowledge


def test_classify_knowledge_accepts_one_word_answers():
    assert classify_knowledge("False") == "false"
    assert classify_knowledge("True.") == "true"
    assert classify_knowledge("Unsure") == "unsure"


def test_classify_knowledge_accepts_short_sentence_answers():
    assert classify_knowledge("The statement is false.") == "false"
    assert classify_knowledge("This is not true.") == "false"
    assert classify_knowledge("That claim is incorrect.") == "false"
    assert classify_knowledge("The statement is true.") == "true"
    assert classify_knowledge("I cannot determine this from the statement.") == "unsure"
