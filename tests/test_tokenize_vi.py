from src.rag.hybrid import tokenize_vi


def test_tokenize_vi_segments_or_falls_back_without_crashing():
    tokens = tokenize_vi("vượt đèn đỏ không chấp hành tín hiệu giao thông")

    assert tokens
    assert any("đèn" in token for token in tokens)
    assert any("giao" in token for token in tokens)


def test_legal_reference_tokens_are_preserved():
    tokens = tokenize_vi("điểm a khoản 3 điều 6")

    assert "điểm_a" in tokens
    assert "khoản_3" in tokens
    assert "điều_6" in tokens
