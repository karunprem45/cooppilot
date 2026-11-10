from app.documents import clean_text, extract_text


def test_plain_text_upload():
    assert extract_text(b"Python   and\n\n\n\nSQL", "jd.txt") == "Python and\n\nSQL"


def test_clean_strips_nulls():
    assert clean_text("a\x00b") == "a b"
