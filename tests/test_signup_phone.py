import importlib


def test_normalize_phone_number_handles_indian_numbers():
    main = importlib.import_module('main')

    assert main.normalize_phone_number('9876543210') == '+919876543210'
    assert main.normalize_phone_number('+91 9876543210') == '+919876543210'
    assert main.normalize_phone_number('987 654 3210') == '+919876543210'
    assert main.normalize_phone_number('') is None
    assert main.normalize_phone_number('abc') is None
