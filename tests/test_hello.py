from steering_geometry.hello import greet


def test_greet_returns_greeting() -> None:
    assert greet("World") == "Hello, World!"


def test_greet_with_empty_string() -> None:
    assert greet("") == "Hello, !"
