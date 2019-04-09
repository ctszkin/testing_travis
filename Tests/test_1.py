import pytest
from src.hello import hello
import sys

def test_1():
    assert 1 == 1


def test_hello(capsys):
    hello()
    captured = capsys.readouterr()
    assert captured.out == "Hello World!\n"
