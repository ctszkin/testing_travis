import pytest
from src.hello import hello
import sys

def test_1():
    assert 1 == 1


def test_hello(capsys):
    hello(1)
    captured = capsys.readouterr()
    assert captured.out == "Hello World!\n"

def test_hello2(capsys):
    hello(0)
    captured = capsys.readouterr()
    assert captured.out == "a\n"
