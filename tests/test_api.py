import numpy as np

def mult(a,b):
    return a*b

def pow(a,b):
    return a**b

def add(a,b):
    return a+b

def test_mult():
    assert mult(3,2) == 6

def test_pow():
    assert pow(3,4) == 81

def test_add():
    assert add(12,13) == 25


