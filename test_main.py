from main import is_prime


def test_is_prime_positive():
    """Tests that a prime number returns True."""
    assert is_prime(5)

def test_is_prime_negative():
    """Tests that a non-prime number returns False."""
    assert not is_prime(4)

def test_is_prime_one():
    """Tests that 1 returns False."""
    assert not is_prime(1)

def test_is_prime_zero():
    """Tests that 0 returns False."""
    assert not is_prime(0)

def test_is_prime_negative_number():
    """Tests that a negative number returns False."""
    assert not is_prime(-1)
