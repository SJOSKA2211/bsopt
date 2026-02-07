from src.pricing.lattice import (
    BinomialTreePricer,
    BSParameters,
    TrinomialTreePricer,
    validate_convergence,
)


def test_binomial_zero_maturity():
    pricer = BinomialTreePricer()
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    assert pricer.price(params, "call") == 10.0
    assert pricer.price(params, "put") == 0.0

def test_trinomial_zero_maturity():
    pricer = TrinomialTreePricer()
    params = BSParameters(110, 100, 0.0, 0.2, 0.05)
    assert pricer.price(params, "call") == 10.0
    assert pricer.price(params, "put") == 0.0

def test_binomial_american_put():
    pricer = BinomialTreePricer(exercise_type="american")
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    price = pricer.price(params, "put")
    assert price > 0

def test_trinomial_american_call_dividend():
    # American call with high dividend might be exercised early
    pricer = TrinomialTreePricer(exercise_type="american")
    params = BSParameters(100, 100, 1.0, 0.2, 0.05, 0.1) # 10% dividend
    price = pricer.price(params, "call")
    assert price > 0

def test_trinomial_american_put():
    pricer = TrinomialTreePricer(exercise_type="american")
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    price = pricer.price(params, "put")
    assert price > 0

def test_binomial_american_call():
    pricer = BinomialTreePricer(exercise_type="american")
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    price = pricer.price(params, "call")
    assert price > 0

def test_validate_convergence():
    res = validate_convergence(100, 100, 1.0, 0.2, 0.05, 0.0, "call", [10, 20])
    assert "binomial_errors" in res
    assert len(res["binomial_errors"]) == 2
    
    res_put = validate_convergence(100, 100, 1.0, 0.2, 0.05, 0.0, "put", [10])
    assert len(res_put["trinomial_errors"]) == 1

def test_binomial_greeks_short_maturity():
    # Hit theta = 0.0 branch
    pricer = BinomialTreePricer()
    params = BSParameters(100, 100, 0.001, 0.2, 0.05)
    greeks = pricer.calculate_greeks(params, "call")
    assert greeks.theta == 0.0

def test_trinomial_greeks_full():
    pricer = TrinomialTreePricer()
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    greeks = pricer.calculate_greeks(params, "call")
    assert greeks.delta > 0
    assert greeks.rho > 0
    
    # Short maturity for trinomial theta
    params_short = BSParameters(100, 100, 0.001, 0.2, 0.05)
    greeks_short = pricer.calculate_greeks(params_short, "call")
    assert greeks_short.theta == 0.0

def test_binomial_build_tree():
    pricer = BinomialTreePricer(n_steps=2)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    tree = pricer.build_tree(params)
    assert tree.shape == (3, 3)
    assert tree[0, 0] == 100

def test_binomial_build_tree_zero_steps():
    pricer = BinomialTreePricer(n_steps=0)
    params = BSParameters(100, 100, 1.0, 0.2, 0.05)
    tree = pricer.build_tree(params)
    assert tree.shape == (1, 1)
    assert tree[0, 0] == 100

def test_binomial_arbitrage_violation():
    # If p < 0 or p > 1
    # dt = T / n_steps. u = exp(sigma*sqrt(dt)). a = exp((r-q)dt).
    # p = (a-d)/(u-d).
    # If r is very large and sigma is small, a might exceed u.
    pricer = BinomialTreePricer(n_steps=100)
    params = BSParameters(100, 100, 1.0, 0.01, 0.5) # small vol, high rate
    price = pricer.price(params, "call")
    assert price > 0
