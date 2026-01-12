package authz

default allow = false

# Allow admin users to perform any action
allow {
    input.user.role == "admin"
}

# Allow traders to view market data and trade
allow {
    input.user.role == "trader"
    input.action == "read"
    input.resource == "market_data"
}

allow {
    input.user.role == "trader"
    input.action == "execute"
    input.resource == "trade"
}

# Allow quants to read all research data
allow {
    input.user.role == "quant"
    input.action == "read"
}

# Deny everything else by default
deny {
    not allow
}
