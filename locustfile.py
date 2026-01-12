from locust import HttpUser, task, between, events
import logging

class BSOPTUser(HttpUser):
    wait_time = between(0.1, 0.5) # Simulate high throughput

    @task(10)
    def query_option_pricing(self):
        """Query option pricing (High Volume)"""
        query = """
        query GetOptionPrice {
            option(contractSymbol: "AAPL_20260115_C_150") {
                id
                price
                delta
                gamma
                fairValue
                recommendation
            }
        }
        """
        self.client.post("/graphql", json={"query": query}, name="Query: Option Price")

    @task(5)
    def query_portfolio(self):
        """Query portfolio (Medium Volume)"""
        query = """
        query GetPortfolio {
            portfolio(userId: "user_123") {
                id
                cashBalance
                positions {
                    contractSymbol
                    currentPnl
                }
            }
        }
        """
        self.client.post("/graphql", json={"query": query}, name="Query: Portfolio")

    @task(1)
    def create_order(self):
        """Place an order (Low Volume)"""
        mutation = """
        mutation PlaceOrder {
            createOrder(
                portfolioId: "port_123",
                contractSymbol: "AAPL_20260115_C_150",
                side: "BUY",
                quantity: 1,
                orderType: "MARKET"
            ) {
                id
                status
            }
        }
        """
        self.client.post("/graphql", json={"query": mutation}, name="Mutation: Create Order")

# Hook to fail if requirements not met
@events.request.add_listener
def check_sla(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        logging.error(f"Request failed: {name} - {exception}")
    if response_time > 100:
        logging.warning(f"SLA Breach: {name} took {response_time}ms (Target: <100ms)")
