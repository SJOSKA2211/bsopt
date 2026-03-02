import requests

def test_post_graphql_returns_valid_graphql_response():
    base_url = "http://127.0.0.1:8000"
    login_url = f"{base_url}/api/v1/auth/login"
    graphql_url = f"{base_url}/graphql"

    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }

    try:
        login_response = requests.post(login_url, json=login_payload, timeout=30)
    except requests.RequestException as e:
        assert False, f"Login request failed: {e}"

    assert login_response.status_code == 200, f"Login failed with status {login_response.status_code}"

    try:
        login_data = login_response.json()
    except ValueError:
        assert False, "Login response is not valid JSON"

    assert "access_token" in login_data, "Login response missing 'access_token'"
    assert login_data.get("token_type", "") == "bearer", "Login token_type is not 'bearer'"

    access_token = login_data["access_token"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    introspection_query = {
        "query": """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
            directives {
              name
              description
              locations
              args {
                ...InputValue
              }
            }
          }
        }

        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
          inputFields {
            ...InputValue
          }
          interfaces {
            ...TypeRef
          }
          enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
          }
          possibleTypes {
            ...TypeRef
          }
        }

        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }

        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """
    }

    try:
        response = requests.post(graphql_url, json=introspection_query, headers=headers, timeout=30)
    except requests.RequestException as e:
        assert False, f"Request to /graphql failed: {e}"

    assert response.status_code == 200, f"Expected status 200 but got {response.status_code}"

    try:
        json_data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "data" in json_data, "Response JSON missing 'data' field"
    assert "__schema" in json_data["data"], "Response JSON missing '__schema' inside data"


test_post_graphql_returns_valid_graphql_response()
