import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_post_graphql_returns_valid_graphql_response():
    url = f"{BASE_URL}/graphql"
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
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=introspection_query, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        assert False, f"Request to {url} failed: {e}"

    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate that the response contains 'data' with '__schema'
    assert "data" in data, "'data' field not in GraphQL response"
    assert "__schema" in data["data"], "'__schema' field not in GraphQL response data"

    # Basic sanity checks for schema content
    schema = data["data"]["__schema"]
    assert "queryType" in schema and schema["queryType"] is not None, "'queryType' missing or null in schema"
    # mutationType and subscriptionType can be None depending on schema, so no strict assertion here

test_post_graphql_returns_valid_graphql_response()