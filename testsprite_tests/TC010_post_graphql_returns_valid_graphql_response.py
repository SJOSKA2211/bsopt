import requests


def test_post_graphql_returns_valid_graphql_response():
    base_url = "http://127.0.0.1:8000"
    graphql_url = f"{base_url}/graphql"
    timeout = 30

    # Introspection query to validate the GraphQL schema and response structure
    introspection_query = """
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
          }
        }
      }
    }
    """

    payload = {"query": introspection_query}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(graphql_url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        assert False, f"Request to /graphql failed: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    try:
        response_json = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate that 'data' key exists and contains '__schema'
    assert "data" in response_json, "'data' key missing in GraphQL response"
    assert "__schema" in response_json["data"], "'__schema' key missing in GraphQL response data"

    # The response errors key may or may not be present, but introspection query should not have errors
    assert "errors" not in response_json, f"GraphQL errors found: {response_json.get('errors')}"


test_post_graphql_returns_valid_graphql_response()
