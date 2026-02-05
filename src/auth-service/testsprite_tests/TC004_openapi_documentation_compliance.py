import requests
from jsonschema import ValidationError, validate


def test_openapi_documentation_compliance():
    base_url = "http://localhost:4000"
    timeout = 30

    # OpenAPI 3.0 schema (simplified but includes required fields for validation)
    openapi_schema = {
        "type": "object",
        "required": ["openapi", "info", "paths"],
        "properties": {
            "openapi": {"type": "string", "pattern": "^3\\.0\\..*"},
            "info": {
                "type": "object",
                "required": ["title", "version"],
                "properties": {
                    "title": {"type": "string"},
                    "version": {"type": "string"}
                },
                "additionalProperties": True
            },
            "paths": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "object"
                }
            }
        },
        "additionalProperties": True
    }

    # Retrieve OpenAPI docs for root '/'
    resp_root = requests.get(f"{base_url}/openapi.json", timeout=timeout)
    assert  # nosec B101 resp_root.status_code == 200, f"Failed to get OpenAPI doc from /openapi.json: {resp_root.status_code}"

    openapi_root = resp_root.json()

    # Validate OpenAPI spec for root '/'
    try:
        validate(instance=openapi_root, schema=openapi_schema)
    except ValidationError as ve:
        assert  # nosec B101 False, f"OpenAPI document at /openapi.json failed schema validation: {ve}"

    # Check that '/' path exists and defines GET operation with 200 response
    paths = openapi_root.get("paths", {})
    assert  # nosec B101 "/" in paths, "OpenAPI doc missing '/' path"
    get_op = paths["/"].get("get")
    assert  # nosec B101 get_op is not None, "OpenAPI doc '/' path missing GET operation"
    responses = get_op.get("responses", {})
    assert  # nosec B101 "200" in responses, "OpenAPI doc '/' GET operation missing 200 response"

    # Retrieve OpenAPI docs for /api/auth/*
    # Assumption: the service exposes a combined OpenAPI doc at /openapi.json that should cover /api/auth/* as well,
    # so verify presence of /api/auth/* or /api/auth/{some parameter} path.
    auth_path = None
    # Try to find demonstration of the /api/auth/* path or a pattern matching it
    for p in paths:
        if p.startswith("/api/auth"):
            auth_path = p
            break
    assert  # nosec B101 auth_path is not None, "OpenAPI doc missing '/api/auth/*' endpoint"

    auth_path_item = paths[auth_path]
    # Since the PRD shows an "all" method, but OpenAPI 3.0 does not support "all" method natively,
    # it might be implemented as multiple methods or a vendor extension; here we accept any method present.
    # Check that at least one HTTP method has 200 response
    methods = ["get", "post", "put", "delete", "patch", "options", "head", "trace"]
    found_200 = False
    for method in methods:
        if method in auth_path_item:
            resp_codes = auth_path_item[method].get("responses", {})
            if "200" in resp_codes:
                found_200 = True
                break
    assert  # nosec B101 found_200, f"OpenAPI doc for '{auth_path}' endpoint missing 200 response in any HTTP method"

test_openapi_documentation_compliance()