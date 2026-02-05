import sys

import requests


def verify_oauth_triad():
    print("🥒 BSOpt OAuth 2.0 Triad Verification")
    print("-------------------------------------")
    
    base_url = "http://localhost:8000"
    discovery_url = f"{base_url}/auth/.well-known/openid-configuration"
    
    try:
        print(f"Testing OIDC Discovery: {discovery_url}")
        res = requests.get(discovery_url)
        if res.status_code == 200:
            config = res.json()
            print("✅ Discovery Successful!")
            print(f"   Issuer: {config.get('issuer')}")
            print(f"   JWKS URI: {config.get('jwks_uri')}")
            
            jwks_uri = config.get('jwks_uri')
            print(f"Testing JWKS Connectivity: {jwks_uri}")
            res_jwks = requests.get(jwks_uri)
            if res_jwks.status_code == 200:
                print("✅ JWKS Reachable!")
                print(f"   Keys Found: {len(res_jwks.json().get('keys', []))}")
            else:
                print(f"❌ JWKS Failed: {res_jwks.status_code}")
        else:
            print(f"❌ Discovery Failed: {res.status_code}")
            print("   Note: Ensure the API server is running (python3 bs_cli.py serve)")
            
    except Exception as e:
        print(f"❌ Verification Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_oauth_triad()
