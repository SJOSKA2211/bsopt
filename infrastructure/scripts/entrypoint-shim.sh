#!/bin/sh
set -e

# RSA Decryption Entrypoint Shim
# Decrypts environment variables prefixed with ENC_

VAULT_KEY="/app/vault.key"

# Ensure /opt/venv/bin is in PATH
export PATH="/opt/venv/bin:$PATH"

# Activate virtual environment if it exists
if [ -d "/opt/venv" ]; then
    . /opt/venv/bin/activate
fi

if [ ! -f "$VAULT_KEY" ]; then
    echo "Error: Vault key not found at $VAULT_KEY" >&2
    exit 1
fi

# Iterate through environment variables starting with ENC_
# We use env to get all variables and grep for those starting with ENC_
for var_name in $(env | grep "^ENC_" | cut -d= -f1); do
    # Validate variable name to prevent shell injection
    case "$var_name" in
        *[!a-zA-Z0-9_]*) 
            echo "Warning: Skipping invalid environment variable name: $var_name" >&2
            continue 
            ;;
    esac

    # Extract the original variable name (remove ENC_ prefix)
    original_name="${var_name#ENC_}"
    
    # Get the value of the ENC_ variable
    eval "encrypted_value=\$$var_name"
    
    if [ -z "$encrypted_value" ]; then
        echo "Warning: $var_name is empty, skipping." >&2
        continue
    fi

    # Decrypt the value
    # 1. Base64 decode
    # 2. Decrypt with openssl pkeyutl
    decrypted_value=$(printf "%s" "$encrypted_value" | base64 -d | openssl pkeyutl -decrypt -inkey "$VAULT_KEY" 2>/dev/null)
    
    # Check if decryption was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to decrypt environment variable $var_name" >&2
        exit 1
    fi
    
    # Export the decrypted value
    export "$original_name=$decrypted_value"
    
    # Unset the ENC_ variable for safety
    unset "$var_name"
done

# Hand off execution to the original container command
exec "$@"
