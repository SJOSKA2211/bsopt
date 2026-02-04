/**
 * High-performance OAuth2 Client for BSOPT Singularity.
 * Implements Authorization Code flow with PKCE.
 */

export class OAuth2Client {
    private baseURL: string;
    private clientId: string;
    private redirectUri: string;

    constructor(baseURL: string, clientId: string, redirectUri: string) {
        this.baseURL = baseURL;
        this.clientId = clientId;
        this.redirectUri = redirectUri;
    }

    /**
     * Start the login flow by redirecting to the Auth Server.
     */
    async login() {
        const state = this.generateRandomString(32);
        const codeVerifier = this.generateRandomString(64);
        const codeChallenge = await this.generateCodeChallenge(codeVerifier);

        localStorage.setItem("oauth_state", state);
        localStorage.setItem("oauth_code_verifier", codeVerifier);

        const params = new URLSearchParams({
            response_type: "code",
            client_id: this.clientId,
            redirect_uri: this.redirectUri,
            state: state,
            code_challenge: codeChallenge,
            code_challenge_method: "S256",
            scope: "openid profile email"
        });

        window.location.href = `${this.baseURL}/auth/authorize?${params.toString()}`;
    }

    /**
     * Exchange the authorization code for tokens.
     */
    async handleCallback(code: string, state: string) {
        const savedState = localStorage.getItem("oauth_state");
        const codeVerifier = localStorage.getItem("oauth_code_verifier");

        if (state !== savedState) {
            throw new Error("Invalid state");
        }

        const response = await fetch(`${this.baseURL}/auth/token`, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({
                grant_type: "authorization_code",
                code: code,
                client_id: this.clientId,
                redirect_uri: this.redirectUri,
                code_verifier: codeVerifier || ""
            })
        });

        const data = await response.json();
        if (data.access_token) {
            localStorage.setItem("access_token", data.access_token);
            if (data.refresh_token) {
                localStorage.setItem("refresh_token", data.refresh_token);
            }
        }
        return data;
    }

    private generateRandomString(length: number): string {
        const array = new Uint8Array(length);
        window.crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }

    private async generateCodeChallenge(verifier: string): Promise<string> {
        const encoder = new TextEncoder();
        const data = encoder.encode(verifier);
        const digest = await window.crypto.subtle.digest("SHA-256", data);
        return btoa(String.fromCharCode(...new Uint8Array(digest)))
            .replace(/\+/g, "-")
            .replace(/\//g, "_")
            .replace(/=+$/, "");
    }
}

export const authClient = new OAuth2Client(
    "http://localhost:8000", // API/Auth Base
    "bsopt-frontend",
    `${window.location.origin}/callback`
);