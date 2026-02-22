import { authClient } from "../../lib/auth-client";

export default function SignIn() {
  const signIn = async () => {
    try {
        await authClient.login();
    } catch (err: any) {
        alert(err.message || "Login failed");
    }
  };

  return (
    <div>
      <h2>Sign In</h2>
      <p>Redirecting to OAuth provider...</p>
      <button onClick={signIn}>Sign In with SSO</button>
    </div>
  );
}
