import { useEffect, useState, type FormEvent } from "react";
import { Api, ApiError } from "../api";
import type { Health, Session } from "../types";
import { Mark } from "./Icon";

export function SignIn({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null | "down">(null);

  useEffect(() => {
    new Api().health().then(setHealth, () => setHealth("down"));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const name = username.trim();
    if (!/^[\w.@+-]+$/.test(name)) {
      setError("Use letters, numbers and . _ @ + - only, with no spaces.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await new Api().signIn(name, password));
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "That password is wrong. It's the AUTH_PASSWORD from your .env file."
          : "Couldn't reach the server. Check that the API is running on port 8000.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="signin">
      <div className="signin-card">
        <Mark size={48} />
        <h1 className="signin-title">Lelouch</h1>
        <p className="signin-lede">Your strategist for research, the web and your own documents.</p>

        <form onSubmit={submit} className="signin-form">
          <label>
            <span>Name</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
              maxLength={80}
            />
          </label>
          <label>
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <button type="submit" className="primary-btn" disabled={busy}>
            {busy ? "Signing in" : "Sign in"}
          </button>
        </form>

        <p className="signin-status" data-ok={health !== "down" && health !== null}>
          {health === null && "Checking the server"}
          {health === "down" && "The server isn't answering on port 8000"}
          {health && health !== "down" && `Server ready, version ${health.version}`}
        </p>
      </div>
    </main>
  );
}
