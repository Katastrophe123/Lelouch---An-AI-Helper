import { useState } from "react";
import { loadSession, saveSession } from "./api";
import { SignIn } from "./components/SignIn";
import { Workspace } from "./components/Workspace";
import type { Session } from "./types";

export function App() {
  const [session, setSession] = useState<Session | null>(loadSession);

  const update = (next: Session | null) => {
    saveSession(next);
    setSession(next);
  };

  return session ? (
    <Workspace key={session.access_token} session={session} onSignOut={() => update(null)} />
  ) : (
    <SignIn onSignedIn={update} />
  );
}
