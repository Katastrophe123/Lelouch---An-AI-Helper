// A small hand-drawn icon set, 1.6px strokes, so nothing looks borrowed.

const paths: Record<string, string> = {
  plus: "M12 5v14M5 12h14",
  send: "M5 12h13M13 6l6 6-6 6",
  stop: "M7 7h10v10H7z",
  paperclip: "M16 7.5 9.2 14.3a2.2 2.2 0 0 0 3.1 3.1l7-7a4.2 4.2 0 0 0-6-6l-7.2 7.2a6.2 6.2 0 0 0 8.8 8.8L20 15",
  docs: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  menu: "M4 7h16M4 12h16M4 17h16",
  close: "M6 6l12 12M18 6 6 18",
  trash: "M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13",
  copy: "M9 9h11v11H9zM5 15V4h11",
  chevron: "M9 6l6 6-6 6",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  signout: "M10 5H5v14h5M14 8l4 4-4 4M18 12H9",
  check: "M5 12.5 10 17 19 7",
  alert: "M12 8v5M12 16.5v.5M10.3 4.3 2.8 18a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z",
};

export function Icon({ name, size = 18, label }: { name: keyof typeof paths | string; size?: number; label?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d={paths[name] ?? ""} />
    </svg>
  );
}

/** The Lelouch mark: a gold L and an emerald eye, as in the favicon. */
export function Mark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <path d="M20 14h8v30h18v6H20z" fill="var(--gold)" />
      <circle cx="44" cy="22" r="5" fill="var(--emerald)" />
    </svg>
  );
}
