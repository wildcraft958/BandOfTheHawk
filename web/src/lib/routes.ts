/**
 * The seven views, in the order the nav shows them.
 *
 * Ordered to the way the submission is judged, not to the way the pages were
 * built: identify the attack surface, generate a world faithful enough to test
 * against, defend it, close the loop, then show it surviving an authorisation
 * path.
 *
 * Labels name what a page holds. Three earlier ones did not: "Dashboard" said
 * nothing, "Simulator" implied you could run something there when the one
 * playable page was called "Demo", and "Live" implied production data rather
 * than a generated stream.
 *
 * Paths match these labels. The paths they replaced redirect in main.tsx, so
 * anything already shared still resolves.
 */
export const VIEWS = [
  { to: '/', label: 'Overview' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/attack-surface', label: 'Attack surface' },
  { to: '/fidelity', label: 'Fidelity' },
  { to: '/detection', label: 'Detection' },
  { to: '/co-evolution', label: 'Co-evolution' },
  { to: '/auth-stream', label: 'Auth stream' },
] as const
