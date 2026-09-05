import { useEffect, useState } from 'react'

/**
 * Motion preference, read once and kept in sync.
 *
 * CSS handles transitions on its own, but the trace animations and the
 * simulation ticker are driven from JS and have to be gated here too. Honouring
 * the preference only in CSS is the usual miss.
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
  )

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(query.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  return reduced
}
