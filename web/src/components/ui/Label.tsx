import type { ReactNode } from 'react'

/** Small uppercase caption that names the group beneath it. */
export function Label({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-ink-3">
      {children}
    </div>
  )
}
