const GROUPED = new Intl.NumberFormat('en-US')

export const int = (n: number | null | undefined): string =>
  n == null ? '—' : GROUPED.format(Math.round(n))

export const fixed = (n: number | null | undefined, places: number): string =>
  n == null ? '—' : n.toFixed(places)

export const pct = (n: number | null | undefined, places = 1): string =>
  n == null ? '—' : `${(n * 100).toFixed(places)}%`

/** Minutes and seconds, for stage runtimes that span 0.4s to 3321s. */
export const duration = (seconds: number | null | undefined): string => {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const mins = Math.floor(seconds / 60)
  return `${mins}m ${Math.round(seconds - mins * 60)}s`
}
