/**
 * The page's own name, matching its tab exactly.
 *
 * Five of the seven views had no heading at all: they opened straight into a KPI
 * strip, so the only thing telling you where you were was the nav. With the tabs
 * renamed, a heading that echoes the tab removes the last of that ambiguity.
 *
 * One h1 per page. Where a view already had one for its content, that content
 * heading becomes an h2 underneath this.
 */
export function PageHead({ title, blurb }: { title: string; blurb: string }) {
  return (
    <header className="border-b border-rule pb-4">
      <h1 className="text-[1.0625rem] font-semibold tracking-[0.02em] text-ink">{title}</h1>
      <p className="prose-sans mt-1 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        {blurb}
      </p>
    </header>
  )
}
