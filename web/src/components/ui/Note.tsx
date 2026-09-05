import type { ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { cn } from './cn'

/**
 * A collapsed explanation under a panel's content.
 *
 * Every figure on this site carries its own reasoning, its caveat, or its
 * arithmetic. Printed in full that is a wall of prose under every chart, and the
 * charts stop being read. Collapsed, a reader who wants the reasoning can ask
 * for it and everyone else gets the figure.
 *
 * Built on <details> rather than React state on purpose: it is keyboard operable
 * and toggleable with no script, it survives a print to PDF with the content
 * still in the document, and a browser find-in-page can reach the text inside a
 * closed one in current Chrome. A judge searching the page for a caveat will
 * still land on it.
 *
 * `lede` is for the case where the hidden text is not an explanation but a
 * disclosure: that a number is a stand-in, or comes from a different run. One
 * line of that stays on screen whatever the reader does, because a caveat that
 * only appears when someone thinks to click is not a caveat.
 */
export function Note({
  label,
  lede,
  children,
  className,
}: {
  /** What is inside, in a few words. It is the only thing shown when closed. */
  label: string
  /** Optional single line that stays visible whether the note is open or not. */
  lede?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn('mt-4 border-t border-rule pt-3', className)}>
      {lede ? (
        <p className="prose-sans mb-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          {lede}
        </p>
      ) : null}
      <details>
        <summary
          className={cn(
            'flex w-fit cursor-pointer list-none items-center gap-1.5 rounded-[3px] py-0.5',
            'text-[0.8125rem] uppercase tracking-[0.09em] text-ink-3',
            'transition-colors duration-150 hover:text-ink-2',
            'focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-3',
            '[&::-webkit-details-marker]:hidden',
          )}
        >
          <ChevronRight
            className="note-chevron size-3 shrink-0"
            aria-hidden="true"
          />
          {label}
        </summary>
        <div className="mt-2.5">{children}</div>
      </details>
    </div>
  )
}
