import { NavLink, Outlet } from 'react-router-dom'
import { VIEWS } from '../lib/routes'
import { cn } from '../components/ui/cn'
import { meta } from '../data/run'
import { fixtureFiles } from '../data/run'

/**
 * Console chrome. Top nav plus the run this prototype reads from, since every
 * number on the page comes from that one recorded run rather than from a live
 * pipeline.
 */
export function Chrome() {
  return (
    <>
      {fixtureFiles.length > 0 && (
        <div
          role="alert"
          className="border-b border-atk bg-atk/12 px-4 py-2 text-center text-[0.8125rem] uppercase tracking-[0.1em] text-atk"
        >
          Demo fixture. Not a measured result ({fixtureFiles.join(', ')})
        </div>
      )}

      <header className="sticky top-0 z-40 border-b border-rule bg-surface/95 backdrop-blur">
        <div className="flex h-14 items-center gap-6 px-4 sm:px-6">
          <NavLink to="/" className="flex shrink-0 items-center gap-2.5 no-underline">
            <span aria-hidden="true" className="text-atk">&#9646;&#9647;</span>
            <span className="text-[1rem] font-semibold tracking-[0.08em] text-ink">
              GAUNTLET
            </span>
            <span aria-hidden="true" className="text-atk">&#9647;&#9646;</span>
          </NavLink>

          {/* The nav scrolls rather than the page. Seven labels at a legible size
              are wider than a narrow viewport, and letting them push the
              document gives every page a sideways scrollbar. */}
          <nav
            aria-label="Primary"
            className="min-w-0 flex-1 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          >
            <ul className="flex items-center gap-5 whitespace-nowrap">
              {VIEWS.map((r) => (
                <li key={r.to}>
                  <NavLink
                    to={r.to}
                    end={r.to === '/'}
                    className={({ isActive }) =>
                      cn(
                        'block border-b-2 py-[1.1rem] text-[0.8125rem] uppercase tracking-[0.12em] no-underline transition-colors duration-150',
                        isActive
                          ? 'border-atk text-atk'
                          : 'border-transparent text-ink-2 hover:text-ink',
                      )
                    }
                  >
                    {r.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          <span className="hidden shrink-0 items-center gap-1.5 sm:flex">
            <span className="status-dot size-1.5" aria-hidden="true" />
            <span className="text-[0.75rem] uppercase tracking-[0.1em] text-pass">replay</span>
            <span className="hidden text-[0.75rem] uppercase tracking-[0.08em] text-ink-3 lg:inline">
              run {meta.started?.slice(0, 10)} &middot; profile {meta.profile}
            </span>
          </span>
        </div>
      </header>

      <main id="main">
        <Outlet />
      </main>

      <footer className="border-t border-rule px-4 py-8 text-[0.8125rem] text-ink-3 sm:px-6">
        <p>
          Band of the Hawk &middot; IIT Kharagpur &middot; Shehryaar Shah Khan, Saksham Tiwari,
          Animesh Raj, Eisa Shaiju, Monika Kumari
        </p>
        <p className="mt-2 max-w-3xl">
          Every entity, amount, device and text artifact is synthetic. Calibration uses aggregate
          distributional statistics, never records. No cardholder data, no PII, and no production
          payment data at any point.
        </p>
      </footer>
    </>
  )
}
