import { createRoot } from 'react-dom/client'
import { Navigate, createHashRouter, RouterProvider } from 'react-router-dom'
import { Chrome } from './app/Chrome'
import { Landing } from './screens/Landing'
import { Architecture } from './screens/Architecture'
import { Dashboard } from './screens/Dashboard'
import { Simulator } from './screens/Simulator'
import { Loop } from './screens/Loop'
import { Live } from './screens/Live'
import { Demo } from './screens/Demo'
import './index.css'

// Hash routing, not browser routing: the built single file has to work when a
// judge opens it straight off disk, and history routing breaks under file://.
//
// Paths match the nav labels, and every path they replaced still resolves. The
// submission link is the bare origin, which carries no hash and lands on '/',
// so it is unaffected either way; these redirects exist for anything already
// shared or bookmarked deeper than that. The catch-all matters independently of
// the rename: without it an unknown hash rendered a blank page.
const LEGACY: Record<string, string> = {
  '/demo': '/attack-surface',
  '/simulator': '/fidelity',
  '/dashboard': '/detection',
  '/loop': '/co-evolution',
  '/live': '/auth-stream',
}

const router = createHashRouter([
  {
    element: <Chrome />,
    children: [
      { path: '/', element: <Landing /> },
      { path: '/architecture', element: <Architecture /> },
      { path: '/attack-surface', element: <Demo /> },
      { path: '/fidelity', element: <Simulator /> },
      { path: '/detection', element: <Dashboard /> },
      { path: '/co-evolution', element: <Loop /> },
      { path: '/auth-stream', element: <Live /> },

      ...Object.entries(LEGACY).map(([from, to]) => ({
        path: from,
        element: <Navigate to={to} replace />,
      })),

      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <RouterProvider router={router} future={{ v7_startTransition: true }} />,
)
