import { createRoot } from 'react-dom/client'
import { createHashRouter, RouterProvider } from 'react-router-dom'
import { Chrome } from './app/Chrome'
import { Landing } from './screens/Landing'
import { Dashboard } from './screens/Dashboard'
import { Simulator } from './screens/Simulator'
import { Loop } from './screens/Loop'
import { Live } from './screens/Live'
import { Demo } from './screens/Demo'
import './index.css'

// Hash routing, not browser routing: the built single file has to work when a
// judge opens it straight off disk, and history routing breaks under file://.
const router = createHashRouter(
  [
    {
      element: <Chrome />,
      children: [
        { path: '/', element: <Landing /> },
        { path: '/dashboard', element: <Dashboard /> },
        { path: '/simulator', element: <Simulator /> },
        { path: '/loop', element: <Loop /> },
        { path: '/live', element: <Live /> },
        { path: '/demo', element: <Demo /> },
      ],
    },
  ],
)

createRoot(document.getElementById('root')!).render(<RouterProvider router={router} future={{ v7_startTransition: true }} />)
