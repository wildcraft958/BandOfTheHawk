import { createRoot } from 'react-dom/client'
import { createHashRouter, RouterProvider } from 'react-router-dom'
import { Chrome } from './app/Chrome'
import { Landing } from './screens/Landing'
import { Dashboard } from './screens/Dashboard'
import { Demo } from './screens/Demo'
import './index.css'

// Hash routing, not browser routing: the built single file has to work when a
// judge opens it straight off disk, and history routing breaks under file://.
const router = createHashRouter([
  {
    element: <Chrome />,
    children: [
      { path: '/', element: <Landing /> },
      { path: '/dashboard', element: <Dashboard /> },
      { path: '/demo', element: <Demo /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(<RouterProvider router={router} />)
