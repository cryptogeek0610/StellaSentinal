import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Layout } from './components/Layout'
import { MockModeProvider } from './hooks/useMockMode'
import { UserRoleProvider } from './hooks/useUserRole'
import ErrorBoundary from './components/ErrorBoundary'
import { LoadingPage } from './components/ui/Loading'

// Eagerly loaded (home page)
import ActionCenter from './pages/ActionCenter'

// Lazy-loaded pages — only fetched when the route is visited
const UnifiedDashboard = lazy(() => import('./pages/UnifiedDashboard'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Insights = lazy(() => import('./pages/Insights'))
const CostManagement = lazy(() => import('./pages/CostManagement'))
const Investigations = lazy(() => import('./pages/Investigations'))
const InvestigationDetail = lazy(() => import('./pages/InvestigationDetail'))
const DeviceDetail = lazy(() => import('./pages/DeviceDetail'))
const Fleet = lazy(() => import('./pages/Fleet'))
const LocationCenter = lazy(() => import('./pages/LocationCenter'))
const TrainingMonitor = lazy(() => import('./pages/TrainingMonitor'))
const Baselines = lazy(() => import('./pages/Baselines'))
const Automation = lazy(() => import('./pages/Automation'))
const ModelStatus = lazy(() => import('./pages/ModelStatus'))
const System = lazy(() => import('./pages/System'))
const DataOverview = lazy(() => import('./pages/DataOverview'))
const NetworkIntelligence = lazy(() => import('./pages/NetworkIntelligence'))
const SecurityPosture = lazy(() => import('./pages/SecurityPosture'))
const NotFound = lazy(() => import('./pages/NotFound'))

function LegacyAnomalyRedirect() {
  const { id } = useParams()
  const target = id ? `/investigations/${id}` : '/investigations'
  return <Navigate to={target} replace />
}

function App() {
  return (
    <ErrorBoundary>
      <UserRoleProvider>
        <MockModeProvider>
          <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#1e293b',
                color: '#f1f5f9',
                border: '1px solid #334155',
              },
              success: {
                iconTheme: {
                  primary: '#10b981',
                  secondary: '#f1f5f9',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#f1f5f9',
                },
              },
            }}
          />
          <Layout>
            <Suspense fallback={<LoadingPage />}>
              <Routes>
                {/* Action Center - The One Screen That Matters */}
                <Route path="/" element={<ActionCenter />} />
                <Route path="/action-center" element={<ActionCenter />} />

                {/* Command Center - Observability Dashboard */}
                <Route path="/dashboard" element={<UnifiedDashboard />} />

                {/* Investigations */}
                <Route path="/investigations" element={<Investigations />} />
                <Route path="/investigations/:id" element={<InvestigationDetail />} />

                {/* Device Detail (from investigation drill-down) */}
                <Route path="/devices/:id" element={<DeviceDetail />} />

                {/* System - Consolidated admin (connections, ML pipeline, config) */}
                <Route path="/system" element={<System />} />

                {/* Debug route for ML engineers (hidden from nav) */}
                <Route path="/debug/data" element={<DataOverview />} />

                {/* Restored dashboards */}
                <Route path="/noc" element={<Dashboard />} />
                <Route path="/insights" element={<Insights />} />
                <Route path="/costs" element={<CostManagement />} />

                {/* Fleet & Location Intelligence */}
                <Route path="/fleet" element={<Fleet />} />
                <Route path="/locations" element={<LocationCenter />} />

                {/* Network & Security Intelligence */}
                <Route path="/network" element={<NetworkIntelligence />} />
                <Route path="/security" element={<SecurityPosture />} />

                {/* ML Operations */}
                <Route path="/training" element={<TrainingMonitor />} />
                <Route path="/baselines" element={<Baselines />} />
                <Route path="/automation" element={<Automation />} />
                <Route path="/model-status" element={<ModelStatus />} />

                {/* Legacy redirects */}
                <Route path="/anomalies" element={<Navigate to="/investigations" replace />} />
                <Route path="/anomalies/:id" element={<LegacyAnomalyRedirect />} />
                <Route path="/dashboard/detailed" element={<Navigate to="/dashboard" replace />} />
                <Route path="/data" element={<Navigate to="/debug/data" replace />} />
                <Route path="/status" element={<Navigate to="/model-status" replace />} />
                <Route path="/setup" element={<Navigate to="/system" replace />} />
                <Route path="/settings" element={<Navigate to="/system" replace />} />

                {/* 404 - Catch all unmatched routes */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </Layout>
          </BrowserRouter>
        </MockModeProvider>
      </UserRoleProvider>
    </ErrorBoundary>
  )
}

export default App
