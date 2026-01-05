import { BrowserRouter, Routes, Route, Navigate, useParams } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import UnifiedDashboard from './pages/UnifiedDashboard'
import Dashboard from './pages/Dashboard'
import Investigations from './pages/Investigations'
import InvestigationDetail from './pages/InvestigationDetail'
import Fleet from './pages/Fleet'
import DeviceDetail from './pages/DeviceDetail'
import Insights from './pages/Insights'
import LocationCenter from './pages/LocationCenter'
import System from './pages/System'
import ModelStatus from './pages/ModelStatus'
import DataOverview from './pages/DataOverview'
import TrainingMonitor from './pages/TrainingMonitor'
import Baselines from './pages/Baselines'
import Automation from './pages/Automation'
import CostManagement from './pages/CostManagement'
import Setup from './pages/Setup'
import NotFound from './pages/NotFound'
import { Layout } from './components/Layout'
import { MockModeProvider } from './hooks/useMockMode'
import { UserRoleProvider } from './hooks/useUserRole'
import ErrorBoundary from './components/ErrorBoundary'

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
          <BrowserRouter>
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
            <Routes>
              {/* Unified Command Center - Primary Entry Point */}
              <Route path="/" element={<UnifiedDashboard />} />
              <Route path="/dashboard" element={<UnifiedDashboard />} />

              {/* Detailed Operations Dashboard (formerly Command Center) */}
              <Route path="/dashboard/detailed" element={<Dashboard />} />

              {/* Investigations */}
              <Route path="/investigations" element={<Investigations />} />
              <Route path="/investigations/:id" element={<InvestigationDetail />} />

              {/* Legacy routes - redirect to new structure */}
              <Route path="/anomalies" element={<Navigate to="/investigations" replace />} />
              <Route path="/anomalies/:id" element={<LegacyAnomalyRedirect />} />

              {/* Fleet */}
              <Route path="/fleet" element={<Fleet />} />
              <Route path="/devices/:id" element={<DeviceDetail />} />

              {/* Insights */}
              <Route path="/insights" element={<Insights />} />

              {/* Location Center */}
              <Route path="/locations" element={<LocationCenter />} />

              {/* ML Training */}
              <Route path="/data" element={<DataOverview />} />
              <Route path="/training" element={<TrainingMonitor />} />
              <Route path="/baselines" element={<Baselines />} />
              <Route path="/automation" element={<Automation />} />

              {/* Cost Intelligence */}
              <Route path="/costs" element={<CostManagement />} />

              {/* System */}
              <Route path="/system" element={<System />} />
              <Route path="/status" element={<ModelStatus />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/settings" element={<Navigate to="/system" replace />} />

              {/* 404 - Catch all unmatched routes */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
          </BrowserRouter>
        </MockModeProvider>
      </UserRoleProvider>
    </ErrorBoundary>
  )
}

export default App
