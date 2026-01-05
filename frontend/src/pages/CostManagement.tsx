import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import {
  formatCurrency,
  type HardwareCostCreate,
  type OperationalCostCreate,
  type CostAlertCreate,
  type CostCategory,
  type CostType,
  type ScopeType,
  type AlertThresholdType,
} from '../types/cost';

type TabType = 'summary' | 'hardware' | 'operational' | 'battery' | 'nff' | 'alerts' | 'history';

// ============================================================================
// SVG Icons for Categories (no emojis)
// ============================================================================

const CategoryIcons: Record<string, React.ReactNode> = {
  labor: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
      />
    </svg>
  ),
  downtime: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  ),
  support: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
      />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  infrastructure: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
      />
    </svg>
  ),
  maintenance: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z"
      />
    </svg>
  ),
  other: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
      />
    </svg>
  ),
  nff: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  ),
};

const getCategoryIconSvg = (category: string) => {
  return CategoryIcons[category] || CategoryIcons.other;
};

// Bell Icon for Alerts
const BellIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
      d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
    />
  </svg>
);

// ============================================================================
// Main Component
// ============================================================================

export default function CostManagement() {
  const [activeTab, setActiveTab] = useState<TabType>('summary');
  const [showHardwareForm, setShowHardwareForm] = useState(false);
  const [showOperationalForm, setShowOperationalForm] = useState(false);
  const [showAlertForm, setShowAlertForm] = useState(false);

  const queryClient = useQueryClient();

  // Queries
  const { data: costSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['costSummary'],
    queryFn: () => api.getCostSummary(),
  });

  const { data: hardwareCosts, isLoading: hardwareLoading } = useQuery({
    queryKey: ['hardwareCosts'],
    queryFn: () => api.getHardwareCosts(),
  });

  const { data: operationalCosts, isLoading: operationalLoading } = useQuery({
    queryKey: ['operationalCosts'],
    queryFn: () => api.getOperationalCosts(),
  });

  const { data: costHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['costHistory'],
    queryFn: () => api.getCostHistory(),
    enabled: activeTab === 'history',
  });

  const { data: deviceModels } = useQuery({
    queryKey: ['deviceModelTypes'],
    queryFn: () => api.getDeviceModelTypes(),
    enabled: showHardwareForm,
  });

  // Battery Forecast Query
  const { data: batteryForecast, isLoading: batteryLoading } = useQuery({
    queryKey: ['batteryForecast'],
    queryFn: () => api.getBatteryForecast(),
    enabled: activeTab === 'battery' || activeTab === 'summary',
  });

  // NFF Summary Query
  const { data: nffSummary, isLoading: nffLoading } = useQuery({
    queryKey: ['nffSummary'],
    queryFn: () => api.getNFFSummary(),
    enabled: activeTab === 'nff' || activeTab === 'summary',
  });

  // Cost Alerts Query
  const { data: costAlerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['costAlerts'],
    queryFn: () => api.getCostAlerts(),
    enabled: activeTab === 'alerts',
  });

  // Mutations
  const createHardwareCost = useMutation({
    mutationFn: (cost: HardwareCostCreate) => api.createHardwareCost(cost),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hardwareCosts'] });
      queryClient.invalidateQueries({ queryKey: ['costSummary'] });
      setShowHardwareForm(false);
    },
  });

  const deleteHardwareCost = useMutation({
    mutationFn: (id: number) => api.deleteHardwareCost(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hardwareCosts'] });
      queryClient.invalidateQueries({ queryKey: ['costSummary'] });
    },
  });

  const createOperationalCost = useMutation({
    mutationFn: (cost: OperationalCostCreate) => api.createOperationalCost(cost),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['operationalCosts'] });
      queryClient.invalidateQueries({ queryKey: ['costSummary'] });
      setShowOperationalForm(false);
    },
  });

  const deleteOperationalCost = useMutation({
    mutationFn: (id: number) => api.deleteOperationalCost(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['operationalCosts'] });
      queryClient.invalidateQueries({ queryKey: ['costSummary'] });
    },
  });

  // Cost Alert Mutations
  const createCostAlert = useMutation({
    mutationFn: (alert: CostAlertCreate) => api.createCostAlert(alert),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['costAlerts'] });
      setShowAlertForm(false);
    },
  });

  const deleteCostAlert = useMutation({
    mutationFn: (id: number) => api.deleteCostAlert(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['costAlerts'] });
    },
  });

  // Form handlers
  const handleHardwareCostSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const cost: HardwareCostCreate = {
      device_model: formData.get('device_model') as string,
      purchase_cost: parseFloat(formData.get('purchase_cost') as string),
      replacement_cost: formData.get('replacement_cost') ? parseFloat(formData.get('replacement_cost') as string) : undefined,
      repair_cost_avg: formData.get('repair_cost_avg') ? parseFloat(formData.get('repair_cost_avg') as string) : undefined,
      battery_replacement_cost: formData.get('battery_replacement_cost') ? parseFloat(formData.get('battery_replacement_cost') as string) : undefined,
      battery_lifespan_months: formData.get('battery_lifespan_months') ? parseInt(formData.get('battery_lifespan_months') as string) : undefined,
      depreciation_months: formData.get('depreciation_months') ? parseInt(formData.get('depreciation_months') as string) : 36,
      currency_code: 'USD',
      notes: formData.get('notes') as string || undefined,
    };
    createHardwareCost.mutate(cost);
  };

  const handleOperationalCostSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const cost: OperationalCostCreate = {
      name: formData.get('name') as string,
      category: formData.get('category') as CostCategory,
      amount: parseFloat(formData.get('amount') as string),
      cost_type: formData.get('cost_type') as CostType,
      scope_type: formData.get('scope_type') as ScopeType || 'tenant',
      currency_code: 'USD',
      is_active: true,
      description: formData.get('description') as string || undefined,
    };
    createOperationalCost.mutate(cost);
  };

  const handleCostAlertSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const alert: CostAlertCreate = {
      name: formData.get('name') as string,
      threshold_type: formData.get('threshold_type') as AlertThresholdType,
      threshold_value: parseFloat(formData.get('threshold_value') as string),
      is_active: true,
      notify_email: formData.get('notify_email') as string || undefined,
    };
    createCostAlert.mutate(alert);
  };

  const tabs: { id: TabType; label: string }[] = [
    { id: 'summary', label: 'Summary' },
    { id: 'hardware', label: 'Hardware Costs' },
    { id: 'operational', label: 'Operational Costs' },
    { id: 'battery', label: 'Battery Forecast' },
    { id: 'nff', label: 'NFF Tracking' },
    { id: 'alerts', label: 'Cost Alerts' },
    { id: 'history', label: 'History' },
  ];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-amber-400">Cost Intelligence</h1>
        <p className="text-slate-400 mt-1">
          Manage hardware and operational costs to track financial impact of device anomalies.
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-slate-700 mb-6">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-amber-500 text-amber-400'
                  : 'border-transparent text-slate-500 hover:text-slate-300 hover:border-slate-600'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Summary Tab */}
      {activeTab === 'summary' && (
        <div className="space-y-6">
          {summaryLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-24 bg-slate-800 rounded-xl" />
                ))}
              </div>
            </div>
          ) : costSummary ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Total Hardware Value</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {formatCurrency(costSummary.total_hardware_value)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {costSummary.device_count} devices
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Monthly Operational</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {formatCurrency(costSummary.total_operational_monthly)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {formatCurrency(costSummary.total_operational_annual)}/year
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Anomaly Impact (MTD)</div>
                  <div className="text-2xl font-bold text-red-400 mt-1">
                    {formatCurrency(costSummary.total_anomaly_impact_mtd)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {costSummary.anomaly_count_period} anomalies
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Anomaly Impact (YTD)</div>
                  <div className="text-2xl font-bold text-red-400 mt-1">
                    {formatCurrency(costSummary.total_anomaly_impact_ytd)}
                  </div>
                  {costSummary.anomaly_cost_trend_30d !== undefined && costSummary.anomaly_cost_trend_30d !== null && (
                    <div className={`text-xs mt-1 ${costSummary.anomaly_cost_trend_30d > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {costSummary.anomaly_cost_trend_30d > 0 ? '+' : ''}{costSummary.anomaly_cost_trend_30d.toFixed(1)}% vs last 30 days
                    </div>
                  )}
                </div>
              </div>

              {/* Category Breakdown */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <h3 className="font-semibold text-white mb-4">Cost by Category</h3>
                  <div className="space-y-3">
                    {costSummary.by_category.map((cat) => (
                      <div key={cat.category} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-slate-400">{getCategoryIconSvg(cat.category)}</span>
                          <span className="capitalize text-slate-300">{cat.category}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="w-32 bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-amber-500 h-2 rounded-full"
                              style={{ width: `${cat.percentage_of_total}%` }}
                            />
                          </div>
                          <span className="text-sm font-medium text-white w-20 text-right font-mono">
                            {formatCurrency(cat.total_cost)}
                          </span>
                        </div>
                      </div>
                    ))}
                    {costSummary.by_category.length === 0 && (
                      <p className="text-slate-500 text-sm">No operational costs configured</p>
                    )}
                  </div>
                </div>

                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <h3 className="font-semibold text-white mb-4">Cost by Device Model</h3>
                  <div className="space-y-3">
                    {costSummary.by_device_model.slice(0, 5).map((model) => (
                      <div key={model.device_model} className="flex items-center justify-between">
                        <div className="text-slate-300">{model.device_model}</div>
                        <div className="text-right">
                          <div className="text-sm font-medium text-white font-mono">
                            {formatCurrency(model.total_value)}
                          </div>
                          <div className="text-xs text-slate-500">
                            {model.device_count} devices @ {formatCurrency(model.unit_cost)}
                          </div>
                        </div>
                      </div>
                    ))}
                    {costSummary.by_device_model.length === 0 && (
                      <p className="text-slate-500 text-sm">No hardware costs configured</p>
                    )}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No cost data available. Add hardware and operational costs to get started.
            </div>
          )}
        </div>
      )}

      {/* Hardware Costs Tab */}
      {activeTab === 'hardware' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold text-white">Hardware Cost Entries</h2>
            <button
              onClick={() => setShowHardwareForm(true)}
              className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 transition-colors font-medium"
            >
              Add Hardware Cost
            </button>
          </div>

          {showHardwareForm && (
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
              <h3 className="font-medium text-white mb-4">New Hardware Cost Entry</h3>
              <form onSubmit={handleHardwareCostSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Device Model *</label>
                    <input
                      type="text"
                      name="device_model"
                      required
                      list="device-models"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="e.g., Zebra TC52"
                    />
                    <datalist id="device-models">
                      {deviceModels?.models.map((m) => (
                        <option key={m.device_model} value={m.device_model} />
                      ))}
                    </datalist>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Purchase Cost (USD) *</label>
                    <input
                      type="number"
                      name="purchase_cost"
                      required
                      min="0"
                      step="0.01"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="500.00"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Replacement Cost (USD)</label>
                    <input
                      type="number"
                      name="replacement_cost"
                      min="0"
                      step="0.01"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="550.00"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Avg Repair Cost (USD)</label>
                    <input
                      type="number"
                      name="repair_cost_avg"
                      min="0"
                      step="0.01"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="100.00"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Battery Replacement Cost (USD)</label>
                    <input
                      type="number"
                      name="battery_replacement_cost"
                      min="0"
                      step="0.01"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="45.00"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Battery Lifespan (months)</label>
                    <input
                      type="number"
                      name="battery_lifespan_months"
                      min="1"
                      max="60"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="18"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Depreciation Period (months)</label>
                    <input
                      type="number"
                      name="depreciation_months"
                      min="1"
                      max="120"
                      defaultValue="36"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Notes</label>
                    <input
                      type="text"
                      name="notes"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="Optional notes"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={createHardwareCost.isPending}
                    className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors font-medium"
                  >
                    {createHardwareCost.isPending ? 'Saving...' : 'Save'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowHardwareForm(false)}
                    className="px-4 py-2 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}

          {hardwareLoading ? (
            <div className="animate-pulse space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 bg-slate-800 rounded-lg" />
              ))}
            </div>
          ) : hardwareCosts?.costs.length ? (
            <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 overflow-hidden overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-700">
                <thead className="bg-slate-800/80">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Device Model</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Purchase</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Replacement</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Avg Repair</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-amber-400 uppercase">Battery</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Battery Life</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Devices</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Fleet Value</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {hardwareCosts.costs.map((cost) => (
                    <tr key={cost.id} className="hover:bg-slate-800/50">
                      <td className="px-4 py-3 text-sm font-medium text-white">{cost.device_model}</td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-right font-mono">{formatCurrency(cost.purchase_cost)}</td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-right font-mono">
                        {cost.replacement_cost ? formatCurrency(cost.replacement_cost) : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-right font-mono">
                        {cost.repair_cost_avg ? formatCurrency(cost.repair_cost_avg) : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-amber-400 text-right font-mono font-medium">
                        {cost.battery_replacement_cost ? formatCurrency(cost.battery_replacement_cost) : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-center">
                        {cost.battery_lifespan_months ? `${cost.battery_lifespan_months} mo` : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-center">{cost.device_count}</td>
                      <td className="px-4 py-3 text-sm font-medium text-emerald-400 text-right font-mono">
                        {formatCurrency(cost.total_fleet_value)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => deleteHardwareCost.mutate(cost.id)}
                          className="text-red-400 hover:text-red-300 text-sm transition-colors"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No hardware cost entries. Click "Add Hardware Cost" to create one.
            </div>
          )}
        </div>
      )}

      {/* Operational Costs Tab */}
      {activeTab === 'operational' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-semibold text-white">Operational Cost Entries</h2>
            <button
              onClick={() => setShowOperationalForm(true)}
              className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 transition-colors font-medium"
            >
              Add Operational Cost
            </button>
          </div>

          {showOperationalForm && (
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
              <h3 className="font-medium text-white mb-4">New Operational Cost Entry</h3>
              <form onSubmit={handleOperationalCostSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Name *</label>
                    <input
                      type="text"
                      name="name"
                      required
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="e.g., IT Support Hourly Rate"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Category *</label>
                    <select
                      name="category"
                      required
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    >
                      <option value="labor">Labor</option>
                      <option value="downtime">Downtime</option>
                      <option value="support">Support</option>
                      <option value="infrastructure">Infrastructure</option>
                      <option value="maintenance">Maintenance</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Amount (USD) *</label>
                    <input
                      type="number"
                      name="amount"
                      required
                      min="0"
                      step="0.01"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="50.00"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Cost Type *</label>
                    <select
                      name="cost_type"
                      required
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    >
                      <option value="hourly">Hourly</option>
                      <option value="daily">Daily</option>
                      <option value="per_incident">Per Incident</option>
                      <option value="fixed_monthly">Fixed Monthly</option>
                      <option value="per_device">Per Device</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Scope</label>
                    <select
                      name="scope_type"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    >
                      <option value="tenant">Tenant-wide</option>
                      <option value="location">Location-specific</option>
                      <option value="device_group">Device Group</option>
                      <option value="device_model">Device Model</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Description</label>
                    <input
                      type="text"
                      name="description"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="Optional description"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={createOperationalCost.isPending}
                    className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors font-medium"
                  >
                    {createOperationalCost.isPending ? 'Saving...' : 'Save'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowOperationalForm(false)}
                    className="px-4 py-2 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}

          {operationalLoading ? (
            <div className="animate-pulse space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 bg-slate-800 rounded-lg" />
              ))}
            </div>
          ) : operationalCosts?.costs.length ? (
            <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 overflow-hidden">
              <table className="min-w-full divide-y divide-slate-700">
                <thead className="bg-slate-800/80">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Name</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Category</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Amount</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Type</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Monthly Equiv.</th>
                    <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Status</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {operationalCosts.costs.map((cost) => (
                    <tr key={cost.id} className="hover:bg-slate-800/50">
                      <td className="px-4 py-3 text-sm font-medium text-white">{cost.name}</td>
                      <td className="px-4 py-3 text-sm text-slate-300">
                        <span className="flex items-center gap-2">
                          <span className="text-slate-400">{getCategoryIconSvg(cost.category)}</span>
                          <span className="capitalize">{cost.category}</span>
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300 text-right font-mono">{formatCurrency(cost.amount)}</td>
                      <td className="px-4 py-3 text-sm text-slate-300 capitalize">{cost.cost_type.replace('_', ' ')}</td>
                      <td className="px-4 py-3 text-sm font-medium text-amber-400 text-right font-mono">
                        {formatCurrency(cost.monthly_equivalent)}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`px-2 py-1 text-xs rounded-full ${cost.is_active ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-600 text-slate-400'}`}>
                          {cost.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => deleteOperationalCost.mutate(cost.id)}
                          className="text-red-400 hover:text-red-300 text-sm transition-colors"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {operationalCosts.total > 0 && (
                <div className="px-4 py-3 bg-slate-800/50 border-t border-slate-700 text-sm text-slate-400">
                  Total Monthly: <span className="text-white font-mono">{formatCurrency(operationalCosts.total_monthly_cost)}</span> |
                  Total Annual: <span className="text-white font-mono">{formatCurrency(operationalCosts.total_annual_cost)}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No operational cost entries. Click "Add Operational Cost" to create one.
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-white">Cost Change History</h2>
          {historyLoading ? (
            <div className="animate-pulse space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-12 bg-slate-800 rounded-lg" />
              ))}
            </div>
          ) : costHistory?.changes.length ? (
            <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 overflow-hidden">
              <table className="min-w-full divide-y divide-slate-700">
                <thead className="bg-slate-800/80">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Timestamp</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Action</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Entity</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Changed By</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/50">
                  {costHistory.changes.map((change) => (
                    <tr key={change.id} className="hover:bg-slate-800/50">
                      <td className="px-4 py-3 text-sm text-slate-300">
                        {new Date(change.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <span className={`px-2 py-1 text-xs rounded-full capitalize ${
                          change.action === 'create' ? 'bg-emerald-500/20 text-emerald-400' :
                          change.action === 'update' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-red-500/20 text-red-400'
                        }`}>
                          {change.action}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300">
                        <div className="text-white">{change.entity_name}</div>
                        <div className="text-xs text-slate-500">{change.entity_type}</div>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-300">{change.changed_by || 'System'}</td>
                      <td className="px-4 py-3 text-sm text-slate-300">
                        {change.field_changed && (
                          <span>
                            {change.field_changed}: <span className="text-slate-500">{change.old_value || '-'}</span> → <span className="text-white">{change.new_value || '-'}</span>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No changes recorded yet.
            </div>
          )}
        </div>
      )}

      {/* Battery Forecast Tab */}
      {activeTab === 'battery' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-white">Battery Replacement Forecast</h2>
              <p className="text-sm text-slate-400">Predicted battery replacements based on device lifespan data</p>
            </div>
          </div>

          {batteryLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-24 bg-slate-800 rounded-xl" />
                ))}
              </div>
            </div>
          ) : batteryForecast ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-amber-500/10 p-5 rounded-xl border border-amber-500/30">
                  <div className="text-sm text-amber-400">Due in 30 Days</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {batteryForecast.total_replacements_due_30_days}
                  </div>
                  <div className="text-sm text-amber-400 mt-1 font-mono">
                    {formatCurrency(batteryForecast.total_estimated_cost_30_days)}
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Due in 90 Days</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {batteryForecast.total_replacements_due_90_days}
                  </div>
                  <div className="text-sm text-slate-400 mt-1 font-mono">
                    {formatCurrency(batteryForecast.total_estimated_cost_90_days)}
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Devices Tracked</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {batteryForecast.total_devices_with_battery_data}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    With battery lifespan data
                  </div>
                </div>
                <div className="bg-emerald-500/10 p-5 rounded-xl border border-emerald-500/30">
                  <div className="text-sm text-emerald-400">Avg Cost/Battery</div>
                  <div className="text-2xl font-bold text-white mt-1 font-mono">
                    {formatCurrency(batteryForecast.total_estimated_cost_90_days / (batteryForecast.total_replacements_due_90_days || 1))}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Across all models
                  </div>
                </div>
              </div>

              {/* Forecast Table */}
              <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 overflow-hidden">
                <table className="min-w-full divide-y divide-slate-700">
                  <thead className="bg-slate-800/80">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase">Device Model</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Fleet Size</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Lifespan</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Avg Age</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-amber-400 uppercase">This Month</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">Next Month</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-slate-400 uppercase">90 Days</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-amber-400 uppercase">30-Day Cost</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase">90-Day Cost</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50">
                    {batteryForecast.forecasts.map((forecast) => (
                      <tr key={forecast.device_model} className="hover:bg-slate-800/50">
                        <td className="px-4 py-3 text-sm font-medium text-white">{forecast.device_model}</td>
                        <td className="px-4 py-3 text-sm text-slate-300 text-center">{forecast.device_count}</td>
                        <td className="px-4 py-3 text-sm text-slate-300 text-center">{forecast.battery_lifespan_months} mo</td>
                        <td className="px-4 py-3 text-sm text-center">
                          <span className={forecast.avg_battery_age_months >= forecast.battery_lifespan_months * 0.8 ? 'text-amber-400' : 'text-slate-300'}>
                            {forecast.avg_battery_age_months} mo
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-center">
                          {forecast.devices_due_this_month > 0 ? (
                            <span className="px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full font-medium">
                              {forecast.devices_due_this_month}
                            </span>
                          ) : (
                            <span className="text-slate-500">0</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-slate-300 text-center">{forecast.devices_due_next_month}</td>
                        <td className="px-4 py-3 text-sm text-slate-300 text-center">{forecast.devices_due_in_90_days}</td>
                        <td className="px-4 py-3 text-sm font-medium text-amber-400 text-right font-mono">
                          {formatCurrency(forecast.estimated_cost_30_days)}
                        </td>
                        <td className="px-4 py-3 text-sm text-slate-300 text-right font-mono">
                          {formatCurrency(forecast.estimated_cost_90_days)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="text-xs text-slate-500 text-right">
                Forecast generated: {new Date(batteryForecast.forecast_generated_at).toLocaleString()}
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No battery data available. Add battery lifespan to hardware costs to enable forecasting.
            </div>
          )}
        </div>
      )}

      {/* NFF Tracking Tab */}
      {activeTab === 'nff' && (
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-white">No Fault Found (NFF) Tracking</h2>
              <p className="text-sm text-slate-400">Track unnecessary repair and investigation costs from false positives</p>
            </div>
          </div>

          {nffLoading ? (
            <div className="animate-pulse space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-24 bg-slate-800 rounded-xl" />
                ))}
              </div>
            </div>
          ) : nffSummary ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-red-500/10 p-5 rounded-xl border border-red-500/30">
                  <div className="text-sm text-red-400">Total NFF Cases</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {nffSummary.total_nff_count}
                  </div>
                  <div className={`text-sm mt-1 ${nffSummary.trend_30_days < 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {nffSummary.trend_30_days > 0 ? '+' : ''}{nffSummary.trend_30_days.toFixed(1)}% vs last 30 days
                  </div>
                </div>
                <div className="bg-amber-500/10 p-5 rounded-xl border border-amber-500/30">
                  <div className="text-sm text-amber-400">Total NFF Cost</div>
                  <div className="text-2xl font-bold text-white mt-1 font-mono">
                    {formatCurrency(nffSummary.total_nff_cost)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Wasted investigation costs
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">Avg Cost per NFF</div>
                  <div className="text-2xl font-bold text-white mt-1 font-mono">
                    {formatCurrency(nffSummary.avg_cost_per_nff)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Per unnecessary investigation
                  </div>
                </div>
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <div className="text-sm text-slate-400">NFF Rate</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {nffSummary.nff_rate_percent.toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Of all investigations
                  </div>
                </div>
              </div>

              {/* Breakdown Grids */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* By Device Model */}
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                    {getCategoryIconSvg('nff')}
                    NFF by Device Model
                  </h3>
                  <div className="space-y-3">
                    {nffSummary.by_device_model.map((item) => (
                      <div key={item.device_model} className="flex items-center justify-between">
                        <div className="text-slate-300">{item.device_model}</div>
                        <div className="flex items-center gap-4">
                          <span className="text-sm text-slate-500">{item.count} cases</span>
                          <span className="text-sm font-medium text-amber-400 w-20 text-right font-mono">
                            {formatCurrency(item.total_cost)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* By Resolution Type */}
                <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
                  <h3 className="font-semibold text-white mb-4">NFF by Resolution Type</h3>
                  <div className="space-y-3">
                    {nffSummary.by_resolution.map((item) => (
                      <div key={item.resolution} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${
                            item.resolution === 'no_fault_found' ? 'bg-red-400' :
                            item.resolution === 'user_error' ? 'bg-amber-400' :
                            item.resolution === 'intermittent' ? 'bg-blue-400' :
                            'bg-purple-400'
                          }`} />
                          <span className="text-slate-300 capitalize">{item.resolution.replace(/_/g, ' ')}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-sm text-slate-500">{item.count} cases</span>
                          <span className="text-sm font-medium text-white w-20 text-right font-mono">
                            {formatCurrency(item.total_cost)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Recommendation */}
              <div className="bg-cyan-500/10 p-4 rounded-xl border border-cyan-500/30">
                <div className="flex items-start gap-3">
                  <span className="text-cyan-400 mt-0.5">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </span>
                  <div>
                    <p className="text-sm font-medium text-cyan-400">Reduce NFF Costs</p>
                    <p className="text-sm text-slate-300 mt-1">
                      Your NFF rate of {nffSummary.nff_rate_percent.toFixed(1)}% is {nffSummary.nff_rate_percent > 20 ? 'above' : 'within'} the industry average of 20-25%.
                      {nffSummary.nff_rate_percent > 20 && ' Consider improving anomaly detection thresholds to reduce false positives.'}
                    </p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No NFF data available yet.
            </div>
          )}
        </div>
      )}

      {/* Cost Alerts Tab */}
      {activeTab === 'alerts' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-lg font-semibold text-white">Cost Threshold Alerts</h2>
              <p className="text-sm text-slate-400">Get notified when costs exceed defined thresholds</p>
            </div>
            <button
              onClick={() => setShowAlertForm(true)}
              className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 transition-colors font-medium"
            >
              Create Alert
            </button>
          </div>

          {showAlertForm && (
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
              <h3 className="font-medium text-white mb-4">New Cost Alert</h3>
              <form onSubmit={handleCostAlertSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Alert Name *</label>
                    <input
                      type="text"
                      name="name"
                      required
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="e.g., Daily Cost Limit"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Threshold Type *</label>
                    <select
                      name="threshold_type"
                      required
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    >
                      <option value="anomaly_cost_daily">Daily Anomaly Cost</option>
                      <option value="anomaly_cost_monthly">Monthly Anomaly Cost</option>
                      <option value="battery_forecast">Battery Replacement (30-day)</option>
                      <option value="operational_cost">Operational Cost (Monthly)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Threshold Amount (USD) *</label>
                    <input
                      type="number"
                      name="threshold_value"
                      required
                      min="0"
                      step="1"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="5000"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">Notify Email</label>
                    <input
                      type="email"
                      name="notify_email"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                      placeholder="ops@company.com"
                    />
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={createCostAlert.isPending}
                    className="px-4 py-2 bg-amber-500 text-slate-900 rounded-lg hover:bg-amber-400 disabled:opacity-50 transition-colors font-medium"
                  >
                    {createCostAlert.isPending ? 'Creating...' : 'Create Alert'}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowAlertForm(false)}
                    className="px-4 py-2 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          )}

          {alertsLoading ? (
            <div className="animate-pulse space-y-2">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-20 bg-slate-800 rounded-lg" />
              ))}
            </div>
          ) : costAlerts?.alerts.length ? (
            <div className="space-y-3">
              {costAlerts.alerts.map((alert) => (
                <div key={alert.id} className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      alert.is_active ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700 text-slate-500'
                    }`}>
                      <BellIcon />
                    </div>
                    <div>
                      <div className="font-medium text-white">{alert.name}</div>
                      <div className="text-sm text-slate-400">
                        Alert when <span className="text-slate-300">{alert.threshold_type.replace(/_/g, ' ')}</span> exceeds{' '}
                        <span className="text-amber-400 font-mono">{formatCurrency(alert.threshold_value)}</span>
                      </div>
                      {alert.notify_email && (
                        <div className="text-xs text-slate-500 mt-1">
                          Notify: {alert.notify_email}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      {alert.last_triggered && (
                        <div className="text-xs text-slate-500">
                          Last triggered: {new Date(alert.last_triggered).toLocaleDateString()}
                        </div>
                      )}
                      <div className="text-xs text-slate-500">
                        Triggered {alert.trigger_count}x
                      </div>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      alert.is_active ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-600 text-slate-400'
                    }`}>
                      {alert.is_active ? 'Active' : 'Inactive'}
                    </span>
                    <button
                      onClick={() => deleteCostAlert.mutate(alert.id)}
                      className="text-red-400 hover:text-red-300 text-sm transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 bg-slate-800/30 rounded-xl border border-slate-700/50">
              No cost alerts configured. Click "Create Alert" to set up threshold notifications.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
