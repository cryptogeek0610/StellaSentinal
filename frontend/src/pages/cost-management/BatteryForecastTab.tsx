/**
 * Battery Replacement Forecast tab for Cost Management.
 * Extracted from CostManagement.tsx for maintainability.
 */

import { formatCurrency, type BatteryForecastResponse } from '../../types/cost';

interface BatteryForecastTabProps {
  batteryForecast: BatteryForecastResponse | undefined;
  isLoading: boolean;
}

function ExportButton({ batteryForecast }: { batteryForecast: BatteryForecastResponse }) {
  const handleExport = () => {
    const today = new Date().toISOString().split('T')[0];
    const lines: string[] = [];

    lines.push('BATTERY REPLACEMENT FORECAST - PROCUREMENT SUMMARY');
    lines.push(`Generated: ${new Date().toLocaleDateString()}`);
    lines.push('');
    lines.push('SUMMARY');
    lines.push(`Total Batteries Due (30 Days),${batteryForecast.total_replacements_due_30_days}`);
    lines.push(`Total Batteries Due (90 Days),${batteryForecast.total_replacements_due_90_days}`);
    lines.push(`Estimated Cost (30 Days),$${batteryForecast.total_estimated_cost_30_days.toFixed(2)}`);
    lines.push(`Estimated Cost (90 Days),$${batteryForecast.total_estimated_cost_90_days.toFixed(2)}`);
    lines.push('');
    lines.push('DETAIL BY DEVICE MODEL');

    const headers = [
      'Device Model',
      'Unit Cost ($)',
      'Qty Needed (Immediate)',
      'Qty Needed (30 Days)',
      'Qty Needed (90 Days)',
      'Total Cost (30 Days)',
      'Total Cost (90 Days)',
      'Fleet Size',
      'Notes',
    ];
    lines.push(headers.join(','));

    const sortedForecasts = [...batteryForecast.forecasts].sort(
      (a, b) => b.devices_due_this_month - a.devices_due_this_month,
    );

    for (const f of sortedForecasts) {
      const immediateQty = f.devices_due_this_month;
      const notes =
        immediateQty > 0
          ? 'URGENT - Order Now'
          : f.devices_due_next_month > 0
            ? 'Plan for next month'
            : '';

      const row = [
        `"${f.device_model}"`,
        f.battery_replacement_cost.toFixed(2),
        immediateQty,
        f.devices_due_this_month,
        f.devices_due_in_90_days,
        f.estimated_cost_30_days.toFixed(2),
        f.estimated_cost_90_days.toFixed(2),
        f.device_count,
        `"${notes}"`,
      ];
      lines.push(row.join(','));
    }

    lines.push('');
    lines.push(
      `TOTAL,,-,${batteryForecast.total_replacements_due_30_days},${batteryForecast.total_replacements_due_90_days},$${batteryForecast.total_estimated_cost_30_days.toFixed(2)},$${batteryForecast.total_estimated_cost_90_days.toFixed(2)},${batteryForecast.total_devices_with_battery_data},`,
    );

    const csv = lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `battery-procurement-${today}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={handleExport}
      className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg flex items-center gap-2 transition-colors"
    >
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
        />
      </svg>
      Export for Procurement
    </button>
  );
}

function DataQualityBadge({ quality, devicesWithData }: { quality: string; devicesWithData?: number }) {
  if (quality === 'real') {
    return (
      <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-md flex items-center gap-1.5">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Based on real battery health data
      </span>
    );
  }
  if (quality === 'mixed') {
    return (
      <span className="px-2 py-1 bg-cyan-500/20 text-cyan-400 rounded-md flex items-center gap-1.5">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Mixed: {devicesWithData} devices with real data
      </span>
    );
  }
  if (quality === 'estimated') {
    return (
      <span className="px-2 py-1 bg-amber-500/20 text-amber-400 rounded-md flex items-center gap-1.5">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Based on estimated battery ages
      </span>
    );
  }
  return null;
}

export function BatteryForecastTab({ batteryForecast, isLoading }: BatteryForecastTabProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Battery Replacement Forecast</h2>
          <p className="text-sm text-slate-400 mt-1">
            Proactive replacement planning based on fleet battery health
          </p>
        </div>
        {batteryForecast && batteryForecast.forecasts.length > 0 && (
          <ExportButton batteryForecast={batteryForecast} />
        )}
      </div>

      {isLoading ? (
        <div className="animate-pulse space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-28 bg-slate-800 rounded-xl" />
            ))}
          </div>
        </div>
      ) : batteryForecast && batteryForecast.forecasts.length > 0 ? (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-amber-500/20 to-amber-600/10 p-5 rounded-xl border border-amber-500/40 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-16 h-16 bg-amber-500/10 rounded-bl-full" />
              <div className="text-xs font-medium text-amber-400 uppercase tracking-wide">Action Required</div>
              <div className="text-3xl font-bold text-white mt-2 tabular-nums">
                {batteryForecast.total_replacements_due_30_days.toLocaleString()}
              </div>
              <div className="text-sm text-amber-300/90 mt-1 font-medium tabular-nums">
                {formatCurrency(batteryForecast.total_estimated_cost_30_days)}
              </div>
              <div className="text-xs text-amber-400/70 mt-2">Due within 30 days</div>
            </div>

            <div className="bg-slate-800/60 p-5 rounded-xl border border-slate-700/50">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">90-Day Outlook</div>
              <div className="text-3xl font-bold text-white mt-2 tabular-nums">
                {batteryForecast.total_replacements_due_90_days.toLocaleString()}
              </div>
              <div className="text-sm text-slate-400 mt-1 tabular-nums">
                {formatCurrency(batteryForecast.total_estimated_cost_90_days)}
              </div>
              <div className="text-xs text-slate-500 mt-2">Total replacements needed</div>
            </div>

            <div className="bg-slate-800/60 p-5 rounded-xl border border-slate-700/50">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">Fleet Coverage</div>
              <div className="text-3xl font-bold text-white mt-2 tabular-nums">
                {batteryForecast.total_devices_with_battery_data.toLocaleString()}
              </div>
              <div className="text-sm text-slate-400 mt-1">devices tracked</div>
              <div className="text-xs text-slate-500 mt-2">
                {batteryForecast.forecasts.length} device models
              </div>
            </div>

            <div className="bg-slate-800/60 p-5 rounded-xl border border-slate-700/50">
              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide">Quarterly Budget</div>
              <div className="text-3xl font-bold text-white mt-2 tabular-nums">
                {formatCurrency(batteryForecast.total_estimated_cost_90_days)}
              </div>
              <div className="text-sm text-slate-400 mt-1">required</div>
              <div className="text-xs text-slate-500 mt-2">Plan procurement now</div>
            </div>
          </div>

          {/* Forecast Table */}
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/50 overflow-hidden">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-slate-700/50">
                  <th className="px-5 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">Device Model</th>
                  <th className="px-4 py-4 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider">Fleet</th>
                  <th className="px-4 py-4 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider w-48">Battery Health</th>
                  <th className="px-4 py-4 text-center text-xs font-semibold text-amber-400 uppercase tracking-wider">This Month</th>
                  <th className="px-4 py-4 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider">Next Month</th>
                  <th className="px-4 py-4 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider">90 Days</th>
                  <th className="px-4 py-4 text-right text-xs font-semibold text-amber-400 uppercase tracking-wider">30-Day Cost</th>
                  <th className="px-5 py-4 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">90-Day Cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/30">
                {batteryForecast.forecasts
                  .slice()
                  .sort((a, b) => b.devices_due_this_month - a.devices_due_this_month)
                  .map((forecast) => {
                    const healthPercent = Math.min(
                      100,
                      Math.round((forecast.avg_battery_age_months / forecast.battery_lifespan_months) * 100),
                    );
                    const isUrgent = healthPercent >= 80;
                    const isWarning = healthPercent >= 60 && healthPercent < 80;

                    return (
                      <tr key={forecast.device_model} className="hover:bg-slate-700/20 transition-colors">
                        <td className="px-5 py-4">
                          <div className="text-sm font-medium text-white">{forecast.device_model}</div>
                          <div className="text-xs text-slate-500 mt-0.5">
                            {formatCurrency(forecast.battery_replacement_cost)} / battery
                          </div>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className="text-sm font-medium text-slate-200 tabular-nums">
                            {forecast.device_count}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex items-center gap-3">
                            <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all ${
                                  isUrgent ? 'bg-amber-500' : isWarning ? 'bg-yellow-500' : 'bg-emerald-500'
                                }`}
                                style={{ width: `${healthPercent}%` }}
                              />
                            </div>
                            <span
                              className={`text-xs font-medium tabular-nums w-20 text-right ${
                                isUrgent ? 'text-amber-400' : isWarning ? 'text-yellow-400' : 'text-slate-400'
                              }`}
                            >
                              {Math.round(forecast.avg_battery_age_months)}/{forecast.battery_lifespan_months} mo
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-center">
                          {forecast.devices_due_this_month > 0 ? (
                            <span className="inline-flex items-center justify-center min-w-[2rem] px-2.5 py-1 bg-amber-500/20 text-amber-400 rounded-full text-sm font-semibold tabular-nums">
                              {forecast.devices_due_this_month}
                            </span>
                          ) : (
                            <span className="text-slate-600 text-sm">-</span>
                          )}
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className="text-sm text-slate-300 tabular-nums">
                            {forecast.devices_due_next_month || '-'}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <span className="text-sm text-slate-300 tabular-nums">
                            {forecast.devices_due_in_90_days}
                          </span>
                        </td>
                        <td className="px-4 py-4 text-right">
                          <span
                            className={`text-sm font-medium tabular-nums ${forecast.estimated_cost_30_days > 0 ? 'text-amber-400' : 'text-slate-500'}`}
                          >
                            {forecast.estimated_cost_30_days > 0
                              ? formatCurrency(forecast.estimated_cost_30_days)
                              : '-'}
                          </span>
                        </td>
                        <td className="px-5 py-4 text-right">
                          <span className="text-sm text-slate-300 tabular-nums">
                            {formatCurrency(forecast.estimated_cost_90_days)}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
              <tfoot>
                <tr className="border-t border-slate-600/50 bg-slate-800/50">
                  <td className="px-5 py-4 text-sm font-semibold text-white">Total</td>
                  <td className="px-4 py-4 text-center text-sm font-semibold text-white tabular-nums">
                    {batteryForecast.total_devices_with_battery_data.toLocaleString()}
                  </td>
                  <td className="px-4 py-4" />
                  <td className="px-4 py-4 text-center">
                    <span className="inline-flex items-center justify-center min-w-[2rem] px-2.5 py-1 bg-amber-500/30 text-amber-300 rounded-full text-sm font-bold tabular-nums">
                      {batteryForecast.total_replacements_due_30_days}
                    </span>
                  </td>
                  <td className="px-4 py-4" />
                  <td className="px-4 py-4 text-center text-sm font-semibold text-white tabular-nums">
                    {batteryForecast.total_replacements_due_90_days}
                  </td>
                  <td className="px-4 py-4 text-right text-sm font-bold text-amber-400 tabular-nums">
                    {formatCurrency(batteryForecast.total_estimated_cost_30_days)}
                  </td>
                  <td className="px-5 py-4 text-right text-sm font-semibold text-white tabular-nums">
                    {formatCurrency(batteryForecast.total_estimated_cost_90_days)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>

          {/* Insight Footer */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              <DataQualityBadge
                quality={batteryForecast.data_quality}
                devicesWithData={batteryForecast.devices_with_health_data}
              />
            </div>
            <span className="text-slate-500">
              Generated{' '}
              {new Date(batteryForecast.forecast_generated_at).toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
        </>
      ) : (
        <div className="text-center py-16 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="text-slate-400 mb-2">
            <svg className="w-12 h-12 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 9v2m0 4h.01M5.07 19H19a2 2 0 001.75-2.97l-7-12a2 2 0 00-3.5 0l-7 12A2 2 0 005.07 19z"
              />
            </svg>
          </div>
          <p className="text-slate-400 font-medium">No battery forecast data available</p>
          <p className="text-slate-500 text-sm mt-1">
            Add battery lifespan information to hardware costs to enable forecasting
          </p>
        </div>
      )}
    </div>
  );
}
