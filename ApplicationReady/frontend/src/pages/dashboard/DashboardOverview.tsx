/**
 * Dashboard Overview View
 *
 * Priority queue, trend chart, quick actions, and system status.
 * Designed for balanced layout in both data-rich and empty states.
 */

import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNowStrict } from 'date-fns';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card } from '../../components/Card';
import { getSeverityFromScore } from '../../utils/severity';
import type { Anomaly, DashboardTrend } from '../../types/anomaly';

interface DashboardOverviewProps {
  anomalies: Anomaly[] | undefined;
  totalAnomalies: number;
  trends: DashboardTrend[] | undefined;
}

export function DashboardOverview({ anomalies, totalAnomalies, trends }: DashboardOverviewProps) {
  const hasAnomalies = anomalies && anomalies.length > 0;
  const hasTrendData = trends && trends.length > 0 && trends.some(t => t.anomaly_count > 0);

  return (
    <motion.div
      key="overview"
      className="space-y-6"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
    >
      {/* When there are anomalies: Show priority queue prominently */}
      {hasAnomalies ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {/* Priority Queue - Takes 2 columns on XL */}
          <div className="xl:col-span-2">
            <Card
              title={
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-3">
                    <span className="telemetry-label">Priority Queue</span>
                    <span className="px-2 py-0.5 text-xs font-bold bg-orange-500/20 text-orange-400 rounded-full border border-orange-500/30">
                      {totalAnomalies} open
                    </span>
                  </div>
                  <Link
                    to="/investigations"
                    className="text-xs text-amber-400 hover:text-amber-300 font-medium transition-colors"
                  >
                    View all →
                  </Link>
                </div>
              }
              noPadding
            >
              <div className="divide-y divide-slate-800/50" role="list" aria-label="Priority anomalies">
                <AnimatePresence mode="popLayout">
                  {anomalies.slice(0, 5).map((anomaly, index) => {
                    const severity = getSeverityFromScore(anomaly.anomaly_score);
                    return (
                      <motion.div
                        key={anomaly.id}
                        layout
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: index * 0.05 }}
                        className="group"
                        role="listitem"
                      >
                        <Link
                          to={`/investigations/${anomaly.id}`}
                          className="flex items-center gap-4 p-4 hover:bg-slate-800/30 transition-colors"
                          aria-label={`Device ${anomaly.device_id}, ${severity.label} severity, score ${anomaly.anomaly_score.toFixed(3)}`}
                        >
                          {/* Severity Indicator */}
                          <div className="relative" aria-hidden="true">
                            <div className={`w-2 h-10 rounded-full ${severity.color.dot}`} />
                            {severity.pulse && (
                              <motion.div
                                className={`absolute inset-0 w-2 h-10 rounded-full ${severity.color.dot}`}
                                animate={{ opacity: [0.5, 0, 0.5], scale: [1, 1.5, 1] }}
                                transition={{ duration: 1.5, repeat: Infinity }}
                              />
                            )}
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-semibold text-slate-200">
                                Device #{anomaly.device_id}
                              </span>
                              <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${severity.color.bg} ${severity.color.text}`}>
                                {severity.label}
                              </span>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-slate-500">
                              <span className="font-mono">Score: {anomaly.anomaly_score.toFixed(3)}</span>
                              <span>•</span>
                              <span>{formatDistanceToNowStrict(new Date(anomaly.timestamp))} ago</span>
                            </div>
                          </div>

                          {/* Action */}
                          <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                            <span className="px-3 py-1.5 text-xs font-medium bg-amber-500/20 text-amber-400 rounded-lg border border-amber-500/30">
                              Investigate
                            </span>
                          </div>
                        </Link>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </Card>
          </div>

          {/* Right Column - Trend & Quick Actions */}
          <div className="space-y-6">
            <TrendChart trends={trends} hasTrendData={hasTrendData} />
            <QuickActionsCard />
          </div>
        </div>
      ) : (
        /* Empty State: Show a welcoming dashboard with useful actions */
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
          {/* Welcome / Status Card */}
          <div className="md:col-span-2 xl:col-span-2">
            <div className="stellar-card rounded-xl p-6 h-full">
              <div className="flex items-start gap-4">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 flex items-center justify-center flex-shrink-0 border border-emerald-500/20">
                  <svg className="w-7 h-7 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white mb-1">All Systems Normal</h3>
                  <p className="text-sm text-slate-400 mb-4">
                    No anomalies detected. Your fleet is operating within expected parameters.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    <Link
                      to="/fleet"
                      className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg hover:bg-amber-500/20 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      View Fleet
                    </Link>
                    <Link
                      to="/investigations"
                      className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-300 bg-slate-800/50 border border-slate-700/50 rounded-lg hover:bg-slate-700/50 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      Past Investigations
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Trend Chart */}
          <div className="xl:col-span-2">
            <TrendChart trends={trends} hasTrendData={hasTrendData} />
          </div>

          {/* Quick Navigation Cards */}
          <QuickNavCard
            to="/fleet"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            }
            title="Fleet Overview"
            description="Monitor all devices"
            color="amber"
          />
          <QuickNavCard
            to="/insights"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            }
            title="AI Insights"
            description="View analysis results"
            color="indigo"
          />
          <QuickNavCard
            to="/system"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            }
            title="System Settings"
            description="Configure connections"
            color="slate"
          />
          <QuickNavCard
            to="/investigations?severity=critical"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            }
            title="Critical Alerts"
            description="Review urgent issues"
            color="red"
          />
        </div>
      )}
    </motion.div>
  );
}

// Trend Chart Component
function TrendChart({ trends, hasTrendData }: { trends: DashboardTrend[] | undefined; hasTrendData: boolean | undefined }) {
  // Calculate average for baseline
  const averageCount = trends && trends.length > 0
    ? trends.reduce((sum, t) => sum + t.anomaly_count, 0) / trends.length
    : 0;

  return (
    <Card
      title={<span className="telemetry-label">7-Day Trend</span>}
      showInfoIcon
      infoContent="Shows anomaly detection count over the last 7 days. Baseline represents average daily detections."
    >
      <div className="h-44" role="img" aria-label="7-day anomaly trend chart">
        {hasTrendData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trends || []}>
              <defs>
                <linearGradient id="stellarGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f5a623" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#f5a623" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                tickFormatter={(v) => format(new Date(v), 'EEE')}
                stroke="#475569"
                fontSize={10}
                tickLine={false}
                axisLine={false}
              />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(14, 17, 23, 0.95)',
                  border: '1px solid rgba(245, 166, 35, 0.2)',
                  borderRadius: '12px',
                  fontSize: '12px',
                }}
                labelFormatter={(v) => format(new Date(v), 'MMM d')}
                formatter={(value: number) => [value, 'Anomalies']}
              />
              {/* Baseline reference line */}
              {averageCount > 0 && (
                <ReferenceLine
                  y={averageCount}
                  stroke="rgba(148, 163, 184, 0.3)"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                  label={{
                    value: `Avg: ${averageCount.toFixed(1)}`,
                    fill: 'rgba(148, 163, 184, 0.6)',
                    fontSize: 10,
                    position: 'right',
                  }}
                />
              )}
              <Area
                type="monotone"
                dataKey="anomaly_count"
                stroke="#f5a623"
                strokeWidth={2}
                fill="url(#stellarGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-10 h-10 rounded-full bg-slate-800/50 flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <p className="text-sm text-slate-400 font-medium">No Activity</p>
            <p className="text-xs text-slate-600">Clean slate for the past 7 days</p>
            <p className="text-xs text-slate-700 mt-2">Baselines will appear when data is available</p>
          </div>
        )}
      </div>
    </Card>
  );
}

// Quick Actions Card Component
function QuickActionsCard() {
  return (
    <div className="stellar-card rounded-xl p-4">
      <p className="telemetry-label mb-3">Quick Actions</p>
      <nav className="space-y-2" aria-label="Quick actions">
        <Link
          to="/investigations?severity=critical"
          className="flex items-center gap-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 hover:bg-red-500/20 transition-colors"
        >
          <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center" aria-hidden="true">
            <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <span className="text-sm font-medium text-red-400">Review Critical Issues</span>
        </Link>
        <Link
          to="/fleet"
          className="flex items-center gap-3 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 hover:bg-amber-500/20 transition-colors"
        >
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center" aria-hidden="true">
            <svg className="w-4 h-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          </div>
          <span className="text-sm font-medium text-amber-400">Fleet Overview</span>
        </Link>
      </nav>
    </div>
  );
}

// Quick Navigation Card Component
interface QuickNavCardProps {
  to: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  color: 'amber' | 'indigo' | 'slate' | 'red' | 'emerald';
}

function QuickNavCard({ to, icon, title, description, color }: QuickNavCardProps) {
  const colorClasses = {
    amber: 'border-amber-500/20 hover:border-amber-500/40 hover:bg-amber-500/5',
    indigo: 'border-indigo-500/20 hover:border-indigo-500/40 hover:bg-indigo-500/5',
    slate: 'border-slate-700/50 hover:border-slate-600/50 hover:bg-slate-800/50',
    red: 'border-red-500/20 hover:border-red-500/40 hover:bg-red-500/5',
    emerald: 'border-emerald-500/20 hover:border-emerald-500/40 hover:bg-emerald-500/5',
  };

  const iconColorClasses = {
    amber: 'bg-amber-500/20 text-amber-400',
    indigo: 'bg-indigo-500/20 text-indigo-400',
    slate: 'bg-slate-700/50 text-slate-400',
    red: 'bg-red-500/20 text-red-400',
    emerald: 'bg-emerald-500/20 text-emerald-400',
  };

  const textColorClasses = {
    amber: 'text-amber-400',
    indigo: 'text-indigo-400',
    slate: 'text-slate-300',
    red: 'text-red-400',
    emerald: 'text-emerald-400',
  };

  return (
    <Link
      to={to}
      className={`stellar-card rounded-xl p-5 transition-all group ${colorClasses[color]}`}
    >
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${iconColorClasses[color]}`} aria-hidden="true">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-semibold ${textColorClasses[color]} group-hover:translate-x-0.5 transition-transform`}>
            {title}
          </p>
          <p className="text-xs text-slate-500 mt-0.5">{description}</p>
        </div>
        <svg
          className={`w-4 h-4 ${textColorClasses[color]} opacity-0 group-hover:opacity-100 transition-opacity`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  );
}
