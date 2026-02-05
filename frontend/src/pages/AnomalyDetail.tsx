import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useAnomaly, useResolveAnomaly, useAddNote } from '../hooks/useAnomalies';
import { format } from 'date-fns';
import { motion } from 'framer-motion';
import { Card } from '../components/Card';
import { Breadcrumb } from '../components/Breadcrumb';
import { formatStorage } from '../utils/storage';

function AnomalyDetail() {
  const { id } = useParams<{ id: string }>();
  const anomalyId = parseInt(id || '0');

  const { data: anomaly, isLoading } = useAnomaly(anomalyId);
  const resolveMutation = useResolveAnomaly();
  const addNoteMutation = useAddNote();

  const [noteText, setNoteText] = useState('');
  const [showResolveForm, setShowResolveForm] = useState(false);

  const getScoreSeverity = (score: number) => {
    if (score <= -0.7) return { level: 'Critical', color: 'text-cyber-red', badge: 'badge-critical' };
    if (score <= -0.5) return { level: 'High', color: 'text-cyber-orange', badge: 'badge-warning' };
    if (score <= -0.3) return { level: 'Medium', color: 'text-cyber-blue', badge: 'badge-info' };
    return { level: 'Low', color: 'text-slate-400', badge: 'badge-neutral' };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="absolute inset-0 rounded-full border-2 border-cyber-blue/20"></div>
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyber-blue animate-spin"></div>
          </div>
          <p className="text-slate-400 font-mono text-sm">Loading anomaly details...</p>
        </div>
      </div>
    );
  }

  if (!anomaly) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cyber-red/20 flex items-center justify-center">
            <svg className="w-8 h-8 text-cyber-red" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <p className="text-cyber-red font-medium">Anomaly not found</p>
        </div>
      </div>
    );
  }

  const severity = getScoreSeverity(anomaly.anomaly_score);

  const handleResolve = () => {
    resolveMutation.mutate(
      {
        id: anomalyId,
        status: 'resolved',
        notes: noteText,
      },
      {
        onSuccess: () => {
          setShowResolveForm(false);
          setNoteText('');
        },
      }
    );
  };

  const handleAddNote = () => {
    if (!noteText.trim()) return;
    addNoteMutation.mutate(
      {
        id: anomalyId,
        note: noteText,
      },
      {
        onSuccess: () => {
          setNoteText('');
        },
      }
    );
  };

  const freeStorageFormatted = formatStorage(anomaly.total_free_storage_kb);
  
  const metricItems = [
    { label: 'Battery Drop', value: anomaly.total_battery_level_drop, unit: '%' },
    { label: 'Free Storage', value: freeStorageFormatted.value, unit: freeStorageFormatted.unit, formattedDisplay: freeStorageFormatted.formatted },
    { label: 'Download', value: anomaly.download, unit: 'MB' },
    { label: 'Upload', value: anomaly.upload, unit: 'MB' },
    { label: 'Offline Time', value: anomaly.offline_time, unit: 'min' },
    { label: 'Disconnect Count', value: anomaly.disconnect_count, unit: '' },
    { label: 'WiFi Signal', value: anomaly.wifi_signal_strength, unit: 'dBm' },
    { label: 'Connection Time', value: anomaly.connection_time, unit: 's' },
  ].filter((item) => item.value !== null && item.value !== undefined);

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: 'Anomalies', to: '/anomalies' },
          { label: `Anomaly #${anomaly.id}` },
        ]}
      />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Anomaly #{anomaly.id}
          </h1>
          <p className="text-sm text-slate-500 mt-1 font-mono">
            Device {anomaly.device_id} • Detected{' '}
            {format(new Date(anomaly.timestamp), 'PPP p')}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {anomaly.status !== 'resolved' && (
            <button
              onClick={() => setShowResolveForm(!showResolveForm)}
              className="btn-success"
            >
              <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Mark Resolved
            </button>
          )}
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="text-center" accent="red">
          <div className="py-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Anomaly Score
            </p>
            <p className={`text-3xl font-bold font-mono ${severity.color}`}>
              {anomaly.anomaly_score.toFixed(3)}
            </p>
          </div>
        </Card>
        <Card className="text-center">
          <div className="py-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Severity
            </p>
            <span className={severity.badge}>{severity.level}</span>
          </div>
        </Card>
        <Card className="text-center">
          <div className="py-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Status
            </p>
            <span
              className={
                anomaly.status === 'open'
                  ? 'badge-critical'
                  : anomaly.status === 'resolved'
                  ? 'badge-success'
                  : 'badge-warning'
              }
            >
              {anomaly.status}
            </span>
          </div>
        </Card>
        <Card className="text-center">
          <div className="py-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Device
            </p>
            <Link
              to={`/devices/${anomaly.device_id}`}
              className="text-lg font-semibold text-cyber-blue hover:text-cyan-300 transition-colors"
            >
              #{anomaly.device_id}
            </Link>
          </div>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline Chart */}
        <Card title="Metric Timeline (24h)" className="lg:col-span-2">
          <div className="h-64 flex items-center justify-center">
            <p className="text-sm text-slate-500">
              Timeline data is not available for this anomaly yet.
            </p>
          </div>
        </Card>

        {/* Metric Values */}
        <Card title="Metric Values">
          <div className="space-y-3">
            {metricItems.length > 0 ? (
              metricItems.map((item, index) => (
                <motion.div
                  key={item.label}
                  className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <span className="text-sm text-slate-400">{item.label}</span>
                  <span className="text-sm font-mono font-semibold text-slate-200">
                    {(item as any).formattedDisplay !== undefined 
                      ? (item as any).formattedDisplay
                      : item.value + (item.unit ? ` ${item.unit}` : '')}
                  </span>
                </motion.div>
              ))
            ) : (
              <p className="text-sm text-slate-500 text-center py-4">
                No metric data available
              </p>
            )}
          </div>
        </Card>
      </div>

      {/* Investigation Notes */}
      <Card title="Investigation Notes">
        <div className="space-y-4">
          {anomaly.investigation_notes.length > 0 ? (
            anomaly.investigation_notes.map((note, index) => (
              <motion.div
                key={note.id}
                className="border-l-2 border-cyber-blue/50 pl-4 py-2"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-slate-200">{note.user}</span>
                  <span className="text-xs text-slate-500 font-mono">
                    {format(new Date(note.created_at), 'MMM dd, HH:mm')}
                  </span>
                </div>
                <p className="text-sm text-slate-400">{note.note}</p>
              </motion.div>
            ))
          ) : (
            <p className="text-sm text-slate-500 text-center py-4">
              No investigation notes yet
            </p>
          )}
        </div>

        {/* Add Note Form */}
        <div className="mt-6 pt-6 border-t border-slate-700/50">
          <textarea
            value={noteText}
            onChange={(e) => setNoteText(e.target.value)}
            placeholder="Add investigation note..."
            className="input-field min-h-[100px] resize-none"
            rows={3}
          />
          <div className="mt-3 flex items-center gap-3">
            <button
              onClick={handleAddNote}
              disabled={!noteText.trim() || addNoteMutation.isPending}
              className="btn-primary"
            >
              {addNoteMutation.isPending ? 'Adding...' : 'Add Note'}
            </button>
            {showResolveForm && (
              <button
                onClick={handleResolve}
                disabled={resolveMutation.isPending}
                className="btn-success"
              >
                {resolveMutation.isPending ? 'Resolving...' : 'Resolve with Note'}
              </button>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}

export default AnomalyDetail;
