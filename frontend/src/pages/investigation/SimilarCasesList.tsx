/**
 * Similar Cases List component.
 */

import { Link } from 'react-router-dom';
import type { SimilarCase } from '../../types/anomaly';

interface SimilarCasesListProps {
  cases: SimilarCase[];
}

export function SimilarCasesList({ cases }: SimilarCasesListProps) {
  if (!cases || cases.length === 0) {
    return <p className="text-sm text-slate-500 text-center py-4">No similar cases found</p>;
  }

  return (
    <div className="space-y-2">
      {cases.slice(0, 5).map((c) => (
        <Link
          key={c.case_id}
          to={`/investigations/${c.anomaly_id}`}
          className="block p-3 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors group"
        >
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-slate-500">Case #{c.anomaly_id}</span>
              <p className="text-sm text-slate-300 group-hover:text-white transition-colors">
                {c.device_name || `Device #${c.device_id}`}
              </p>
            </div>
            <div className="text-right">
              <span className={`text-xs px-2 py-0.5 rounded ${
                c.resolution_status === 'resolved' ? 'bg-emerald-500/20 text-emerald-400' :
                c.resolution_status === 'false_positive' ? 'bg-slate-700/50 text-slate-400' :
                'bg-orange-500/20 text-orange-400'
              }`}>
                {c.resolution_status}
              </span>
              <p className="text-[10px] text-slate-500 mt-1">
                {(c.similarity_score * 100).toFixed(0)}% similar
              </p>
            </div>
          </div>
          {c.resolution_summary && (
            <p className="text-xs text-slate-500 mt-1 truncate">{c.resolution_summary}</p>
          )}
        </Link>
      ))}
    </div>
  );
}
