import { Link } from 'react-router-dom';

interface EmptyStateAction {
  label: string;
  to: string;
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: EmptyStateAction;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      {icon && (
        <div className="w-14 h-14 mb-4 rounded-full bg-slate-800 flex items-center justify-center">
          {icon}
        </div>
      )}
      <p className="text-slate-300 font-medium">{title}</p>
      {description && (
        <p className="text-sm text-slate-500 mt-1 max-w-sm">{description}</p>
      )}
      {action && (
        <Link
          to={action.to}
          className="mt-4 px-4 py-2 text-sm font-medium rounded-lg bg-cyber-blue/20 text-cyber-blue hover:bg-cyber-blue/30 transition-colors"
        >
          {action.label}
        </Link>
      )}
    </div>
  );
}
