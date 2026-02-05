/**
 * Reusable form components for the Setup page.
 * Extracted from Setup.tsx to improve maintainability.
 */

import { clsx } from 'clsx';

// ============================================================================
// FormField Component
// ============================================================================

export interface FormFieldProps {
  label: string;
  description?: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}

export function FormField({ label, description, required, error, children }: FormFieldProps) {
  return (
    <div className="space-y-2">
      <label className="block">
        <span className="text-sm font-medium text-white">
          {label}
          {required && <span className="text-amber-400 ml-1">*</span>}
        </span>
        {description && (
          <span className="block text-xs text-slate-500 mt-0.5">{description}</span>
        )}
      </label>
      {children}
      {error && (
        <p className="text-xs text-red-400 flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </p>
      )}
    </div>
  );
}

// ============================================================================
// TextInput Component
// ============================================================================

export interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password' | 'number';
  placeholder?: string;
  disabled?: boolean;
}

export function TextInput({ value, onChange, type = 'text', placeholder, disabled }: TextInputProps) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className={clsx(
        'w-full px-4 py-2.5 rounded-lg',
        'bg-slate-800/50 border border-slate-700/50',
        'text-white placeholder-slate-500',
        'focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/50',
        'transition-all duration-200',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    />
  );
}

// ============================================================================
// SelectInput Component
// ============================================================================

export interface SelectInputProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
}

export function SelectInput({ value, onChange, options, disabled }: SelectInputProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className={clsx(
        'w-full px-4 py-2.5 rounded-lg',
        'bg-slate-800/50 border border-slate-700/50',
        'text-white',
        'focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500/50',
        'transition-all duration-200',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// ============================================================================
// PasswordInput Component (TextInput with toggle visibility)
// ============================================================================

export interface PasswordInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  showPassword: boolean;
  onTogglePassword: () => void;
}

export function PasswordInput({
  value,
  onChange,
  placeholder,
  showPassword,
  onTogglePassword
}: PasswordInputProps) {
  return (
    <div className="relative">
      <TextInput
        type={showPassword ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
      />
      <button
        type="button"
        onClick={onTogglePassword}
        aria-label={showPassword ? 'Hide password' : 'Show password'}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
      >
        {showPassword ? (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        )}
      </button>
    </div>
  );
}

// ============================================================================
// LoadingSpinner Component
// ============================================================================

export function LoadingSpinner() {
  return (
    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

// ============================================================================
// SaveSectionButton Component
// ============================================================================

export interface SaveSectionButtonProps {
  sectionId: string;
  isPending: boolean;
  onSave: () => void;
}

const SECTION_LABELS: Record<string, string> = {
  'environment': 'Environment',
  'backend-db': 'Database',
  'xsight-db': 'XSight',
  'mobicontrol-db': 'MobiControl DB',
  'llm': 'LLM',
  'mobicontrol-api': 'API',
  'streaming': 'Streaming',
  'security': 'Security',
};

export function SaveSectionButton({ sectionId, isPending, onSave }: SaveSectionButtonProps) {
  const label = SECTION_LABELS[sectionId] || sectionId;

  return (
    <div className="flex items-center justify-end pt-4 border-t border-slate-700/30 mt-6">
      <button
        type="button"
        onClick={onSave}
        disabled={isPending}
        className={clsx(
          'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
          isPending ? 'opacity-50 cursor-not-allowed' : '',
          'bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30'
        )}
      >
        {isPending ? (
          <>
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Saving...
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            Save {label} Settings
          </>
        )}
      </button>
    </div>
  );
}
