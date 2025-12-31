/**
 * Toggle Switch Component
 *
 * A modern toggle switch for boolean settings.
 * Matches the Stellar Operations design system.
 */

import { motion } from 'framer-motion';

interface ToggleSwitchProps {
    enabled: boolean;
    onChange: (enabled: boolean) => void;
    size?: 'sm' | 'md' | 'lg';
    variant?: 'stellar' | 'aurora' | 'emerald' | 'danger';
    disabled?: boolean;
    label?: string;
    'aria-label'?: string;
}

export function ToggleSwitch({
    enabled,
    onChange,
    size = 'md',
    variant = 'stellar',
    disabled = false,
    label,
    'aria-label': ariaLabel,
}: ToggleSwitchProps) {
    // Size configurations with proper spacing
    const sizeConfig = {
        sm: { track: 'w-9 h-5', thumb: 'w-4 h-4', offset: 2, travel: 16 },
        md: { track: 'w-11 h-6', thumb: 'w-5 h-5', offset: 2, travel: 20 },
        lg: { track: 'w-14 h-7', thumb: 'w-6 h-6', offset: 2, travel: 28 },
    };

    // Color variants matching Stellar design system
    const variantConfig = {
        stellar: {
            active: 'bg-gradient-to-r from-amber-500 to-amber-400',
            glow: '0 0 12px rgba(245, 166, 35, 0.4)',
        },
        aurora: {
            active: 'bg-gradient-to-r from-cyan-500 to-cyan-400',
            glow: '0 0 12px rgba(0, 217, 192, 0.4)',
        },
        emerald: {
            active: 'bg-gradient-to-r from-emerald-500 to-emerald-400',
            glow: '0 0 12px rgba(16, 185, 129, 0.4)',
        },
        danger: {
            active: 'bg-gradient-to-r from-red-500 to-red-400',
            glow: '0 0 12px rgba(255, 71, 87, 0.4)',
        },
    };

    const { track, thumb, offset, travel } = sizeConfig[size];
    const { active, glow } = variantConfig[variant];

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (disabled) return;
        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            onChange(!enabled);
        }
    };

    return (
        <button
            type="button"
            role="switch"
            aria-checked={enabled}
            aria-label={ariaLabel || label}
            disabled={disabled}
            onClick={() => !disabled && onChange(!enabled)}
            onKeyDown={handleKeyDown}
            className={`
                relative inline-flex shrink-0 cursor-pointer rounded-full
                transition-all duration-200 ease-out
                focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/50 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900
                ${track}
                ${enabled ? active : 'bg-slate-700'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:brightness-110'}
            `}
            style={{
                boxShadow: enabled && !disabled ? glow : 'inset 0 1px 2px rgba(0,0,0,0.3)',
            }}
        >
            {label && <span className="sr-only">{label}</span>}
            <motion.span
                aria-hidden="true"
                className={`
                    pointer-events-none rounded-full bg-white
                    ${thumb}
                `}
                style={{
                    position: 'absolute',
                    top: '50%',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)',
                }}
                initial={false}
                animate={{
                    x: enabled ? travel : offset,
                    y: '-50%',
                }}
                transition={{
                    type: 'spring',
                    stiffness: 500,
                    damping: 30,
                }}
            />
        </button>
    );
}
