/**
 * CardMenu Component
 * 
 * Kebab menu (⋮) for card-level actions like fullscreen, export, settings.
 * Provides discoverable widget customization without cluttering the UI.
 */

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface MenuAction {
    label: string;
    icon?: React.ReactNode;
    onClick: () => void;
    variant?: 'default' | 'danger';
}

interface CardMenuProps {
    onFullscreen?: () => void;
    onExport?: () => void;
    onSettings?: () => void;
    onRemove?: () => void;
    customActions?: MenuAction[];
    className?: string;
}

export const CardMenu: React.FC<CardMenuProps> = ({
    onFullscreen,
    onExport,
    onSettings,
    onRemove,
    customActions = [],
    className = '',
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const menuRef = useRef<HTMLDivElement>(null);

    // Close on outside click
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        const handleEscape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setIsOpen(false);
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            document.addEventListener('keydown', handleEscape);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [isOpen]);

    const defaultActions: MenuAction[] = [
        ...(onFullscreen
            ? [
                {
                    label: 'Fullscreen',
                    icon: (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
                            />
                        </svg>
                    ),
                    onClick: onFullscreen,
                },
            ]
            : []),
        ...(onExport
            ? [
                {
                    label: 'Export Data',
                    icon: (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                            />
                        </svg>
                    ),
                    onClick: onExport,
                },
            ]
            : []),
        ...(onSettings
            ? [
                {
                    label: 'Settings',
                    icon: (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                            />
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                            />
                        </svg>
                    ),
                    onClick: onSettings,
                },
            ]
            : []),
        ...customActions,
        ...(onRemove
            ? [
                {
                    label: 'Remove',
                    icon: (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                        </svg>
                    ),
                    onClick: onRemove,
                    variant: 'danger' as const,
                },
            ]
            : []),
    ];

    const allActions = defaultActions;

    if (allActions.length === 0) {
        return null;
    }

    return (
        <div ref={menuRef} className={`relative ${className}`}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="kebab-menu p-1 rounded hover:bg-slate-700/50 transition-colors"
                aria-label="Card actions"
                aria-haspopup="true"
                aria-expanded={isOpen}
            >
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
                    />
                </svg>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: -8 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: -8 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-full mt-1 z-50 min-w-[160px]"
                    >
                        <div className="stellar-card rounded-lg border border-slate-700/50 shadow-xl overflow-hidden">
                            <nav className="py-1" role="menu">
                                {allActions.map((action, index) => (
                                    <button
                                        key={index}
                                        onClick={() => {
                                            action.onClick();
                                            setIsOpen(false);
                                        }}
                                        className={`
                      w-full flex items-center gap-3 px-4 py-2.5 text-sm
                      transition-colors text-left
                      ${action.variant === 'danger'
                                                ? 'text-red-400 hover:bg-red-500/10'
                                                : 'text-slate-300 hover:bg-slate-700/50'
                                            }
                    `}
                                        role="menuitem"
                                    >
                                        {action.icon && <span className="flex-shrink-0">{action.icon}</span>}
                                        <span className="font-medium">{action.label}</span>
                                    </button>
                                ))}
                            </nav>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default CardMenu;
