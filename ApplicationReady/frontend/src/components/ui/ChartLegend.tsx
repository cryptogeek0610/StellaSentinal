/**
 * ChartLegend Component
 * 
 * Gradient legend for heatmaps and color-encoded visualizations.
 * Provides "Low ← → High" context for color intensity scales.
 */

import React from 'react';

interface ChartLegendProps {
    orientation?: 'horizontal' | 'vertical';
    colorStops: string[];
    labels: { low: string; high: string };
    position?: 'top-right' | 'bottom-right' | 'bottom-left' | 'top-left';
    className?: string;
}

export const ChartLegend: React.FC<ChartLegendProps> = ({
    orientation = 'horizontal',
    colorStops,
    labels,
    position = 'bottom-right',
    className = '',
}) => {
    const gradientId = `legend-gradient-${Math.random().toString(36).substr(2, 9)}`;

    const positionClasses = {
        'top-right': 'top-3 right-3',
        'bottom-right': 'bottom-3 right-3',
        'bottom-left': 'bottom-3 left-3',
        'top-left': 'top-3 left-3',
    };

    const isHorizontal = orientation === 'horizontal';

    return (
        <div className={`absolute ${positionClasses[position]} ${className}`}>
            <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="font-medium">{labels.low}</span>

                <svg
                    width={isHorizontal ? 80 : 16}
                    height={isHorizontal ? 16 : 80}
                    className="flex-shrink-0"
                >
                    <defs>
                        <linearGradient
                            id={gradientId}
                            x1={isHorizontal ? '0%' : '50%'}
                            y1={isHorizontal ? '50%' : '0%'}
                            x2={isHorizontal ? '100%' : '50%'}
                            y2={isHorizontal ? '50%' : '100%'}
                        >
                            {colorStops.map((color, index) => (
                                <stop
                                    key={index}
                                    offset={`${(index / (colorStops.length - 1)) * 100}%`}
                                    stopColor={color}
                                />
                            ))}
                        </linearGradient>
                    </defs>
                    <rect
                        x="0"
                        y="0"
                        width={isHorizontal ? 80 : 16}
                        height={isHorizontal ? 16 : 80}
                        fill={`url(#${gradientId})`}
                        rx="2"
                        className="stroke-slate-700/50"
                        strokeWidth="1"
                    />
                </svg>

                <span className="font-medium">{labels.high}</span>
            </div>
        </div>
    );
};

/**
 * Preset legend for common use cases
 */
export const HeatmapLegend: React.FC<{
    position?: 'top-right' | 'bottom-right' | 'bottom-left' | 'top-left';
    type?: 'stellar' | 'aurora' | 'danger' | 'plasma';
}> = ({ position = 'bottom-right', type = 'stellar' }) => {
    const presets = {
        stellar: {
            colorStops: ['rgba(245, 166, 35, 0.1)', 'rgba(245, 166, 35, 0.5)', 'rgba(245, 166, 35, 0.9)'],
            labels: { low: 'Low', high: 'High' },
        },
        aurora: {
            colorStops: ['rgba(0, 217, 192, 0.1)', 'rgba(0, 217, 192, 0.5)', 'rgba(0, 217, 192, 0.9)'],
            labels: { low: 'Low', high: 'High' },
        },
        danger: {
            colorStops: ['rgba(255, 71, 87, 0.1)', 'rgba(255, 71, 87, 0.5)', 'rgba(255, 71, 87, 0.9)'],
            labels: { low: 'Low', high: 'High' },
        },
        plasma: {
            colorStops: ['rgba(99, 102, 241, 0.1)', 'rgba(99, 102, 241, 0.5)', 'rgba(99, 102, 241, 0.9)'],
            labels: { low: 'Low', high: 'High' },
        },
    };

    const preset = presets[type];

    return (
        <ChartLegend
            orientation="horizontal"
            colorStops={preset.colorStops}
            labels={preset.labels}
            position={position}
        />
    );
};

export default ChartLegend;
