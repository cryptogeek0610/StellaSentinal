/**
 * Storage Formatting Utilities
 *
 * Helper functions for formatting storage values
 * (KB, MB, GB) with appropriate scaling
 */

export interface FormattedStorage {
  value: number;
  unit: 'KB' | 'MB' | 'GB';
  formatted: string;
}

/**
 * Format storage value from KB to the most appropriate unit (KB, MB, or GB)
 * 
 * @param valueInKB - Storage value in kilobytes
 * @returns Formatted storage object with value, unit, and formatted string
 * 
 * @example
 * formatStorage(512000) // { value: 512, unit: 'MB', formatted: '512.00 MB' }
 * formatStorage(1536000) // { value: 1.536, unit: 'GB', formatted: '1.54 GB' }
 * formatStorage(512) // { value: 512, unit: 'KB', formatted: '512.0 KB' }
 */
export function formatStorage(valueInKB: number | null | undefined): FormattedStorage {
  if (valueInKB === null || valueInKB === undefined) {
    return { value: 0, unit: 'KB', formatted: '0.0 KB' };
  }

  // Convert to GB if >= 1,000,000 KB (1 GB)
  if (valueInKB >= 1_000_000) {
    const valueInGB = valueInKB / 1_000_000;
    return {
      value: valueInGB,
      unit: 'GB',
      formatted: `${valueInGB.toFixed(2)} GB`,
    };
  }

  // Convert to MB if >= 1,000 KB (1 MB)
  if (valueInKB >= 1_000) {
    const valueInMB = valueInKB / 1_000;
    return {
      value: valueInMB,
      unit: 'MB',
      formatted: `${valueInMB.toFixed(2)} MB`,
    };
  }

  // Keep as KB
  return {
    value: valueInKB,
    unit: 'KB',
    formatted: `${valueInKB.toFixed(1)} KB`,
  };
}

