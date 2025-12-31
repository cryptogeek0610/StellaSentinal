/**
 * Location Attribute Settings Component
 *
 * Configure custom attributes from MobiControl for location grouping.
 * Extracted from System.tsx for better maintainability.
 */

import { useQuery } from '@tanstack/react-query';
import { api } from '../../api/client';
import { Card } from '../Card';
import { useLocationAttribute } from '../../hooks/useLocationAttribute';

// Loading spinner component
function LoadingSpinner({ size = 'sm' }: { size?: 'sm' | 'md' }) {
  const sizeClasses = { sm: 'h-4 w-4', md: 'h-6 w-6' };
  return (
    <svg className={`animate-spin ${sizeClasses[size]}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

export function LocationAttributeSettings() {
  const [attributeName, setAttributeName] = useLocationAttribute();

  // Fetch custom attributes from MobiControl
  const { data: customAttributesData, isLoading: isLoadingAttributes, error: queryError } = useQuery({
    queryKey: ['custom-attributes'],
    queryFn: () => api.getCustomAttributes(),
    retry: false,
    refetchOnWindowFocus: false,
  });

  const customAttributes = customAttributesData?.custom_attributes || [];
  const hasError = (queryError || customAttributesData?.error) && customAttributes.length === 0;
  const errorMessage = queryError?.message || customAttributesData?.error || 'Make sure MobiControl API is connected and devices are available.';

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">Custom Attributes</h2>
        <p className="text-sm text-slate-500 mt-1">
          Configure which custom attribute from MobiControl to use for grouping devices in the location heatmap
        </p>
      </div>

      <Card
        title={
          <div className="flex items-center justify-between w-full">
            <span className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Custom Attributes
            </span>
            <span className={`text-xs font-mono ${attributeName ? 'text-amber-400' : 'text-slate-500'}`}>
              {attributeName ? `Current: ${attributeName}` : 'Not selected'}
            </span>
          </div>
        }
        accent="stellar"
      >
        <div className="space-y-4">
          <div>
            <label
              htmlFor="location-attribute"
              className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3"
            >
              Custom Attribute Name
            </label>

            {isLoadingAttributes ? (
              <div className="flex items-center gap-2 text-sm text-slate-500" role="status" aria-label="Loading attributes">
                <LoadingSpinner />
                Loading custom attributes from MobiControl...
              </div>
            ) : hasError ? (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg" role="alert">
                <p className="text-sm text-red-400 mb-2">
                  Failed to fetch custom attributes from MobiControl
                </p>
                <p className="text-xs text-slate-500">
                  {errorMessage}
                </p>
                <div className="mt-3">
                  <label htmlFor="location-attribute-fallback" className="sr-only">
                    Enter custom attribute name manually
                  </label>
                  <input
                    id="location-attribute-fallback"
                    type="text"
                    value={attributeName}
                    onChange={(e) => setAttributeName(e.target.value)}
                    placeholder="Enter custom attribute name manually"
                    className="input-field w-full font-medium text-sm"
                  />
                </div>
              </div>
            ) : customAttributes.length > 0 ? (
              <select
                id="location-attribute"
                value={attributeName}
                onChange={(e) => setAttributeName(e.target.value)}
                className="select-field w-full font-medium text-sm"
                aria-describedby="attribute-help"
              >
                <option value="">Select a custom attribute...</option>
                {customAttributes.map((attr) => (
                  <option key={attr} value={attr}>
                    {attr}
                  </option>
                ))}
              </select>
            ) : (
              <div className="p-3 bg-slate-800/30 border border-slate-700/50 rounded-lg">
                <p className="text-sm text-slate-400 mb-2">
                  No custom attributes found on MobiControl devices
                </p>
                <p className="text-xs text-slate-500 mb-3">
                  Devices may not have custom attributes configured, or the MobiControl API connection may need to be checked.
                </p>
                <label htmlFor="location-attribute-manual" className="sr-only">
                  Enter custom attribute name manually
                </label>
                <input
                  id="location-attribute-manual"
                  type="text"
                  value={attributeName}
                  onChange={(e) => setAttributeName(e.target.value)}
                  placeholder="Enter custom attribute name manually"
                  className="input-field w-full font-medium text-sm"
                />
              </div>
            )}
          </div>

          <div
            id="attribute-help"
            className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50"
          >
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-sm font-medium text-slate-300 mb-1">How it works</p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Devices are grouped by the values of the selected custom attribute (e.g., "A101", "Warehouse-East").
                  The attribute name must match exactly what's configured in SOTI MobiControl.
                </p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
