/**
 * Example Usage - UI Components Demo
 *
 * This file demonstrates how to use the new UI components.
 * Import individual examples into your pages as needed.
 */

import React, { useState } from 'react';
import { EmptyState } from './EmptyState';
import { Modal, ConfirmDialog } from './Modal';
import { useToast } from './useToast';
import { AnimatedList, AnimatedListItem } from './AnimatedList';
import { ErrorBoundary } from './ErrorBoundary';

// ============================================
// Example 1: EmptyState Usage
// ============================================

export const EmptyStateExamples: React.FC = () => {
  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-white">EmptyState Examples</h2>

      {/* No Data */}
      <div className="stellar-card p-6 rounded-xl">
        <h3 className="text-sm text-slate-400 mb-4">No Data Variant</h3>
        <EmptyState
          variant="no-data"
          title="No anomalies detected"
          description="Your fleet is operating normally. Check back later for updates."
        />
      </div>

      {/* No Results with Action */}
      <div className="stellar-card p-6 rounded-xl">
        <h3 className="text-sm text-slate-400 mb-4">No Results with Action</h3>
        <EmptyState
          variant="no-results"
          title="No matching devices"
          description="Try adjusting your search terms or clearing filters."
          action={{
            label: 'Clear filters',
            onClick: () => alert('Filters cleared!'),
          }}
        />
      </div>

      {/* Not Configured with Two Actions */}
      <div className="stellar-card p-6 rounded-xl">
        <h3 className="text-sm text-slate-400 mb-4">Not Configured</h3>
        <EmptyState
          variant="not-configured"
          title="Database not connected"
          description="Connect your data warehouse to start analyzing device telemetry."
          action={{
            label: 'Configure connection',
            onClick: () => alert('Opening setup...'),
          }}
          secondaryAction={{
            label: 'Learn more',
            onClick: () => alert('Opening docs...'),
          }}
        />
      </div>
    </div>
  );
};

// ============================================
// Example 2: Toast Notifications
// ============================================

export const ToastExamples: React.FC = () => {
  const toast = useToast();

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">Toast Examples</h2>
      <p className="text-sm text-slate-400">
        Click the buttons to show different toast types.
      </p>

      <div className="flex flex-wrap gap-3">
        <button
          className="btn-success"
          onClick={() => toast.success('Success!', 'Your changes have been saved.')}
        >
          Success Toast
        </button>

        <button
          className="btn-danger"
          onClick={() => toast.error('Error', 'Failed to save changes. Please try again.')}
        >
          Error Toast
        </button>

        <button
          className="btn-primary"
          onClick={() => toast.warning('Warning', 'Connection is unstable.')}
        >
          Warning Toast
        </button>

        <button
          className="btn-secondary"
          onClick={() => toast.info('Info', 'A new update is available.')}
        >
          Info Toast
        </button>

        <button
          className="btn-ghost"
          onClick={() =>
            toast.addToast({
              type: 'info',
              title: 'Custom Toast',
              description: 'This toast has a custom action button.',
              duration: 10000,
              action: {
                label: 'Undo',
                onClick: () => alert('Action undone!'),
              },
            })
          }
        >
          Toast with Action
        </button>
      </div>
    </div>
  );
};

// ============================================
// Example 3: Modal & ConfirmDialog
// ============================================

export const ModalExamples: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handleConfirm = async () => {
    setIsLoading(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setIsLoading(false);
    setIsConfirmOpen(false);
    toast.success('Deleted', 'Item has been removed.');
  };

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">Modal Examples</h2>

      <div className="flex gap-3">
        <button className="btn-primary" onClick={() => setIsModalOpen(true)}>
          Open Form Modal
        </button>

        <button className="btn-danger" onClick={() => setIsConfirmOpen(true)}>
          Delete with Confirm
        </button>
      </div>

      {/* Form Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Edit Device"
        description="Update the device configuration"
        size="md"
        footer={
          <>
            <button className="btn-secondary" onClick={() => setIsModalOpen(false)}>
              Cancel
            </button>
            <button
              className="btn-primary"
              onClick={() => {
                toast.success('Saved', 'Device updated successfully.');
                setIsModalOpen(false);
              }}
            >
              Save Changes
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <label className="block">
            <span className="text-sm font-medium text-slate-300">Device Name</span>
            <input
              type="text"
              className="input-stellar mt-1"
              placeholder="Enter device name"
              defaultValue="Device-001"
            />
          </label>

          <label className="block">
            <span className="text-sm font-medium text-slate-300">Location</span>
            <select className="select-field mt-1">
              <option>Warehouse A</option>
              <option>Warehouse B</option>
              <option>Office</option>
            </select>
          </label>

          <label className="block">
            <span className="text-sm font-medium text-slate-300">Notes</span>
            <textarea
              className="input-stellar mt-1"
              rows={3}
              placeholder="Add any notes..."
            />
          </label>
        </div>
      </Modal>

      {/* Confirm Dialog */}
      <ConfirmDialog
        isOpen={isConfirmOpen}
        onClose={() => setIsConfirmOpen(false)}
        onConfirm={handleConfirm}
        title="Delete this device?"
        description="This action cannot be undone. All device data and history will be permanently removed."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        isLoading={isLoading}
      />
    </div>
  );
};

// ============================================
// Example 4: AnimatedList
// ============================================

const sampleDevices = [
  { id: 1, name: 'Device-001', status: 'Online', location: 'Warehouse A' },
  { id: 2, name: 'Device-002', status: 'Warning', location: 'Warehouse B' },
  { id: 3, name: 'Device-003', status: 'Offline', location: 'Office' },
  { id: 4, name: 'Device-004', status: 'Online', location: 'Warehouse A' },
];

export const AnimatedListExamples: React.FC = () => {
  const [showList, setShowList] = useState(true);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">AnimatedList Example</h2>

      <button className="btn-secondary" onClick={() => setShowList(!showList)}>
        {showList ? 'Hide List' : 'Show List'}
      </button>

      {showList && (
        <AnimatedList
          className="space-y-2"
          animation="slide-up"
          staggerDelay={0.08}
        >
          {sampleDevices.map((device) => (
            <div
              key={device.id}
              className="stellar-card-hover p-4 rounded-xl flex items-center justify-between"
            >
              <div>
                <h4 className="font-medium text-white">{device.name}</h4>
                <p className="text-sm text-slate-400">{device.location}</p>
              </div>
              <span
                className={`badge ${
                  device.status === 'Online'
                    ? 'badge-aurora'
                    : device.status === 'Warning'
                      ? 'badge-warning'
                      : 'badge-neutral'
                }`}
              >
                {device.status}
              </span>
            </div>
          ))}
        </AnimatedList>
      )}

      <h3 className="text-sm font-medium text-slate-300 mt-8">Individual Items with Scale Animation</h3>
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Total Devices', value: '1,234' },
          { label: 'Online', value: '1,180' },
          { label: 'Warnings', value: '42' },
          { label: 'Offline', value: '12' },
        ].map((stat, index) => (
          <AnimatedListItem key={stat.label} delay={index * 0.1} animation="scale">
            <div className="stellar-card p-4 text-center rounded-xl">
              <div className="text-2xl font-bold text-white">{stat.value}</div>
              <div className="text-xs text-slate-400">{stat.label}</div>
            </div>
          </AnimatedListItem>
        ))}
      </div>
    </div>
  );
};

// ============================================
// Example 5: ErrorBoundary
// ============================================

// Component that will throw an error
const BuggyComponent: React.FC<{ shouldCrash?: boolean }> = ({ shouldCrash }) => {
  if (shouldCrash) {
    throw new Error('This component crashed intentionally!');
  }
  return (
    <div className="stellar-card p-4 rounded-xl">
      <p className="text-slate-300">This component is working fine.</p>
    </div>
  );
};

export const ErrorBoundaryExample: React.FC = () => {
  const [shouldCrash, setShouldCrash] = useState(false);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">ErrorBoundary Example</h2>

      <button className="btn-danger" onClick={() => setShouldCrash(true)}>
        Trigger Error
      </button>

      <ErrorBoundary
        onRetry={() => setShouldCrash(false)}
        onError={(error) => console.error('Caught error:', error)}
      >
        <BuggyComponent shouldCrash={shouldCrash} />
      </ErrorBoundary>
    </div>
  );
};

// ============================================
// Full Demo Page
// ============================================

export const UIComponentsDemo: React.FC = () => {
  return (
    <div className="space-y-12 p-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">UI Components Demo</h1>
        <p className="text-slate-400">
          Interactive examples of all new UI components.
        </p>
      </div>

      <hr className="border-slate-700" />

      <ToastExamples />

      <hr className="border-slate-700" />

      <ModalExamples />

      <hr className="border-slate-700" />

      <AnimatedListExamples />

      <hr className="border-slate-700" />

      <ErrorBoundaryExample />

      <hr className="border-slate-700" />

      <EmptyStateExamples />
    </div>
  );
};

export default UIComponentsDemo;
