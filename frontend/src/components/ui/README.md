# UI Components Usage Guide

This guide shows how to use the new UI components added to the Stella Sentinel design system.

## Table of Contents

- [EmptyState](#emptystate)
- [Toast Notifications](#toast-notifications)
- [Modal & ConfirmDialog](#modal--confirmdialog)
- [AnimatedList](#animatedlist)
- [ErrorBoundary](#errorboundary)

---

## EmptyState

Use `EmptyState` when there's no data to display or when user action is required.

### Basic Usage

```tsx
import { EmptyState } from './components/ui';

// No data state
<EmptyState
  variant="no-data"
  title="No anomalies detected"
  description="Your fleet is operating normally. Check back later for updates."
/>

// No search results
<EmptyState
  variant="no-results"
  title="No results found"
  description="Try adjusting your search terms or filters."
  action={{
    label: 'Clear filters',
    onClick: () => clearFilters(),
  }}
/>

// Not configured
<EmptyState
  variant="not-configured"
  title="Connection not configured"
  description="Set up your database connection to start analyzing data."
  action={{
    label: 'Configure now',
    onClick: () => navigate('/setup'),
  }}
  secondaryAction={{
    label: 'Learn more',
    onClick: () => openDocs(),
  }}
/>

// Error state
<EmptyState
  variant="error"
  title="Failed to load data"
  description="There was a problem fetching the device list."
  action={{
    label: 'Try again',
    onClick: () => refetch(),
  }}
/>

// Success state
<EmptyState
  variant="success"
  title="All issues resolved"
  description="Great work! There are no pending issues to address."
/>
```

### Variants

| Variant | Use Case |
|---------|----------|
| `no-data` | Empty tables, lists with no items |
| `no-results` | Search/filter returned nothing |
| `error` | Failed API calls, errors |
| `not-configured` | Feature needs setup |
| `success` | All tasks complete, no issues |

---

## Toast Notifications

Use toasts for transient feedback messages that auto-dismiss.

### Setup

Wrap your app with `ToastProvider`:

```tsx
// In App.tsx or main.tsx
import { ToastProvider } from './components/ui';

function App() {
  return (
    <ToastProvider>
      <Router>
        {/* Your routes */}
      </Router>
    </ToastProvider>
  );
}
```

### Usage in Components

```tsx
import { useToast } from './components/ui';

function MyComponent() {
  const toast = useToast();

  const handleSave = async () => {
    try {
      await saveData();
      toast.success('Settings saved', 'Your changes have been applied.');
    } catch (error) {
      toast.error('Save failed', 'Please try again later.');
    }
  };

  const handleWarning = () => {
    toast.warning('Connection unstable', 'Some features may be slow.');
  };

  const handleInfo = () => {
    toast.info('New update available', 'Refresh to see the latest changes.');
  };

  // Advanced: custom toast with action
  const handleCustom = () => {
    toast.addToast({
      type: 'info',
      title: 'Export ready',
      description: 'Your report is ready for download.',
      duration: 10000, // 10 seconds
      action: {
        label: 'Download',
        onClick: () => downloadReport(),
      },
    });
  };

  return (
    <button onClick={handleSave}>Save Settings</button>
  );
}
```

### Toast Types

| Method | Duration | Use Case |
|--------|----------|----------|
| `success()` | 5s | Successful operations |
| `error()` | 8s | Failures (longer for reading) |
| `warning()` | 5s | Cautions, degraded states |
| `info()` | 5s | General information |

---

## Modal & ConfirmDialog

Use `Modal` for dialogs, forms, and detail views. Use `ConfirmDialog` for simple confirmations.

### Basic Modal

```tsx
import { useState } from 'react';
import { Modal } from './components/ui';

function MyComponent() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button onClick={() => setIsOpen(true)}>Open Settings</button>

      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Settings"
        description="Configure your preferences"
        size="md"
        footer={
          <>
            <button className="btn-secondary" onClick={() => setIsOpen(false)}>
              Cancel
            </button>
            <button className="btn-primary" onClick={handleSave}>
              Save
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <label className="block">
            <span className="text-sm text-slate-400">Name</span>
            <input type="text" className="input-stellar mt-1" />
          </label>
          {/* More form fields */}
        </div>
      </Modal>
    </>
  );
}
```

### ConfirmDialog

```tsx
import { useState } from 'react';
import { ConfirmDialog } from './components/ui';

function DeleteButton({ onDelete }) {
  const [isOpen, setIsOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleConfirm = async () => {
    setIsDeleting(true);
    await onDelete();
    setIsDeleting(false);
    setIsOpen(false);
  };

  return (
    <>
      <button className="btn-danger" onClick={() => setIsOpen(true)}>
        Delete
      </button>

      <ConfirmDialog
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        onConfirm={handleConfirm}
        title="Delete this item?"
        description="This action cannot be undone. All associated data will be permanently removed."
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="danger"
        isLoading={isDeleting}
      />
    </>
  );
}
```

### Modal Sizes

| Size | Width | Use Case |
|------|-------|----------|
| `sm` | 384px | Simple confirmations |
| `md` | 448px | Forms, settings |
| `lg` | 512px | Complex forms |
| `xl` | 576px | Multi-column content |
| `full` | 896px | Data tables, detailed views |

---

## AnimatedList

Use `AnimatedList` for staggered entrance animations on lists.

### Basic Usage

```tsx
import { AnimatedList } from './components/ui';

function DeviceList({ devices }) {
  return (
    <AnimatedList
      className="space-y-2"
      animation="slide-up"
      staggerDelay={0.05}
    >
      {devices.map((device) => (
        <div key={device.id} className="stellar-card p-4">
          <h3>{device.name}</h3>
          <p className="text-slate-400">{device.status}</p>
        </div>
      ))}
    </AnimatedList>
  );
}
```

### Individual Items

```tsx
import { AnimatedListItem } from './components/ui';

function StatsGrid({ stats }) {
  return (
    <div className="grid grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <AnimatedListItem
          key={stat.label}
          delay={index * 0.1}
          animation="scale"
        >
          <div className="stellar-card p-4 text-center">
            <div className="text-2xl font-bold">{stat.value}</div>
            <div className="text-sm text-slate-400">{stat.label}</div>
          </div>
        </AnimatedListItem>
      ))}
    </div>
  );
}
```

### Animation Types

| Animation | Effect |
|-----------|--------|
| `fade` | Simple opacity fade |
| `slide-up` | Fade + slide from bottom |
| `slide-right` | Fade + slide from left |
| `scale` | Fade + scale up |

---

## ErrorBoundary

Use `ErrorBoundary` to catch and handle errors gracefully.

### Wrap Components

```tsx
import { ErrorBoundary } from './components/ui';

function Dashboard() {
  return (
    <div className="grid grid-cols-2 gap-4">
      <ErrorBoundary>
        <AnomalyChart />
      </ErrorBoundary>

      <ErrorBoundary>
        <DeviceList />
      </ErrorBoundary>
    </div>
  );
}
```

### With Custom Fallback

```tsx
<ErrorBoundary
  fallback={
    <div className="text-center p-8">
      <p className="text-red-400">Widget failed to load</p>
      <button className="btn-secondary mt-2">Retry</button>
    </div>
  }
  onError={(error, info) => {
    // Log to error tracking service
    trackError(error, info);
  }}
>
  <ComplexWidget />
</ErrorBoundary>
```

### Higher-Order Component

```tsx
import { withErrorBoundary } from './components/ui';

const SafeChart = withErrorBoundary(AnomalyChart, {
  onError: (error) => console.error('Chart error:', error),
});

// Use like a regular component
<SafeChart data={data} />
```

---

## Accessibility Features

All components include accessibility support:

- **Keyboard navigation**: Tab, Enter, Space, Escape
- **ARIA attributes**: roles, labels, live regions
- **Focus management**: Focus trapping in modals
- **Reduced motion**: Respects `prefers-reduced-motion`
- **Screen reader**: Announcements for state changes

---

## Theme Support

Components automatically adapt to light/dark mode via CSS variables.

To toggle the theme programmatically:

```tsx
import { useTheme } from '../hooks/useTheme';

function ThemeToggle() {
  const { theme, toggleTheme, resolvedTheme } = useTheme();

  return (
    <button onClick={toggleTheme}>
      {resolvedTheme === 'dark' ? '☀️ Light' : '🌙 Dark'}
    </button>
  );
}
```
