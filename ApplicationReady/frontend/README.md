# Stella Sentinel Frontend

Modern web dashboard for enterprise device anomaly detection and investigation.

## Tech Stack

- **React 18** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS** for styling
- **React Router** for navigation
- **TanStack Query** for data fetching
- **Recharts** for data visualization

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn

### Installation

```bash
cd frontend
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

Make sure the FastAPI backend is running on `http://localhost:8000` (or update the proxy in `vite.config.ts`).

### Build

Build for production:

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── api/          # API client functions
│   ├── components/   # Reusable UI components
│   ├── hooks/       # Custom React hooks
│   ├── pages/        # Page components
│   ├── types/        # TypeScript type definitions
│   ├── App.tsx       # Main app component with routing
│   └── main.tsx      # Entry point
├── package.json
└── vite.config.ts
```

## Features

- **Dashboard**: Overview with KPIs and anomaly trends
- **Anomaly List**: Filterable, sortable list of all anomalies
- **Anomaly Detail**: Deep investigation view with charts and notes
- **Device Detail**: View all anomalies for a specific device

## API Integration

The frontend communicates with the FastAPI backend via REST API. The API base URL is configured in `src/api/client.ts` and defaults to `/api` (proxied to `http://localhost:8000` in development).

