# =============================================================================
# Unified Frontend Dockerfile
# =============================================================================
# Usage:
#   Production (default): docker build -t frontend .
#   Development:          docker build --target dev -t frontend-dev .
# =============================================================================

# -----------------------------------------------------------------------------
# Development target - Vite dev server with hot reload
# -----------------------------------------------------------------------------
FROM node:20-alpine AS dev

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

# -----------------------------------------------------------------------------
# Build stage - compiles TypeScript and bundles assets
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# -----------------------------------------------------------------------------
# Production target (default) - nginx serving static files
# -----------------------------------------------------------------------------
FROM nginx:alpine AS production

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
