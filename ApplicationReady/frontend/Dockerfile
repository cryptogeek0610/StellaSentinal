# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

ARG VITE_TENANT_ID
ARG VITE_API_KEY
ARG VITE_USER_ID
ARG VITE_USER_ROLE

ENV VITE_TENANT_ID=$VITE_TENANT_ID
ENV VITE_API_KEY=$VITE_API_KEY
ENV VITE_USER_ID=$VITE_USER_ID
ENV VITE_USER_ROLE=$VITE_USER_ROLE

# Copy package files first for better layer caching
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
