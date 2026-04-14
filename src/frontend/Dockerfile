# Stage 1: Build Frontend Application
FROM node:22-alpine as frontend_builder

WORKDIR /app/frontend

# Copy dependencies first for better caching
COPY package.json package-lock.json ./

# Install dependencies with cache mount
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# Copy the rest of the frontend source code
COPY . .

# Build the frontend application for production
RUN npm run build

# Stage 2: Serve Frontend Application with Nginx
# Use a lightweight Nginx image based on Alpine
FROM nginx:stable-alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy the optimized build output from the builder stage
# Assuming the build output is in a 'dist' directory
COPY --from=frontend_builder /app/frontend/dist /usr/share/nginx/html

# Copy custom Nginx configuration for the frontend
# This assumes a specific Nginx config file exists or needs to be created.
# For now, we'll use the default Nginx config but could replace it.
COPY infrastructure/nginx/frontend.conf /etc/nginx/conf.d/default.conf
# If infrastructure/nginx/frontend.conf doesn't exist, a default one might be needed,
# or Nginx can be configured via nginx.conf. For now, assuming it might exist or be created.

# Expose port 80 for the frontend
EXPOSE 80

# Default command to start Nginx
CMD ["nginx", "-g", "daemon off;"]
