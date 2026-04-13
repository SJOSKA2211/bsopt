# Stage 1: Nginx Reverse Proxy
FROM nginx:stable-alpine

# Copy custom Nginx configuration for the reverse proxy
# This Dockerfile assumes the main Nginx configuration is in infrastructure/nginx/nginx.conf
# and potentially includes other specific configurations.
COPY infrastructure/nginx/nginx.conf /etc/nginx/nginx.conf
COPY infrastructure/nginx/global_lb.conf /etc/nginx/conf.d/global_lb.conf

# If a specific frontend Nginx config exists, copy it too.
# Based on previous findings, src/frontend/Dockerfile includes 'infrastructure/nginx/frontend.conf'.
# We assume here this Dockerfile is for a general proxy, possibly routing to multiple services.
# If this Nginx instance is *only* for the frontend, the frontend Dockerfile might be sufficient.
# For a general reverse proxy, it would route to different backend services.

# Expose port 80 for the reverse proxy
EXPOSE 80

# Default command to start Nginx
CMD ["nginx", "-g", "daemon off;"]
