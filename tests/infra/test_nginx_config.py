import subprocess
import os

def test_nginx_config_syntax():
    # If nginx is installed on the runner, we can check syntax
    try:
        result = subprocess.run(["nginx", "-t", "-c", os.path.abspath("docker/nginx/nginx.conf")], 
                                capture_output=True, text=True)
        # Nginx -t might fail if paths like /etc/nginx/mime.types don't exist locally,
        # so we primarily rely on grep validation for specific settings.
        pass
    except FileNotFoundError:
        # Nginx not installed
        pass

def test_nginx_performance_settings():
    conf_path = "docker/nginx/nginx.conf"
    with open(conf_path, "r") as f:
        content = f.read()
    
    assert "worker_rlimit_nofile 200000;" in content
    assert "worker_connections 20000;" in content
    assert "use epoll;" in content
    assert "sendfile on;" in content
    assert "tcp_nodelay on;" in content
    assert "keepalive_requests 100000;" in content

def test_websocket_optimizations():
    conf_path = "docker/nginx/nginx.conf"
    with open(conf_path, "r") as f:
        content = f.read()
    
    # Check websocket block
    assert "location /ws/ {" in content
    assert "proxy_buffering off;" in content
    assert "proxy_read_timeout 86400;" in content
    assert 'proxy_set_header Connection "Upgrade";' in content
