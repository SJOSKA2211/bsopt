import subprocess


def get_sysctl(key):
    try:
        output = subprocess.check_output(["sysctl", "-n", key]).decode().strip()
        return output
    except Exception:
        return None

def test_file_max_limit():
    # C100k requires significantly higher file descriptor limits
    file_max = get_sysctl("fs.file-max")
    assert file_max is not None
    assert int(file_max) >= 200000, f"fs.file-max should be at least 200,000, got {file_max}"

def test_tcp_tw_reuse():
    # Allows faster recycling of sockets in TIME_WAIT state
    tw_reuse = get_sysctl("net.ipv4.tcp_tw_reuse")
    assert tw_reuse in ["1", "2"], f"net.ipv4.tcp_tw_reuse should be 1 or 2, got {tw_reuse}"

def test_somaxconn():
    # Prevents connection drops during traffic spikes
    somaxconn = get_sysctl("net.core.somaxconn")
    assert somaxconn is not None
    assert int(somaxconn) >= 4096, f"net.core.somaxconn should be at least 4096, got {somaxconn}"

def test_ulimit_nofile():
    # Check current process soft limit for open files
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft >= 65535, f"ulimit nofile soft limit should be at least 65535, got {soft}"
