import os

def test_anti_freeze_docs_exist():
    """Verify that the Anti-Freeze documentation file exists."""
    doc_path = os.path.join("docs", "mlops", "anti-freeze.md")
    assert os.path.exists(doc_path), f"Documentation file not found at {doc_path}"

def test_anti_freeze_content():
    """Verify that the Anti-Freeze documentation contains key sections."""
    doc_path = os.path.join("docs", "mlops", "anti-freeze.md")
    with open(doc_path, "r") as f:
        content = f.read()
    
    assert "COMPOSE_PARALLEL_LIMIT" in content
    assert "remote-builder" in content
    assert "limit" in content.lower()
