from .core.hashing import PasswordHasherService


class PasswordService(PasswordHasherService):
    """Shim for PasswordService."""
    pass

class PasswordValidator:
    """Shim for PasswordValidator."""
    def __init__(self, min_length: int = 8):
        self.min_length = min_length
    
    def validate(self, password: str) -> bool:
        return len(password) >= self.min_length
