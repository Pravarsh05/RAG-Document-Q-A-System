"""Re-export settings from app.core.config for backward compatibility."""
from app.core.config import Settings, settings

__all__ = ["Settings", "settings"]
