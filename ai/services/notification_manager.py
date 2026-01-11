"""
Notification Manager Service
============================

Controls notification frequency with configurable cooldowns per message type.
Prevents notification spam by tracking when each type was last sent.

Usage:
    from .notification_manager import notification_manager

    # Only sends if cooldown has passed
    if notification_manager.can_send("gcp_budget_alert"):
        send_telegram_message(...)
        notification_manager.mark_sent("gcp_budget_alert")
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default cooldowns in seconds
DEFAULT_COOLDOWNS = {
    # GCP Budget Pub/Sub alerts - max once per day
    "gcp_budget_alert": 86400,  # 24 hours

    # Cost threshold alerts - once per threshold crossing
    "cost_threshold_50": 86400,
    "cost_threshold_80": 86400,
    "cost_threshold_95": 3600,   # 1 hour for critical
    "cost_threshold_100": 3600,  # 1 hour for exceeded

    # Daily reports - once per day
    "daily_report": 86400,

    # Fallback warnings - once per hour
    "ai_fallback_warning": 3600,

    # Error notifications - once per 15 minutes
    "error_notification": 900,

    # Default for unknown types
    "default": 3600,
}


class NotificationManager:
    """
    Singleton manager for controlling notification frequency.

    Features:
    - Configurable cooldowns per notification type
    - Persists state to JSON file
    - Thread-safe operations
    - Environment variable overrides
    """

    _instance: Optional["NotificationManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "NotificationManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Configuration
        self.data_dir = Path(os.getenv("COST_TRACKER_DATA_DIR", "/var/lib/api-ai"))
        self.data_file = self.data_dir / "notification_state.json"

        # Cooldowns (can be overridden via environment)
        self._cooldowns = DEFAULT_COOLDOWNS.copy()
        self._load_cooldown_overrides()

        # State: {notification_type: last_sent_iso_timestamp}
        self._last_sent: dict[str, str] = {}
        self._data_lock = threading.Lock()

        # Load persisted state
        self._load_state()

        logger.info(f"NotificationManager initialized with {len(self._cooldowns)} cooldown types")

    def _load_cooldown_overrides(self) -> None:
        """Load cooldown overrides from environment variables."""
        # Format: NOTIFICATION_COOLDOWN_<TYPE>=<seconds>
        # Example: NOTIFICATION_COOLDOWN_GCP_BUDGET_ALERT=43200
        for key, value in os.environ.items():
            if key.startswith("NOTIFICATION_COOLDOWN_"):
                notification_type = key[22:].lower()  # Remove prefix, lowercase
                try:
                    self._cooldowns[notification_type] = int(value)
                    logger.info(f"Cooldown override: {notification_type}={value}s")
                except ValueError:
                    logger.warning(f"Invalid cooldown value for {key}: {value}")

    def _load_state(self) -> None:
        """Load persisted notification state."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.data_file.exists():
                with open(self.data_file, "r") as f:
                    data = json.load(f)
                    self._last_sent = data.get("last_sent", {})
                logger.info(f"Loaded notification state: {len(self._last_sent)} entries")
        except Exception as e:
            logger.error(f"Failed to load notification state: {e}")
            self._last_sent = {}

    def _save_state(self) -> None:
        """Persist notification state to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "last_sent": self._last_sent,
                "updated_at": datetime.now().isoformat(),
            }
            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save notification state: {e}")

    def get_cooldown(self, notification_type: str) -> int:
        """Get cooldown in seconds for a notification type."""
        return self._cooldowns.get(notification_type, self._cooldowns["default"])

    def set_cooldown(self, notification_type: str, seconds: int) -> None:
        """Set cooldown for a notification type (runtime override)."""
        self._cooldowns[notification_type] = seconds
        logger.info(f"Set cooldown: {notification_type}={seconds}s")

    def can_send(self, notification_type: str) -> bool:
        """
        Check if a notification can be sent (cooldown has passed).

        Args:
            notification_type: Type of notification (e.g., "gcp_budget_alert")

        Returns:
            True if notification can be sent, False if still in cooldown
        """
        with self._data_lock:
            last_sent_str = self._last_sent.get(notification_type)

            if not last_sent_str:
                return True

            try:
                last_sent = datetime.fromisoformat(last_sent_str)
            except ValueError:
                # Invalid timestamp, allow sending
                return True

            cooldown_seconds = self.get_cooldown(notification_type)
            cooldown_delta = timedelta(seconds=cooldown_seconds)
            next_allowed = last_sent + cooldown_delta

            can_send = datetime.now() >= next_allowed

            if not can_send:
                remaining = (next_allowed - datetime.now()).total_seconds()
                logger.debug(
                    f"Notification '{notification_type}' in cooldown. "
                    f"Remaining: {remaining:.0f}s"
                )

            return can_send

    def mark_sent(self, notification_type: str) -> None:
        """
        Mark a notification as sent (updates last_sent timestamp).

        Args:
            notification_type: Type of notification
        """
        with self._data_lock:
            self._last_sent[notification_type] = datetime.now().isoformat()
            self._save_state()
            logger.info(f"Marked notification sent: {notification_type}")

    def get_last_sent(self, notification_type: str) -> Optional[datetime]:
        """Get when a notification type was last sent."""
        with self._data_lock:
            last_sent_str = self._last_sent.get(notification_type)
            if last_sent_str:
                try:
                    return datetime.fromisoformat(last_sent_str)
                except ValueError:
                    return None
            return None

    def get_next_allowed(self, notification_type: str) -> Optional[datetime]:
        """Get when the next notification of this type can be sent."""
        last_sent = self.get_last_sent(notification_type)
        if not last_sent:
            return datetime.now()  # Can send now

        cooldown_seconds = self.get_cooldown(notification_type)
        return last_sent + timedelta(seconds=cooldown_seconds)

    def reset(self, notification_type: str) -> None:
        """Reset cooldown for a specific notification type."""
        with self._data_lock:
            if notification_type in self._last_sent:
                del self._last_sent[notification_type]
                self._save_state()
                logger.info(f"Reset notification cooldown: {notification_type}")

    def reset_all(self) -> None:
        """Reset all notification cooldowns."""
        with self._data_lock:
            self._last_sent = {}
            self._save_state()
            logger.info("Reset all notification cooldowns")

    def get_status(self) -> dict:
        """Get current notification manager status."""
        with self._data_lock:
            status = {}
            for notification_type, last_sent_str in self._last_sent.items():
                try:
                    last_sent = datetime.fromisoformat(last_sent_str)
                    cooldown = self.get_cooldown(notification_type)
                    next_allowed = last_sent + timedelta(seconds=cooldown)
                    can_send = datetime.now() >= next_allowed

                    status[notification_type] = {
                        "last_sent": last_sent_str,
                        "cooldown_seconds": cooldown,
                        "next_allowed": next_allowed.isoformat(),
                        "can_send_now": can_send,
                    }
                except ValueError:
                    status[notification_type] = {
                        "last_sent": last_sent_str,
                        "error": "Invalid timestamp",
                    }

            return {
                "notification_types": status,
                "cooldown_config": self._cooldowns,
            }


# Global singleton instance
notification_manager = NotificationManager()
