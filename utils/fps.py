"""
FPS Counter helper utility for real-time video processing performance tracking.
"""

import time
from collections import deque

class FPSCounter:
    """Calculates smoothed frames per second (FPS) over a rolling window."""
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.timestamps = deque(maxlen=buffer_size)

    def update(self) -> float:
        """Call on each new frame. Returns current calculated FPS."""
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) <= 1:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def get_fps(self) -> float:
        """Returns current FPS without updating timestamp."""
        if len(self.timestamps) <= 1:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed
