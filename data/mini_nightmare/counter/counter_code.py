import random
import threading
import time


class ThreadSafeCounter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
        self._active_threads = 0
        self._thread_local = threading.local()
        self._increment_history = []  # For debugging purposes

    def increment(self):
        # Simulate varying work loads
        time.sleep(random.uniform(0.001, 0.003))
        
        with self._lock:
            self._active_threads += 1
            self._thread_local.last_value = self._count

        time.sleep(random.uniform(0.001, 0.003))
        local_value = self._thread_local.last_value
        new_value = local_value + 1
        
        with self._lock:
            # Only update if no other thread has modified the value
            if self._count == local_value:
                self._count = new_value
                self._increment_history.append(
                    (threading.get_ident(), local_value, new_value)
                )
            self._active_threads -= 1

    def get_count(self) -> int:
        with self._lock:
            return self._count

    def get_active_threads(self) -> int:
        with self._lock:
            return self._active_threads

    def get_increment_history(self):
        with self._lock:
            return self._increment_history.copy()


def increment_counter(counter: ThreadSafeCounter, num_increments: int):
    for _ in range(num_increments):
        counter.increment()
