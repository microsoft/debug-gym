import threading
import unittest
from typing import List

from counter_code import ThreadSafeCounter, increment_counter


class TestThreadSafeCounter(unittest.TestCase):
    def test_single_threaded(self):
        counter = ThreadSafeCounter()
        increment_counter(counter, 10)
        self.assertEqual(counter.get_count(), 10)

    def test_multi_threaded(self):
        counter = ThreadSafeCounter()
        num_threads = 10  # Increased thread count
        increments_per_thread = 50  # Increased iterations
        expected_total = num_threads * increments_per_thread

        threads: List[threading.Thread] = []
        for _ in range(num_threads):
            thread = threading.Thread(
                target=increment_counter,
                args=(counter, increments_per_thread)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        final_count = counter.get_count()
        
        # If test fails, print debugging information
        if final_count != expected_total:
            print("\nTest failed! Debugging information:")
            print(f"Expected: {expected_total}, Got: {final_count}")
            print("\nIncrement History:")
            history = counter.get_increment_history()
            for thread_id, old_val, new_val in history:
                print(f"Thread {thread_id}: {old_val} -> {new_val}")
            
        self.assertEqual(
            final_count,
            expected_total,
            f"Expected {expected_total} but got {final_count}"
        )

    def test_no_active_threads_after_completion(self):
        counter = ThreadSafeCounter()
        num_threads = 5
        increments_per_thread = 20

        threads: List[threading.Thread] = []
        for _ in range(num_threads):
            thread = threading.Thread(
                target=increment_counter,
                args=(counter, increments_per_thread)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(counter.get_active_threads(), 0)


if __name__ == '__main__':
    unittest.main()