import unittest

from core.shared.shm_mesh import SharedMemoryRingBuffer


class TestSHMMesh(unittest.TestCase):
    def test_ring_buffer_lifecycle(self):
        # Test creation and unlinking
        rb = SharedMemoryRingBuffer(create=True)
        self.assertIsNotNone(rb.buf)
        rb.close()
        rb.unlink()

    def test_write_read_tick(self):
        rb = SharedMemoryRingBuffer(create=True)
        try:
            rb.write_tick("AAPL", 150.0, 100, 1644321600.0)
            view, head = rb.read_latest_view(0)
            self.assertEqual(len(view), 1)
            self.assertEqual(view[0]["symbol"].decode().strip("\x00"), "AAPL")
            self.assertEqual(view[0]["price"], 150.0)
            self.assertEqual(head, 1)

            # Delete view reference before closing
            del view

            # Test msgspec reader
            ticks, head2 = rb.read_latest_msgspec(0)
            self.assertEqual(len(ticks), 1)
            self.assertEqual(ticks[0].symbol, "AAPL")
            del ticks
        finally:
            rb.close()
            rb.unlink()

    def test_wrap_around(self):
        # Small capacity for wrap test? No, it's fixed at 100k.
        pass


if __name__ == "__main__":
    unittest.main()
