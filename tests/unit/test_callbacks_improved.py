import unittest

from services.ml.callbacks import EarlyStopping


class TestCallbacks(unittest.TestCase):
    def test_early_stopping(self):
        cb = EarlyStopping(patience=3, min_delta=0.01)

        # Improve
        cb(0.9)
        self.assertEqual(cb.best_loss, 0.9)
        self.assertFalse(cb.early_stop)

        cb(0.8)
        self.assertEqual(cb.best_loss, 0.8)

        # Stagnate
        cb(0.8)
        self.assertEqual(cb.counter, 1)
        cb(0.8)
        self.assertEqual(cb.counter, 2)
        cb(0.8)
        self.assertTrue(cb.early_stop)


if __name__ == "__main__":
    unittest.main()
