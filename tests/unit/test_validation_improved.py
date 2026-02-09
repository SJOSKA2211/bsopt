import unittest

import numpy as np

from src.ml.utils.validation import WalkForwardValidator


class TestValidation(unittest.TestCase):
    def test_expanding_window(self):
        X = np.arange(100)
        validator = WalkForwardValidator(n_splits=4)
        splits = list(validator.split(X))
        self.assertEqual(len(splits), 4)
        
        # Check first split
        train, test = splits[0]
        self.assertLess(train[-1], test[0])
        
        # Check last split
        train, test = splits[-1]
        self.assertEqual(test[-1], 99)

    def test_sliding_window(self):
        X = np.arange(100)
        validator = WalkForwardValidator(n_splits=3, test_size=10)
        splits = list(validator.split(X))
        self.assertEqual(len(splits), 3)
        
        # In sliding window, test size should be 10
        train, test = splits[0]
        self.assertEqual(len(test), 10)

if __name__ == '__main__':
    unittest.main()
