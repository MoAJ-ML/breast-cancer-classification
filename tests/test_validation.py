import unittest
from validate_model import load_wdbc, split_data


class WDBCTests(unittest.TestCase):
    def test_unique_patients_and_disjoint_splits(self):
        X, y = load_wdbc()
        self.assertEqual(set(y.unique()), {0, 1})
        train, val, test = split_data(X, y)
        self.assertFalse(set(train[0].index) & set(val[0].index))
        self.assertFalse(set(train[0].index) & set(test[0].index))
        self.assertFalse(set(val[0].index) & set(test[0].index))


if __name__ == '__main__':
    unittest.main()
