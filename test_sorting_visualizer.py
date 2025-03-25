import unittest
import numpy as np
from sorting import SortingVisualizer  # Adjust import as necessary


class TestSortingAlgorithms(unittest.TestCase):
    def setUp(self):
        self.visualizer = SortingVisualizer(array_size=20)
        # A fixed test array for consistency
        self.test_array = np.array([5, 3, 8, 1, 9, 2, 7, 4, 6])

    def test_bubble_sort(self):
        sorted_array = self.visualizer.bubble_sort(
            self.test_array.copy(), update_plot=None
        )
        np.testing.assert_array_equal(np.sort(self.test_array), sorted_array)

    def test_quick_sort(self):
        sorted_array = self.visualizer.quick_sort(
            self.test_array.copy(), update_plot=None
        )
        np.testing.assert_array_equal(np.sort(self.test_array), sorted_array)

    def test_merge_sort(self):
        sorted_array = self.visualizer.merge_sort(
            self.test_array.copy(), update_plot=None
        )
        np.testing.assert_array_equal(np.sort(self.test_array), sorted_array)

    def test_insertion_sort(self):
        # Assuming you've added the insertion_sort method
        sorted_array = self.visualizer.insertion_sort(
            self.test_array.copy(), update_plot=None
        )
        np.testing.assert_array_equal(np.sort(self.test_array), sorted_array)


if __name__ == "__main__":
    unittest.main()
