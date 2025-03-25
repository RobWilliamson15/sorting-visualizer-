import matplotlib.pyplot as plt
import numpy as np
import random
import time


class SortingVisualizer:
    def __init__(self, array_size=100):
        """
        Initialize the Sorting Visualizer with configurable parameters

        :param array_size: Number of elements in the array to be sorted
        """
        self.array_size = array_size
        self.array = self.generate_array()

    def generate_array(self, method="random"):
        """
        Generate an array based on specified method

        :param method: Generation method ('random', 'nearly_sorted', 'reverse_sorted')
        :return: Numpy array of integers
        """
        if method == "random":
            return np.random.randint(1, 1000, self.array_size)
        elif method == "nearly_sorted":
            arr = np.arange(1, self.array_size + 1)
            for _ in range(self.array_size // 10):
                i, j = np.random.randint(0, self.array_size, 2)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        elif method == "reverse_sorted":
            return np.arange(self.array_size, 0, -1)

    def visualize_sorting(self, sorting_algorithm, title):
        """
        Visualize the sorting process for a given algorithm

        :param sorting_algorithm: Function that sorts the array
        :param title: Title of the visualization
        """
        # Create a copy of the original array to preserve it
        working_array = self.array.copy()

        # Set up the plot
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Value")

        # Sorting with visualization steps
        steps = []

        def update_plot(arr, current_step=None, compare_indices=None):
            plt.clf()
            plt.title(f"{title} - Step {len(steps)}")
            plt.xlabel("Index")
            plt.ylabel("Value")

            # Highlight comparison indices if provided
            bar_colors = ["blue"] * len(arr)
            if compare_indices:
                for idx in compare_indices:
                    bar_colors[idx] = "red"

            plt.bar(range(len(arr)), arr, color=bar_colors)
            plt.pause(0.01)

            # Store the step
            steps.append(arr.copy())

        # Monkey patch the sorting algorithm to capture visualization steps
        original_algorithm = sorting_algorithm

        def visualized_algorithm(arr):
            return original_algorithm(arr, update_plot)

        # Perform sorting with visualization
        start_time = time.time()
        sorted_array = visualized_algorithm(working_array)
        end_time = time.time()

        # Final sorted array
        plt.clf()
        plt.title(f"{title} - Sorted")
        plt.bar(range(len(sorted_array)), sorted_array, color="green")
        plt.xlabel("Index")
        plt.ylabel("Value")

        # Performance metrics
        print(f"\n{title} Performance:")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print(f"Total visualization steps: {len(steps)}")

        plt.tight_layout()
        plt.show()

    def bubble_sort(self, arr, update_plot=None):
        """
        Bubble Sort implementation with optional visualization

        :param arr: Array to be sorted
        :param update_plot: Callback for visualization
        :return: Sorted array
        """
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if update_plot:
                    update_plot(arr, current_step=i, compare_indices=[j, j + 1])

                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def quick_sort(self, arr, update_plot=None):
        """
        Quick Sort implementation with optional visualization

        :param arr: Array to be sorted
        :param update_plot: Callback for visualization
        :return: Sorted array
        """

        def _quick_sort(subarr, low, high):
            if low < high:
                pivot_index = _partition(subarr, low, high)
                _quick_sort(subarr, low, pivot_index - 1)
                _quick_sort(subarr, pivot_index + 1, high)
            return subarr

        def _partition(subarr, low, high):
            pivot = subarr[high]
            i = low - 1

            for j in range(low, high):
                if update_plot:
                    update_plot(subarr, compare_indices=[j, high])

                if subarr[j] <= pivot:
                    i += 1
                    subarr[i], subarr[j] = subarr[j], subarr[i]

            subarr[i + 1], subarr[high] = subarr[high], subarr[i + 1]
            return i + 1

        return _quick_sort(arr, 0, len(arr) - 1)

    def merge_sort(self, arr, update_plot=None):
        """
        Merge Sort implementation with optional visualization

        :param arr: Array to be sorted
        :param update_plot: Callback for visualization
        :return: Sorted array
        """

        def _merge_sort(subarr):
            if len(subarr) > 1:
                mid = len(subarr) // 2
                left_half = subarr[:mid].copy()
                right_half = subarr[mid:].copy()

                _merge_sort(left_half)
                _merge_sort(right_half)

                i = j = k = 0

                while i < len(left_half) and j < len(right_half):
                    if update_plot:
                        update_plot(subarr, compare_indices=[i, j])

                    if left_half[i] < right_half[j]:
                        subarr[k] = left_half[i]
                        i += 1
                    else:
                        subarr[k] = right_half[j]
                        j += 1
                    k += 1

                while i < len(left_half):
                    subarr[k] = left_half[i]
                    i += 1
                    k += 1

                while j < len(right_half):
                    subarr[k] = right_half[j]
                    j += 1
                    k += 1

            return subarr

        return _merge_sort(arr)

    def insertion_sort(self, arr, update_plot=None):
        """
        Insertion Sort implementation with optional visualization

        :param arr: Array to be sorted
        :param update_plot: Callback for visualization
        :return: Sorted array
        """
        # Traverse through 1 to len(arr)
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1

            # Move elements of arr[0..i-1], that are greater than key,
            # to one position ahead of their current position
            while j >= 0 and arr[j] > key:
                if update_plot:
                    update_plot(arr, compare_indices=[j, j + 1])
                arr[j + 1] = arr[j]
                j -= 1

            arr[j + 1] = key

            # Optionally update the plot after inserting the key
            if update_plot:
                update_plot(arr, compare_indices=[j + 1])

        return arr


def main():
    # Create visualizer
    visualizer = SortingVisualizer(array_size=50)

    # Different array generation methods
    generation_methods = ["random", "nearly_sorted", "reverse_sorted"]
    sorting_algorithms = [
        (visualizer.bubble_sort, "Bubble Sort"),
        (visualizer.quick_sort, "Quick Sort"),
        (visualizer.merge_sort, "Merge Sort"),
        (visualizer.insertion_sort, "Insertion Sort"),
    ]

    # Visualize sorting for different array types
    for method in generation_methods:
        visualizer.array = visualizer.generate_array(method)
        print(f"\n--- Sorting {method.replace('_', ' ').title()} Array ---")

        for algorithm, name in sorting_algorithms:
            visualizer.visualize_sorting(
                algorithm, f"{name} - {method.replace('_', ' ').title()} Array"
            )


if __name__ == "__main__":
    main()
