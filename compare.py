import numpy as np
import doctest
from time import perf_counter
from scipy.optimize import root
import matplotlib.pyplot as plt


def measure_time(func, *args, **kwargs):
    start = perf_counter()
    func(*args, **kwargs)
    end = perf_counter()
    return end - start


def random_check(size):
    """
    Compares between the result given by numpy.linalg.solve and scipy.optimize.root

    >>> random_check(3)
    True
    """
    a = np.random.randn(size, size)
    b = np.random.randn(size)

    return np.allclose(solver(a, b), np.linalg.solve(a, b))


def solver(a, b):
    """
    Solves the equation ax = b.
    :param a: numpy array (Matrix)
    :param b: numpy array (Vector)
    :return: solution to ax = b

    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([5, 11])
    >>> result = solver(a, b)
    >>> np.allclose(result, [1,2])
    True
    """
    return root(lambda x: a @ x - b, np.zeros(len(b))).x


def plot_performance(size, rep):
    sizes = sorted(set(np.random.randint(1, 1001, size)))
    plt.plot(sizes,
             [np.mean(list(measure_time(solver, np.random.randn(i, i), np.random.randn(i)) for _ in range(rep))) for i
              in sizes], label="scipy")
    plt.plot(sizes, [
        np.mean(list(measure_time(np.linalg.solve, np.random.randn(i, i), np.random.randn(i)) for _ in range(rep))) for
        i in sizes],
             label="numpy")
    plt.xlabel("Input length")
    plt.ylabel("Average time")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison.png")


if __name__ == '__main__':
    plot_performance(500, 10)
    print(doctest.testmod())
