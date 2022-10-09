import itertools
from typing import Iterator, Iterable, Optional, List, Any

def partition(iterable: Iterable, size : int, start: Optional[int] =0) -> Iterator[List[Any]]:
    """Partitions the iterable into chunks of length size
    Args:
        iterable : Any iterables that needs to be partioned 
        size : size of each chunk
        start : index on where to start the split
    Yields:
        when start > 0,
            the first chunk is from 0 to start -1
            succeeding ones, chunk from starts to end, each with size
        when start = 0,
            chunk from 0 to end, each with size
    Examples:
        >>> children = partition([1, 2, 3, 4, 5, 6], 2)
        >>> [list(child) for child in children]
        [[1, 2], [3, 4], [5, 6]]

        When length of iterable is not divisible by size
        >>> children = partition([1, 2, 3, 4, 5, 6], 4)
        >>> [list(child) for child in children]
        [[1, 2, 3, 4], [5, 6]]

        When the start = 3:
        >>> children = partition([1, 2, 3, 4, 5, 6], 2, start=3)
        >>> [list(child) for child in children]
        [[1, 2, 3], [4, 5], [6]]

    """
    assert start >= 0, "start should be a nonnegative integer"

    try:
        iterable[:0]
    except TypeError:
        seq = list(iterable)
    else:
        seq = iterable

    if start:
        yield itertools.islice(seq, 0, start)

    for i in range(start, len(seq), size):
        yield itertools.islice(seq, i, i + size)



def divide(n: int, iterable: Iterable) -> List[Iterable]:
    """Divide the elements from *iterable* into *n* parts, maintaining
    order.

        >>> group_1, group_2 = divide(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 2, 3]
        >>> list(group_2)
        [4, 5, 6]

    If the length of *iterable* is not evenly divisible by *n*, then the
    length of the returned iterables will not be identical:

        >>> children = divide(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 2, 3], [4, 5], [6, 7]]

    If the length of the iterable is smaller than n, then the last returned
    iterables will be empty:

        >>> children = divide(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]

    This function will exhaust the iterable before returning and may require
    significant storage. If order is not important, see :func:`distribute`,
    which does not first pull the iterable into memory.
    Source:
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#divide
    """
    if n < 1:
        raise ValueError('n must be at least 1')

    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable

    q, r = divmod(len(seq), n)

    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(iter(seq[start:stop]))

    return ret
