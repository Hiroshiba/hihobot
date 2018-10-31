from typing import List

import numpy as np


class Transformer(object):
    def __init__(self, chars: List[str]):
        self.chars = chars

        self._char_to_index = {c: i for i, c in enumerate(chars)}
        self._end_id = len(chars)
        self._num_chars = len(chars)
        self._num_id = len(chars) + 1
        self._start_array = np.zeros(self._num_chars, dtype=np.float32)

    def to_char_id(self, char: str):
        return self._char_to_index[char]

    def to_array(self, char_id: int):
        array = np.zeros(self._num_chars, dtype=np.float32)
        array[char_id] = 1
        return array

    def unshift_start_array(self, array: np.ndarray):
        """
        :param array: shape (length, num_id)
        :return: shape (length+1, num_id)
        """
        st = self._start_array[np.newaxis]  # shape (1, num_id)
        return np.concatenate([st, array], axis=0)  # shape (length+1, num_id)

    def push_end_id(self, char_ids: List[int]):
        return char_ids + [self._end_id]
