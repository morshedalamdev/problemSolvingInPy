# Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
# The overall run time complexity should be O(log (m+n)).

from typing import List
import math

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        merged_arr = sorted(nums1 + nums2)
        arr_len = len(merged_arr)
        median_index = math.floor(arr_len / 2)
        if arr_len % 2 == 0:
            median_num = (merged_arr[median_index] + merged_arr[median_index - 1]) / 2
        else:
            median_num = merged_arr[median_index]
        return float(median_num)

sol = Solution();

result1 = sol.findMedianSortedArrays([1, 3], [2]);
result1 = sol.findMedianSortedArrays([1, 2], [3, 4]);
result1 = sol.findMedianSortedArrays([0, 0], [0, 0]);
result1 = sol.findMedianSortedArrays([2, 5, 6], [1, 3, 4, 7]);
result1 = sol.findMedianSortedArrays([1, 2, 3], [4, 5, 6, 7, 8]);