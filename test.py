class Solution:
    def reverseBits(self, n: int) -> int:
        binary_str = format(n, '032b')
        
        reverse_binary = binary_str[::-1]
        return int(reverse_binary, 2)
    

sol = Solution()

result1 = sol.reverseBits(43261596)
print(result1)

result2 = sol.reverseBits(1)
print(result2)

result3 = sol.reverseBits(4294967293)
print(result3)