def reverse(S, start, stop):
    if start < stop - 1:
        S[start], S[stop - 1] = S[stop - 1], S[start]
        print(S)
        reverse(S, start + 1, stop - 1)


if __name__ == "__main__":
    reverse([2, 4, 5, 7, 8, 9, 12, 14, 17, 19, 22, 25, 27, 28, 33, 37], 0, 16)