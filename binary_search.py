def binary_search(data, target, low, high):
    if low > high:
        return False
    else:
        mid = low + high
        if target == data[mid]:
            return True
        elif target < data[mid]:
            return binary_search(data, target, low, mid - 1)
        else:
            return binary_search(data, target, mid + 1, high)


if __name__ == "__main__":
    output = binary_search(
        [2, 4, 5, 7, 8, 9, 12, 14, 17, 19, 22, 25, 27, 28, 33, 37], 22, 0, 15
    )
    print(output)
