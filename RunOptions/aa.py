# coding=utf-8
import sys

# str = input()
# print(str)
print('Hello,World!')


def find_prime_number(m, n):
    result = []
    m_ind = 0
    prime_numbers = [2]

    if m <= 1 or n <= 1 or m >= n:
        return result

    for num in range(2, n+1):
        is_prime = True

        for pn in prime_numbers:
            if num % pn == 0:
                is_prime = False
                break

        if is_prime:
            prime_numbers.append(num)

            if m_ind == 0 and num >= m != 2:
                m_ind = len(prime_numbers) - 1

    result.extend(prime_numbers[m_ind:])

    return result


print(find_prime_number(2, 100))


def fine_common_substring(A, B):
    len_A = len(A)
    len_B = len(B)

    if len_A <=0 or len_B <= 0:
        return ""

    cache = [[0 for i in range(len_B + 1)] for j in range(len_A + 1)]
    max_length = 0
    ind = 0

    for i in range(len_A):
        for j in range(len_B):
            if A[i] == B[j]:
                cache[i+1][j+1] = cache[i][j] + 1

                if cache[i+1][j+1] > max_length:
                    max_length = cache[i+1][j+1]
                    ind = i + 1

    return A[ind - max_length: ind]


print(fine_common_substring("abbac", "dsarabbadewq"))
