import numpy as np
matrix = np.matrix([[np.nan, np.nan, 9, 1],
                    [3, np.nan, 7, np.nan],
                    [5, np.nan, np.nan, 10],
                    [np.nan, 2, np.nan, np.nan]])
print(matrix)
print(np.shape(matrix)[0])
users_count = np.shape(matrix)[0]
print(type(np.nan))

def maxnum(matrix_op):
    numbers = []
    for i in range(np.shape(matrix_op)[0]):
        for j in range(np.shape(matrix_op)[1]):
            if not np.isnan(matrix_op[i, j]):
                numbers.append(matrix_op[i, j])
    print(numbers)
    return max(numbers)


max_number = maxnum(matrix)
hidden_factors = int(input())
U_matrix = np.round(max_number * np.random.random_sample((np.shape(matrix)[0], hidden_factors)), 2)
print(U_matrix)

V_matrix = np.round(max_number * np.random.random_sample((hidden_factors, np.shape(matrix)[1])), 2)
print(V_matrix)

