import numpy as np
matrix = np.matrix([[np.nan, np.nan, 9, 1],
                    [3, np.nan, 7, np.nan],
                    [5, np.nan, np.nan, 10],
                    [np.nan, 2, np.nan, np.nan]])


print(matrix)
print(np.shape(matrix)[0])



def maxnum(matrix_op):
    return max([j for i in matrix_op.tolist() for j in i if not np.isnan(j)])

def indexnums(matrix_op):
    return [(i, j) for i in range(len(matrix_op.tolist())) for j in range(len(matrix_op.tolist()[i])) if not np.isnan(matrix_op.tolist()[i][j])]

max_number = maxnum(matrix)
index_numbers = indexnums(matrix)
hidden_factors = int(input())
U_matrix = np.round(max_number * np.random.random_sample((np.shape(matrix)[0], hidden_factors)), 2)
print(U_matrix)

V_matrix = np.round(max_number * np.random.random_sample((hidden_factors, np.shape(matrix)[1])), 2)
print(V_matrix)

