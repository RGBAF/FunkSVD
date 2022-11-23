import numpy as np
matrix = np.matrix([[np.nan, np.nan, 9, 1],
                    [3, np.nan, 7, np.nan],
                    [5, np.nan, np.nan, 10],
                    [np.nan, 2, np.nan, np.nan]])
print(matrix)
print(np.shape(matrix)[0])
users_count = np.shape(matrix)[0]
print(type(users_count))
hidden_factors = int(input())

U_matrix = np.round(5 * np.random.random_sample((users_count, hidden_factors)), 2)
print(U_matrix)