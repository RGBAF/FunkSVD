import numpy as np

matrix = np.matrix([[np.nan, np.nan, 9, 1],
                    [3, np.nan, 7, np.nan],
                    [5, np.nan, np.nan, 10],
                    [np.nan, 2, np.nan, np.nan]])


def test_matrix():
    U = np.matrix([[0.8, 1.2, -0.2], [2, 1.8, 0.4], [0.8, 3, 0.1], [1, 0.8, 2.4]])
    V = np.matrix([[-1.8, 1, -0.2, 1], [0.5, 1.2, 0.1, 5], [1.4, 4, 0.14, 2]])
    return U, V


def generate_matrix(h_f, m_n):
    U = np.round(m_n * np.random.random_sample((np.shape(matrix)[0], h_f)), 2)
    V = np.round(m_n * np.random.random_sample((h_f, np.shape(matrix)[1])), 2)
    return U, V


def maxnum(matrix_op):
    return max([j for i in matrix_op.tolist() for j in i if not np.isnan(j)])


print(f'Исходная матрица:\n{matrix}')
hidden_factors = int(input('Введите количество скрытых факторов:'))
max_number = maxnum(matrix)

while True:
    mode = int(input('Выберите режим:0-для теста по методичке, 1-для штатной работы'))
    if mode == 0:
        U_matrix, V_matrix = test_matrix()
        break
    elif mode == 1:
        U_matrix, V_matrix = generate_matrix(hidden_factors, max_number)
        break
    else:
        print('!!!Введены некорректные значения!!!')


print(matrix)
print(np.shape(matrix)[0])
users_count = np.shape(matrix)[0]
print(type(np.nan))
print(U_matrix)
print()
print(V_matrix)

print(max_number)
