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


def indexnums(matrix_op):
    return [(i, j) for i in range(len(matrix_op.tolist())) for j in range(len(matrix_op.tolist()[i])) if not np.isnan(matrix_op.tolist()[i][j])]


def quad_err(x_actual, x_predicted):
    return (x_actual - x_predicted)**2


def predict(row, column):
    return


print(f'Исходная матрица:\n{matrix}')

while True:
    mode = int(input('Выберите режим:0-для теста по методичке, 1-для штатной работы'))
    if mode == 0:
        U_matrix, V_matrix = test_matrix()
        break
    elif mode == 1:
        max_number = maxnum(matrix)
        hidden_factors = int(input('Введите количество скрытых факторов:'))
        U_matrix, V_matrix = generate_matrix(hidden_factors, max_number)

        break
    else:
        print('!!!Введены некорректные значения!!!')


print(matrix)
print(np.shape(matrix)[0])

max_number = maxnum(matrix)
index_numbers = indexnums(matrix)
print(index_numbers)
print(type(np.nan))
print(U_matrix)
print()
print(V_matrix)
print(quad_err(9, -0.07))
