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
    return [(i, j)
            for i in range(len(matrix_op.tolist()))
            for j in range(len(matrix_op.tolist()[i]))
            if not np.isnan(matrix_op.tolist()[i][j])]


def quad_err(x_actual, x_predicted):
    return (x_actual - x_predicted)**2


def predict(row, column, U, V):
    p = 0
    for i in range(np.shape(U)[1]):
        p += U[row, i] * V[i, column]
    return p


def factor_values(x_row, y_row, x_actual, x_predicted):
    value_list = []
    for i in range(np.shape(x_row)[1]):
        value_list.append(round(x_row[0, i] + 0.1 * 2 * (x_actual - x_predicted) * y_row[0, i], 2))
    return value_list

def assemble_U(row_i, U_matrix, factor_U):
    return 0
def assemble_V(row_i, V_matrix, factor_V):
    V_matrix = np.transpose(V_matrix)
    return np.transpose(V_matrix)


def Iteration(index, matrix_op, U, V):
    x_act = matrix_op[index[0], index[1]]
    x_predict = round(predict(index[0], index[1], U, V), 2)
    error = round(quad_err(x_act, x_predict), 2)
    V_row = factor_values(U[index[0]], np.transpose(V)[index[1]], x_act, x_predict)
    U_row = factor_values(np.transpose(V)[index[1]], U[index[0]], x_act, x_predict)
    return V_row, U_row, error







print(f'Исходная матрица:\n{matrix}')

while True:
    # mode = int(input('Выберите режим:\n0 - для теста по методичке;\n1 - для штатной работы;\n'))
    mode = 0
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

EPS = 100
index_numbers = indexnums(matrix)
print(index_numbers)
count = 1
for i in index_numbers:
    while True:
        print(i)
        print(f'_______ИТЕРАЦИЯ № {count}_______')
        f_V, f_U, error = Iteration(i, matrix, U_matrix, V_matrix)
        print(f_V)
        print(f_U)
        print(error)
        count += 1
        if error <= EPS:
            break


print(f"U matrix:\n{U_matrix}")
print()
print(f"V.T matrix:\n{V_matrix}")
print(f"e = {round(predict(index_numbers[0][0],index_numbers[0][1], U_matrix, V_matrix), 2)}")
print(f"Error = {round(quad_err(matrix[index_numbers[0][0], index_numbers[0][1]],round(predict(index_numbers[0][0],index_numbers[0][1], U_matrix, V_matrix), 2)),2)}")
print(Iteration(index_numbers[0], matrix, U_matrix, V_matrix))
print()