import numpy as np

matrix = np.matrix([[3, np.nan, 5, np.nan, 1, 2, np.nan, 8],
                    [np.nan, 2, np.nan, 1, 3, np.nan, 6, np.nan],
                    [np.nan, 7, np.nan, 3, np.nan, 4, np.nan, 5]])


def generate_matrix(h_f):
    U = np.matrix(2 * (np.random.random_sample((np.shape(matrix)[0], h_f))))
    V = np.matrix(2 * (np.random.random_sample((h_f, np.shape(matrix)[1]))))
    return U, V


def indexnums(matrix_op):
    return [(i, j) for i in range(len(matrix_op.tolist())) for j in range(len(matrix_op.tolist()[i])) if not np.isnan(matrix_op.tolist()[i][j])]


def quad_err(x_actual, x_predicted):
    return 0.5*((x_actual - x_predicted)**2)


def predict(row, column, U, V):
    p = 0
    for i in range(np.shape(U)[1]):
        p += U[row, i] * V[i, column]
    return p


def factor_values(x_row, y_row, x_actual, x_predicted):
    value_list = []
    for i in range(np.shape(x_row)[1]):
        value_list.append(x_row[0, i] + 0.05 * 2 * (x_actual - x_predicted) * y_row[0, i])
    return value_list


def assemble_U(row_i, U_fuck, factor_U):
    U_fuck[row_i] = factor_U
    return U_fuck


def assemble_V(row_i, V_fact, factor_V):
    np.transpose(V_fact)[row_i] = factor_V
    return V_fact


def Iteration(index, matrix_op, U, V):
    x_act = matrix_op[index[0], index[1]]
    x_predict = predict(index[0], index[1], U, V)
    err = quad_err(x_act, x_predict)
    U_row = factor_values(U[index[0]], np.transpose(V)[index[1]], x_act, x_predict)
    V_row = factor_values(np.transpose(V)[index[1]], U[index[0]], x_act, x_predict)
    print(f'Рассчитанное значение: {np.round(x_predict, 3)}')
    print(f'Квадратичная ошибка: {np.round(err, 3)}')
    print(f'Обновленные значения скрытых факторов пользователя : {np.round(U_row, 3)}')
    print(f'Обновленные значения скрытых факторов объектов : {np.round(V_row, 3)}')
    return U_row, V_row, err

print(f'Исходная матрица:\n{matrix}')
hidden_factors = int(input('Введите количество скрытых факторов:'))
U_matrix, V_matrix = generate_matrix(hidden_factors)
index_numbers = indexnums(matrix)
print(f"U matrix:\n{U_matrix}\n")
print(f"Vt matrix:\n{V_matrix}")
for i in range(50):
    print(f'_______Итерация № {i}_______')
    for element in index_numbers:
        print(f"Число: {matrix[element]}")
        f_U, f_V, error = Iteration(element, matrix, U_matrix, V_matrix)
        U_matrix = assemble_U(element[0], U_matrix, f_U)
        V_matrix = assemble_V(element[1], V_matrix, f_V)
print(f'Исходная матрица:\n{matrix}')
print(f'Заполненная матрица:\n{np.round(U_matrix * V_matrix, 3)}')
