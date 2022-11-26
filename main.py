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
    return ((x_actual - x_predicted)**2)/2


def predict(row, column, U, V):
    p = 0
    for i in range(np.shape(U)[1]):
        p += U[row, i] * V[i, column]
    return p


def factor_values(x_row, y_row, x_actual, x_predicted):
    value_list = []
    for i in range(np.shape(x_row)[1]):
        value_list.append(round(x_row[0, i] + 0.05 * 2 * (x_actual - x_predicted) * y_row[0, i], 2))
    return value_list


def assemble_U(row_i, U_fuck, factor_U):
    U_fuck[row_i] = factor_U
    return U_fuck


def assemble_V(row_i, V_fact, factor_V):
    np.transpose(V_fact)[row_i] = factor_V
    return V_fact


def Iteration(index, matrix_op, U, V):
    x_act = matrix_op[index[0], index[1]]
    x_predict = round(predict(index[0], index[1], U, V), 2)
    err = round(quad_err(x_act, x_predict), 2)
    U_row = factor_values(U[index[0]], np.transpose(V)[index[1]], x_act, x_predict)
    V_row = factor_values(np.transpose(V)[index[1]], U[index[0]], x_act, x_predict)
    print(f'Фактическое значение: {x_act}')
    print(f'Рассчитанное значение: {x_predict}')
    print(f'Квадратичная ошибка: {err}')
    print(f'Обновленные значения скрытых факторов пользователя : {U_row}')
    print(f'Обновленные значения скрытых факторов объектов : {V_row}')
    return U_row, V_row, err


def replace_nan(matrix_op, U, V):
    nan_list = [(i, j)
                for i in range(len(matrix_op.tolist()))
                for j in range(len(matrix_op.tolist()[i]))
                if np.isnan(matrix_op.tolist()[i][j])]
    for el in nan_list:
        matrix_op[el[0], el[1]] = round(predict(el[0], el[1], U, V), 2)
    return matrix_op


def replace_all(matrix_op, U, V):
    all_list = [(i, j)
                for i in range(len(matrix_op.tolist()))
                for j in range(len(matrix_op.tolist()[i]))]
    for el in all_list:
        matrix_op[el[0], el[1]] = round(predict(el[0], el[1], U, V), 2)
    return matrix_op


print(f'Исходная матрица:\n{matrix}')

while True:
    mode = int(input('Выберите режим:\n0 - для теста по методичке;\n1 - для штатной работы;\n'))
    # mode = 0
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

EPS = 0.5
index_numbers = indexnums(matrix)
print(f"U matrix:\n{U_matrix}")
print()
print(f"Vt matrix:\n{V_matrix}")
count = 1
check = [index_numbers[0]]
for element in index_numbers:
    print(f'########## ЭЛЕМЕНТ : {matrix[element[0], element[1]]} ##########')
    while True:
        print(f'_______Итерация № {count}_______')
        f_U, f_V, error = Iteration(element, matrix, U_matrix, V_matrix)
        for chi in check:
            if chi == element:
                case = 0
                break
            elif chi[1] == element[1]:
                case = 1
                break
            elif chi[0] == element[0]:
                case = 2
                break
            else:
                case = 0
                break
        if case == 0:
            U_matrix = assemble_U(element[0], U_matrix, f_U)
            V_matrix = assemble_V(element[1], V_matrix, f_V)
        elif case == 1:
            print('Салам 2')
            U_matrix = assemble_U(element[0], U_matrix, f_U)
        elif case == 2:
            print('Салам 3')
            V_matrix = assemble_V(element[1], V_matrix, f_V)
        case = -1
        print(f"U matrix:\n{U_matrix}")
        print()
        print(f"Vt matrix:\n{V_matrix}")
        count += 1
        if error <= EPS:
            count = 1
            check.append(element)
            break
print(f'Заполненная матрица:\n{replace_nan(matrix, U_matrix, V_matrix)}')
print(f'Заполненная матрица ПОЛНОСТЬЮ:\n{replace_all(matrix, U_matrix, V_matrix)}')