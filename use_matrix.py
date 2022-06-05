def make_matrix(row, column):
    """
    The function make unit matrix
    :param row: number of rows
    :param column: number of columns
    :return: unit matrix
    """
    matrix = []
    for j in range(1, row + 1):
        _list = []
        for y in range(1, column + 1):
            if j == y:
                _list.append(1)
            else:
                _list.append(0)
        matrix.append(_list)
    return matrix


def mul_two_matrix(m1, m2):
    """
    The function mul two matrix
    :param m1: matrix 1
    :param m2: matrix 2
    :return: the solution matrix
    """
    v = []
    new = []
    index = 0
    for row in m1:
        vector = []
        for j in range(0, len(row)):
            temp = []
            for r in m2:
                temp.append(r[j])
                v = []
            for i in range(0, len(row)):
                v.append((row[i]) * (temp[i]))
            vector.append(sum(v))
        new.append(vector)
        index += 1
    return new


def mul_matrix_vector(matrix, vector):
    """
    The function mul matrix with vector
    :param matrix: the matrix
    :param vector: the vector
    :return: the new matrix
    """
    new_matrix = []
    index = 0
    for row in matrix:
        v = []
        for i in range(0, len(row)):
            v.append((row[i]) * (vector[i]))
        new_matrix.append(sum(v))
        index += 1
    return new_matrix


def find(matrix, vector_b):
    """
    The function gets a matrix and b vector and calculates and prints the solution
    :param matrix: the matrix
    :param vector_b: the vector
    :return: there is no return value
    """
    e_temp1 = make_matrix(len(matrix), len(matrix))  # Variable to save the multiplication of elementary matrices
    for i in range(len(matrix)):
        _max = abs(matrix[i][i])
        for j in range(i, len(matrix)):  # this loop for replacing rows
            if matrix[j][i] > _max:
                elementary = make_matrix(len(matrix), len(matrix))  # make unit matrix
                row_temp = elementary[i]
                elementary[i] = elementary[j]
                elementary[j] = row_temp
                e_temp1 = mul_two_matrix(elementary, e_temp1)
                # save_matrix = matrix
                matrix = mul_two_matrix(elementary, matrix)
                _max = abs(matrix[i][i])
        elementary = make_matrix(len(matrix), len(matrix))
        elementary[i][i] = 1 / matrix[i][i]  # to convert the pivot to "1"
        # save_matrix = matrix
        matrix = mul_two_matrix(elementary, matrix)
        e_temp1 = mul_two_matrix(elementary, e_temp1)
        elementary = make_matrix(len(matrix), len(matrix))

        for k in range(len(matrix)):  # this loop to reset the organs above and below the pivot
            if k != i:
                elementary[k][i] = ((-1) * matrix[k][i]) / matrix[i][i]
                e_temp1 = mul_two_matrix(elementary, e_temp1)
                save_matrix = matrix
                matrix = mul_two_matrix(elementary, matrix)
                elementary = make_matrix(len(matrix), len(matrix))
    return mul_matrix_vector(e_temp1, vector_b)  # the solution


_matrix = [[2, 1, 0], [3, -1, 0], [1, 4, -2]]
_vector_b = [-3, 1, -5]  # the values of y
