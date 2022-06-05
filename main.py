import use_matrix
from typing import Tuple, List
import bisect


def linear_interpolation(_table, _x):
    for i in range(0, len(_table)):
        if _table[i][0] < _x < _table[i + 1][0]:
            y = ((_table[i][1] - _table[i + 1][1]) / (_table[i][0] - _table[i + 1][0])) * _x + ((_table[i + 1][1]) * _table[i][0] - _table[i][1] * _table[i + 1][0]) / (_table[i][0] - _table[i + 1][0])
            return y


def polynomial_interpolation(_table, _x):
    matrix = []
    vector_y = []
    for j in range(0, len(_table)):
        vector_y.append(_table[j][1])
        temp = []
        for k in range(0, len(_table)):
            temp.append(_table[j][0] ** k)
        matrix.append(temp)

    vector = use_matrix.find(matrix, vector_y)
    _sum = 0
    for i in range(0, len(vector)):
        _sum = _sum + vector[i] * _x ** i
    return _sum


def lagrange_interpolation(_table, _x):
    _polynomial = 0
    for i in range(0, len(_table)):
        l_x = 1
        for j in range(0, len(_table)):
            if i != j:
                l_x = l_x * ((_x - _table[j][0]) / (_table[i][0] - _table[j][0]))
        _polynomial = _polynomial + l_x * _table[i][1]
    return _polynomial


def neville_interpolation(_table, _x):
    n = len(_table)
    x = 0
    y = 1
    for i in range(1, n, +1):
        for j in range(n-1, i-1, -1):
            _table[j][y] = ((_x - _table[j-i][x]) * _table[j][y]-(_x - _table[j][x]) * _table[j-1][y])/(_table[j][x] - _table[j-i][x])
            print(_table[j][y])
    result = str(_table[n-1][y])
    return result


def compute_changes(x: List[float]) -> List[float]:
    return [x[i+1] - x[i] for i in range(len(x) - 1)]


def create_tridiagonalmatrix(n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
    A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
    B = [2] * n
    C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
    return A, B, C


def create_target(n: int, h: List[float], y: List[float]):
    return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + [0]


def solve_tridiagonalsystem(A: List[float], B: List[float], C: List[float], D: List[float]):
    c_p = C + [0]
    d_p = [0] * len(B)
    X = [0] * len(B)

    c_p[0] = C[0] / B[0]
    d_p[0] = D[0] / B[0]
    for i in range(1, len(B)):
        c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
        d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / (B[i] - c_p[i - 1] * A[i - 1])

    X[-1] = d_p[-1]
    for i in range(len(B) - 2, -1, -1):
        X[i] = d_p[i] - c_p[i] * X[i + 1]

    return X


def compute_spline(x: List[float], y: List[float]):
    n = len(x)
    if n < 3:
        raise ValueError('Too short an array')
    if n != len(y):
        raise ValueError('Array lengths are different')

    h = compute_changes(x)
    if any(v < 0 for v in h):
        raise ValueError('X must be strictly increasing')

    A, B, C = create_tridiagonalmatrix(n, h)
    D = create_target(n, h, y)

    M = solve_tridiagonalsystem(A, B, C, D)

    coefficients = [[(M[i+1]-M[i])*h[i]*h[i]/6, M[i]*h[i]*h[i]/2, (y[i+1] - y[i] - (M[i+1]+2*M[i])*h[i]*h[i]/6), y[i]] for i in range(n-1)]

    def spline(val):
        idx = min(bisect.bisect(x, val)-1, n-2)
        z = (val - x[idx]) / h[idx]
        C = coefficients[idx]
        return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

    return spline


'''example for Linear Interpolation and Polynomial Interpolation'''
table = ((0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794))
value_x = 2.5

'''example for Lagrange Interpolation'''
table_to_lagrange = ((1, 1), (2, 0), (4, 1.5))
x_l = 3

'''example for Neville Interpolation'''
table_neville = [[1, 0], [1.2, 0.112463], [1.3, 0.167996], [1.4, 0.222709]]
x_neville = 1.28

'''example for Cubic Spline Interpolation'''
x___ = [0, 3.1415926535/6, 3.1415926535/4, 3.1415926535/2]
y___ = [0, 0.5, 0.7072, 1]
val_x = 3.1415926535/3

print("Approximations\n")
print(f'Linear Interpolation:\n({value_x}, {linear_interpolation(table, value_x):.4f})\n\n')
print(f'Polynomial Interpolation:\n({value_x}, {polynomial_interpolation(table, value_x):.4f})\n\n')
print(f'Lagrange Interpolation:\n({x_l}, {lagrange_interpolation(table_to_lagrange, x_l):.4f})\n\n')
print(f'Cubic Spline:\n({val_x:.4f}, {compute_spline(x___, y___)(val_x):.4f})\n\n')
print(f'Neville Interpolation:\n({x_neville}, {lagrange_interpolation(table_neville, x_neville):.4f})\n\n')
