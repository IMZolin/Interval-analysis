import numpy as np
import intvalpy as ip
import matplotlib.pyplot as plt
from intvalpy.utils import asinterval, intersection, dist, infinity, isnan
from intvalpy.RealInterval import ARITHMETICS


def f1(x, y):
    return (x - 1.22) ** 2 + y ** 2 - 1


def f2(x, y):
    return x - y ** 2


def df1_x(x):
    return 2 * (x - 1.22)


def df1_y(y):
    return 2 * y


def df2_x(x):
    return 1


def df2_y(y):
    return -2 * y


def J(X):
    midL = np.zeros((2, 2))
    radL = np.zeros((2, 2))

    radL[0][0] = df1_x(X[0]).rad
    midL[0][0] = df1_x(X[0]).mid

    radL[0][1] = df1_y(X[1]).rad
    midL[0][1] = df1_y(X[1]).mid

    midL[1][0] = 1
    radL[1][0] = 0

    radL[1][1] = df2_y(X[1]).rad
    midL[1][1] = df2_y(X[1]).mid
    return ip.Interval(midL, radL, midRadQ=True)


def F(X):
    return [f1(X[0], X[1]), f2(X[0], X[1])]


def Krawczyk(func, J, x0, maxiter=2000, tol=1e-5):
    def K(X, c):
        L = J(X)
        # print("\nL:\n", L)
        LAMBDA = np.linalg.inv(L.to_float().mid)
        # print("\nLAMBDA:\n", LAMBDA)
        B = np.eye(2) - LAMBDA @ L
        # print("\nB:\n", B)
        w, _ = np.linalg.eigh(B.to_float().mag)
        return c - LAMBDA @ func(c) + B @ (X - c)

    result = x0
    pre_result = result.copy
    c = asinterval(result.mid)

    error = infinity
    nit = 0
    X_k = []
    X_k.append(result)
    Kr = []
    i = 0
    while nit <= maxiter and error > tol:
        i += 1
        krav = K(result, c)
        Kr.append(krav)
        # print("\nkrav:\n", krav)
        result = intersection(result, krav)
        # print("\nresult:\n", result)
        if isnan(result).any():
            X_k.append(result)
            return result
        X_k.append(result)
        c = asinterval(result.mid)
        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1

    return result, X_k, Kr, i


def print_int(X):
    for i in range(len(X)):
        a_1 = float(X_k[i][0].a)
        a_2 = float(X_k[i][1].a)
        b_1 = float(X_k[i][0].b)
        b_2 = float(X_k[i][1].b)
        print(i + 1, " & [", "{:.6}".format(a_1), ",", "{:.6}".format(b_1), "] & [", "{:.6}".format(a_2), ",",
              "{:.6}".format(b_2), "] \\\\")


if __name__ == "__main__":
    midX = [1.08, 1.0]
    radX = [0.2, 0.25]

    # midX = [1.25, 0.7]
    # radX = [0.45, 0.45]

    X = ip.Interval(midX, radX, midRadQ=True)

    z, X_k, Kr, i = Krawczyk(F, J, X)

    for i in range(len(X_k)):
        one = abs(X_k[i][0].b - X_k[i][0].a)
        two = abs(X_k[i][1].b - X_k[i][1].a)
        iveRect = plt.Rectangle((X_k[i][0].a, X_k[i][1].a), one, two, edgecolor='black', facecolor='none',
                                linewidth=1.2)
        plt.gca().add_patch(iveRect)

    # for i in range(len(Kr)):
    #     one = abs(Kr[i][0].b - Kr[i][0].a)
    #     two = abs(Kr[i][1].b - Kr[i][1].a)
    #     iveRect = plt.Rectangle((Kr[i][0].a, Kr[i][1].a), one, two, edgecolor='blue', facecolor='none', linewidth=1.2)
    #     plt.gca().add_patch(iveRect)

    print("Кол-во итераций", i)
    plt.plot(midX[0], midX[1], 'b*', ms=4, label='start')
    print_int(X_k)
    x = np.arange(0, 2, 0.01)
    y = np.sqrt(1 - pow(x - 1.22, 2))
    plt.plot(x, y, '--m', linewidth=0.8, label=r'$\sqrt{1 - (x - 1.22)^2}$')
    y = np.sqrt(x)
    plt.plot(x, y, 'g', linewidth=0.8, label=r'$\sqrt{x}$')
    plt.grid()
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    # plt.xlim(0.8, 1.7)
    # plt.ylim(0.35, 1.05)
    plt.xlim(0.88, 1.28)
    plt.ylim(0.75, 1.25)
    plt.legend()
    plt.show()
