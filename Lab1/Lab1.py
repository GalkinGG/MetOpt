import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from sympy import *

phi = (1 + 5 ** 0.5) / 2


# Вычисление значения функции
def fun(f, x, y):
    return f.subs([(X, x), (Y, y)]).evalf(15)


# Вычисление градиента в точке
def gradient(f, x, y):
    return (diff(f, X).subs([(X, x), (Y, y)])).evalf(15), (diff(f, Y).subs([(X, x), (Y, y)])).evalf(15)


# Условие останова
def stop_condition(x, y, prev, eps):
    return ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5 < eps
    # abs(f(x, y) - f(prev[0], prev[1])) >= eps


# Градиентный спуск с фиксированным шагом
def gradient_descent(f, start_point, learning_rate=0.05, eps=1e-5):
    points = [(start_point[0], start_point[1])]
    counter = 0
    while True:
        grad = gradient(f, points[-1][0], points[-1][1])
        x = points[-1][0] - learning_rate * grad[0]
        y = points[-1][1] - learning_rate * grad[1]
        if stop_condition(x, y, points[-1], eps):
            break
        points.append((x, y))
        counter += 1
    return counter, points[-1], fun(f, points[-1][0], points[-1][1]), points, 0, counter


# Метод дихотомии
def dichotomy(f, x, y, grad_x, grad_y, eps=1e-5):
    func_counter = 0
    l = eps
    r = 0.5
    delta = eps / 2

    def step(rate):
        return x - rate * grad_x, y - rate * grad_y

    while not r - l < eps:
        x1 = (r + l - delta) / 2
        x2 = (r + l + delta) / 2
        if fun(f, *step(x1)) < fun(f, *step(x2)):
            r = x2
        else:
            l = x1
        func_counter += 2

    return r + l / 2, func_counter


# Градиентный спуск на основе дихотомии
def dichotomy_descent(f, start_point, eps=1e-5):
    func_counter = 0
    grad_counter = 0
    points = [(start_point[0], start_point[1])]
    counter = 0
    while True:
        grad = gradient(f, points[-1][0], points[-1][1])
        grad_counter += 1
        res = dichotomy(f, points[-1][0], points[-1][1], grad[0], grad[1], eps)
        learning_rate = res[0]
        func_counter += res[1]
        x = points[-1][0] - learning_rate * grad[0]
        y = points[-1][1] - learning_rate * grad[1]
        if stop_condition(x, y, points[-1], eps):
            break
        points.append((x, y))
        counter += 1
    return counter, points[-1], fun(f, points[-1][0], points[-1][1]), points, func_counter, grad_counter


# Метод Нелдера-Мида
def nelder_mead(f, x0, eps):
    res = optimize.minimize(f, x0, method="Nelder-Mead", options={"xatol": eps, "return_all": True})
    return res["nit"], res["x"], res["fun"], res["allvecs"], res["nfev"], 0


# Метод золотого сечения
def golden_ratio(f, x, y, grad_x, grad_y, eps=1e-5):
    func_counter = 0
    l = eps
    r = 0.5

    def step(rate):
        return x - rate * grad_x, y - rate * grad_y

    while not r - l < eps:
        delta = (r - l) / phi
        x1 = r - delta
        x2 = l + delta
        if fun(f, *step(x1)) < fun(f, *step(x2)):
            r = x2
        else:
            l = x1
        func_counter += 2

    return r + l / 2, func_counter


# Градиентный спуск на основе золотого сечения
def golden_ratio_descent(f, start_point, eps=1e-5):
    func_counter = 0
    grad_counter = 0
    points = [(start_point[0], start_point[1])]
    counter = 0
    while True:
        grad = gradient(f, points[-1][0], points[-1][1])
        grad_counter += 1
        res = golden_ratio(f, points[-1][0], points[-1][1], grad[0], grad[1], eps)
        learning_rate = res[0]
        func_counter += res[1]
        x = points[-1][0] - learning_rate * grad[0]
        y = points[-1][1] - learning_rate * grad[1]
        if stop_condition(x, y, points[-1], eps):
            break
        points.append((x, y))
        counter += 1
    return counter, points[-1], fun(f, points[-1][0], points[-1][1]), points, func_counter, grad_counter


# Вывод графика функции и её линий уровня
def draw(sym, f, Xs, Ys, name, points, counter):
    xs = [points[i][0] for i in range(counter)]
    ys = [points[i][1] for i in range(counter)]
    zs = [fun(sym, xs[i], ys[i]) for i in range(counter)]
    Zs = f(Xs, Ys)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xs, Ys, Zs, cmap='viridis', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    ax.scatter(xs, ys, zs, color='r', label='Траектория', alpha=0.8)
    ax.legend()
    plt.title("График функции. " + name)
    plt.show()
    plt.figure()
    plt.contour(Xs, Ys, Zs, levels=20, cmap='viridis')
    plt.plot(xs, ys, color='r')
    plt.scatter(xs, ys, color='r', label='Траектория')
    plt.annotate('Result', xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 1, ys[-1] + 1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Линии уровня. " + name)
    plt.legend()
    plt.show()


# Основная функция для получения результатов работы методов на переданной функции
def testMethods(f, p, eps=1e-5, learning_rate=0.01):
    res1 = gradient_descent(f, p, learning_rate, eps)

    counter_grad = res1[0]
    point_grad = res1[1]
    val_grad = res1[2]
    points_grad = res1[3]
    fev_grad = res1[4]
    gev_grad = res1[5]
    distance = ((p[0] - point_grad[0]) ** 2 + (p[1] - point_grad[1]) ** 2) ** 0.5
    x_range = np.linspace(float(point_grad[0] - distance), float(point_grad[0] + distance), 100)
    y_range = np.linspace(float(point_grad[1] - distance), float(point_grad[1] + distance), 100)
    Xs, Ys = np.meshgrid(x_range, y_range)
    print("Градиентный спуск\n")
    print(f"Количество итераций: {counter_grad}")
    print(f"Полученная точка: {point_grad}")
    print(f"Полученное значение функции: {val_grad}")
    print(f"Количество вычислений функции: {fev_grad}")
    print(f"Количество вычислений градиента: {gev_grad}")

    res2 = dichotomy_descent(f, p, eps)
    counter_dich = res2[0]
    point_dich = res2[1]
    val_dich = res2[2]
    points_dich = res2[3]
    fev_dich = res2[4]
    gev_dich = res2[5]
    print("\nДихотомия\n")
    print(f"Количество итераций: {counter_dich}")
    print(f"Полученная точка: {point_dich}")
    print(f"Полученное значение функции: {val_dich}")
    print(f"Количество вычислений функции: {fev_dich}")
    print(f"Количество вычислений градиента: {gev_dich}")

    res3 = nelder_mead(lambdify([(X, Y)], f, 'scipy'), p, eps)
    counter_nelder = res3[0]
    point_nelder = res3[1]
    val_nelder = res3[2]
    points_nelder = res3[3]
    fev_nelder = res3[4]
    gev_nelder = res3[5]
    print("\nНелдер-Мид\n")
    print(f"Количество итераций: {counter_nelder}")
    print(f"Полученная точка: {point_nelder}")
    print(f"Полученное значение функции: {val_nelder}")
    print(f"Количество вычислений функции: {fev_nelder}")
    print(f"Количество вычислений градиента: {gev_nelder}")

    res4 = golden_ratio_descent(f, p, eps)
    counter_gold = res4[0]
    point_gold = res4[1]
    val_gold = res4[2]
    points_gold = res4[3]
    fev_gold = res4[4]
    gev_gold = res4[5]
    print("\nЗолотое сечение\n")
    print(f"Количество итераций: {counter_gold}")
    print(f"Полученная точка: {point_gold}")
    print(f"Полученное значение функции: {val_gold}")
    print(f"Количество вычислений функции: {fev_gold}")
    print(f"Количество вычислений градиента: {gev_gold}")
    draw(f, lambdify((X, Y), f, 'numpy'), Xs, Ys, "Градиентный спуск", points_grad, counter_grad)
    draw(f, lambdify((X, Y), f, 'numpy'), Xs, Ys, "Дихотомия", points_dich, counter_dich)
    draw(f, lambdify((X, Y), f, 'numpy'), Xs, Ys, "Недлер-Мид", points_nelder, counter_nelder)
    draw(f, lambdify((X, Y), f, 'numpy'), Xs, Ys, "Золотое сечение", points_gold, counter_gold)


if __name__ == '__main__':
    X, Y = symbols('x y')
    # f1 = -cos(X) * cos(Y) * exp(-((X - pi) ** 2 + (Y - pi) ** 2))
    # testMethods(f1, (1.6, 1.6), 1e-5)
    f2 = X ** 2 + Y ** 2
    testMethods(f2, (-2, 2), 1e-7)
