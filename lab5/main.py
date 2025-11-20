import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.tan(x)

def build_natural_cubic_spline(x, y):

    n = len(x) - 1  # кількість відрізків
    h = np.diff(x)  # h_i = x_{i+1} - x_i

    # Крок 1: збірка СЛАР для других похідних (M_i)
    # Природні умови: M_0 = 0, M_n = 0
    A = np.zeros((n+1, n+1))
    rhs = np.zeros(n+1)

    # Ліва границя (природна)
    A[0, 0] = 1.0
    rhs[0] = 0.0

    # Внутрішні вузли
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        rhs[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    # Права границя (природна)
    A[n, n] = 1.0
    rhs[n] = 0.0

    # Розв’язуємо тридіагональну систему (можна через np.linalg.solve)
    M = np.linalg.solve(A, rhs)  # M_i = S''(x_i)

    # Крок 2: обчислення коефіцієнтів a, b, c, d для кожного відрізка
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        a[i] = y[i]
        c[i] = M[i] / 2.0
        d[i] = (M[i+1] - M[i]) / (6.0 * h[i])
        b[i] = (y[i+1] - y[i]) / h[i] - (2.0 * M[i] + M[i+1]) * h[i] / 6.0

    return a, b, c, d

def spline_eval(x_eval, x_nodes, a, b, c, d):

    x_eval = np.asarray(x_eval)
    S = np.zeros_like(x_eval, dtype=float)
    n = len(x_nodes) - 1

    for k, xv in enumerate(x_eval):
        # Знаходимо індекс i, такий що x_i <= xv <= x_{i+1}
        if xv <= x_nodes[0]:
            i = 0
        elif xv >= x_nodes[-1]:
            i = n - 1
        else:
            i = np.searchsorted(x_nodes, xv) - 1

        dx = xv - x_nodes[i]
        S[k] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return S

# --- Основна частина ---

# Вузли інтерполяції (можеш змінити кількість при потребі)
start = -0.5
end = 0.5
num_nodes = 15
x_nodes = np.linspace(start, end, num_nodes)
y_nodes = f(x_nodes)

# Будуємо природний кубічний сплайн
a, b, c, d = build_natural_cubic_spline(x_nodes, y_nodes)

# Точна функція та сплайн на щільній сітці для аналізу наближення
x_plot = np.linspace(start, end, 1000)
y_true = f(x_plot)
y_spline = spline_eval(x_plot, x_nodes, a, b, c, d)

# Аналіз похибки
abs_error = np.abs(y_true - y_spline)
max_error = np.max(abs_error)
x_at_max_error = x_plot[np.argmax(abs_error)]

print("Вузли інтерполяції:")
print(x_nodes)
print("\nЗначення tan(x) у вузлах:")
print(y_nodes)

print("\nМаксимальна абсолютна похибка на [-0.5; 0.5]:")
print(f"max|S(x) - tan(x)| ≈ {max_error:.6e} в точці x ≈ {x_at_max_error:.6f}")

# Табличка похибок у деяких контрольних точках
control_points = np.linspace(start, end, 11)
true_cp = f(control_points)
spline_cp = spline_eval(control_points, x_nodes, a, b, c, d)
err_cp = np.abs(true_cp - spline_cp)

print("\nКонтрольні точки (табличка похибок):")
print(f"{'x':>10} | {'tan(x)':>15} | {'S(x)':>15} | {'|S-f|':>15}")
print("-"*60)
for xv, tv, sv, ev in zip(control_points, true_cp, spline_cp, err_cp):
    print(f"{xv:10.4f} | {tv:15.8f} | {sv:15.8f} | {ev:15.8e}")

# Графіки
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(x_plot, y_true, label='f(x) = tan(x)', linewidth=2)
plt.plot(x_plot, y_spline, label='Природний кубічний сплайн', linestyle='--')
plt.scatter(x_nodes, y_nodes, color='red', label='Вузли')
plt.title('Інтерполяція tan(x) природним кубічним сплайном')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_plot, abs_error)
plt.title('Абсолютна похибка |S(x) - tan(x)|')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)

plt.tight_layout()
plt.show()
