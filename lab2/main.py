import numpy as np

np.set_printoptions(linewidth=np.inf)


# -------------------------------
# Генератори матриць
# -------------------------------

def generate_augmented_matrix(size=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    matrix = np.random.randint(-9, 9, (size, size))
    vector = np.random.randint(-30, 30, size)

    # забезпечуємо діагональне домінування
    for i in range(size):
        s = sum(abs(matrix[i][j]) for j in range(size)) - abs(matrix[i][i])
        if abs(matrix[i][i]) <= s:
            matrix[i][i] = s + np.random.randint(1, 5)

    return np.column_stack((matrix, vector))


def generate_tridiagonal_augmented_matrix(size=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    lower = np.random.randint(-9, 9, size - 1)
    upper = np.random.randint(-9, 9, size - 1)

    matrix = np.zeros((size, size))

    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(matrix[1:], lower)
    np.fill_diagonal(matrix[:, 1:], upper)

    # діагональне домінування
    for i in range(size):
        row_sum = sum(abs(matrix[i][j]) for j in range(size))
        if abs(matrix[i][i]) <= row_sum:
            matrix[i][i] = row_sum + np.random.randint(1, 5)

    vector = np.random.randint(-30, 30, size)
    return np.column_stack((matrix, vector))


# -------------------------------
# Метод Гауса
# -------------------------------

def gauss(aug):
    n = len(aug)
    matrix = np.array(aug, dtype=float)

    # Прямий хід
    for k in range(n):
        if matrix[k][k] == 0:
            for i in range(k + 1, n):
                if matrix[i][k] != 0:
                    matrix[[k, i]] = matrix[[i, k]]
                    break
            else:
                raise ValueError("Zero pivot.")

        for i in range(k + 1, n):
            factor = matrix[i][k] / matrix[k][k]
            matrix[i] -= factor * matrix[k]

        print(matrix)

    # Визначник
    det = np.prod(np.diag(matrix))
    print("determinant:", det)

    # Зворотній хід
    for k in range(n - 1, -1, -1):
        for i in range(k):
            factor = matrix[i][k] / matrix[k][k]
            matrix[i] -= factor * matrix[k]

    # Нормалізація
    for i in range(n):
        matrix[i] /= matrix[i][i]

    return det, matrix[:, -1]


# -------------------------------
# Обернена матриця методом Гауса
# -------------------------------

def gauss_inv(matrix):
    n = len(matrix)
    aug = np.hstack((matrix.astype(float), np.eye(n)))

    # Прямий хід
    for k in range(n):
        if aug[k][k] == 0:
            for i in range(k + 1, n):
                if aug[i][k] != 0:
                    aug[[k, i]] = aug[[i, k]]
                    break
            else:
                raise ValueError("Zero pivot.")

        aug[k] /= aug[k][k]

        for i in range(k + 1, n):
            factor = aug[i][k]
            aug[i] -= factor * aug[k]

        print(aug)

    # Зворотній хід
    for k in range(n - 1, -1, -1):
        for i in range(k):
            factor = aug[i][k]
            aug[i] -= factor * aug[k]

    return aug[:, n:]


# -------------------------------
# Метод Зейделя
# -------------------------------

def seidel(aug, tol=1e-5, max_iterations=1000):
    A = np.array([row[:-1] for row in aug], float)
    b = np.array([row[-1] for row in aug], float)

    n = len(A)
    x = np.zeros(n)

    print(f"{'Iteration':^10} | {'x_new':^50} | {'Norm':^10}")
    print("-" * 90)

    for it in range(1, max_iterations + 1):
        x_new = np.copy(x)

        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        norm = np.linalg.norm(x_new - x, np.inf)

        print(f"{it:<10} | {np.array2string(x_new, precision=6, floatmode='fixed'):<50} | {norm:<10}")

        if norm < tol:
            print(f"\nЗбіжність досягнута після {it} ітерацій.")
            return x_new

        x = x_new

    raise ValueError("Метод Зейделя не збігається.")


# -------------------------------
# Метод прогонки
# -------------------------------

def tridiagonal_matrix_algorithm(matrix):
    A = np.array([row[:-1] for row in matrix], float)
    f = np.array([row[-1] for row in matrix], float)
    n = len(f)

    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = -A[0][1] / A[0][0]
    b[0] = f[0] / A[0][0]

    for i in range(1, n - 1):
        denom = A[i][i] + A[i][i - 1] * a[i - 1]
        a[i] = -A[i][i + 1] / denom
        b[i] = (f[i] - A[i][i - 1] * b[i - 1]) / denom

    b[-1] = (f[-1] - A[-1][-2] * b[-2]) / (A[-1][-1] + A[-1][-2] * a[-2])

    x = np.zeros(n)
    x[-1] = b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = a[i] * x[i + 1] + b[i]

    return x


# -------------------------------
# Циклічне меню
# -------------------------------

seed = 69

while True:
    print("\nSelect algorithm:")
    print("1. Метод Гауса")
    print("2. Метод Зейделя")
    print("3. Метод прогонки")
    print("4. Exit")

    mode = input("> ")

    if mode == "1":
        aug = generate_augmented_matrix(seed=seed)
        print("\nAugmented matrix:\n", aug)
        det, sol = gauss(aug)
        print("\nSolution:", sol)

    elif mode == "2":
        aug = generate_augmented_matrix(seed=seed)
        print("\nAugmented matrix:\n", aug)
        tol = float(input("Tolerance: "))
        sol = seidel(aug, tol=tol)
        print("\nSolution:", sol)

    elif mode == "3":
        tri = generate_tridiagonal_augmented_matrix(seed=seed)
        print("\nTridiagonal matrix:\n", tri)
        sol = tridiagonal_matrix_algorithm(tri)
        print("\nThomas result:", sol)

    elif mode == "4":
        print("\nExiting program...")
        break

    else:
        print("\nInvalid option, try again.")
