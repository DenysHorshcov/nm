import math

A = [
    [6, 2, 1, 0, 0],
    [2, 7, 2, 1, 0],
    [1, 2, 8, 2, 1],
    [0, 1, 2, 7, 2],
    [0, 0, 1, 2, 6]
]

eps = 1e-8
n = 5


def print_row(values, widths):
    """Prints a formatted row with fixed column widths."""
    row = ""
    for v, w in zip(values, widths):
        row += f"{str(v):>{w}} "
    print(row)

def print_table_header():
    widths = [5, 20, 20, 20, 20, 20, 20, 20]
    header = ["Iter", "x1", "x2", "x3", "x4", "x5", "lambda", "|Δλ|"]
    print_row(header, widths)
    print("-" * (sum(widths) + len(widths)))
    return widths

def format_number(x):
    return f"{x:.6f}".rstrip("0").rstrip(".")


def shorten_large_number(x):
    s = str(int(round(x)))
    if len(s) > 7:
        return f"{s[:3]}...{s[-3:]}"
    return s


def normalize(v):
    norm = math.sqrt(sum(vi * vi for vi in v))
    for i in range(len(v)):
        v[i] /= norm


# ----------------------------------------------------------------------
# 1. Scalar Product Method
# ----------------------------------------------------------------------
def scalar_product_method(A, eps=1e-8):
    n = len(A)
    x = [1] * n
    x_next = [0] * n

    lambda_prev = 0
    iteration = 0

    widths = print_table_header()

    while True:
        iteration += 1

        # x_next = A * x
        for i in range(n):
            s = 0
            for j in range(n):
                s += A[i][j] * x[j]
            x_next[i] = s

        # λ = (Ax, x) / (x, x)
        scalar_num = sum(x_next[i] * x[i] for i in range(n))
        scalar_den = sum(x[i] * x[i] for i in range(n))
        lambda_next = scalar_num / scalar_den

        # Absolute difference
        diff = abs(lambda_next - lambda_prev)

        print_row(
            [iteration] + [x_next[i] for i in range(n)] +
            [f"{lambda_next:.12f}", f"{diff:.12f}"],
            widths
        )

        if diff < eps:
            break

        lambda_prev = lambda_next
        x = x_next.copy()

    print("\nLargest lambda =", lambda_next)

# ----------------------------------------------------------------------
# 2. Modified Scalar Product Method (Normalized)
# ----------------------------------------------------------------------
def modified_scalar_product(A, n, eps):
    e = [1] * n
    x = [0] * n
    temp = [0] * n

    normalize(e)

    lambda_prev = 0
    lambda_next = 0
    iteration = 0

    print("\n---------- Modified Scalar Product Method (Normalized) ----------")
    print(f"{'Iter':>4} | {'x1':>10} {'x2':>10} {'x3':>10} {'x4':>10} {'x5':>10} | {'lambda':>12} | {'|lambda-lambda_prev|':>14}")
    print("---------------------------------------------------------------")

    while True:
        iteration += 1

        for i in range(n):
            x[i] = sum(A[i][j] * e[j] for j in range(n))

        lambda_next = sum(x[i] * e[i] for i in range(n))

        temp = x.copy()
        normalize(temp)

        print(f"{iteration:6} |", end="")
        for i in range(n):
            print(f" {format_number(x[i]):>10}", end="")
        print(f" | {format_number(lambda_next):>12} | {format_number(abs(lambda_next - lambda_prev)):>14}")

        if abs(lambda_next - lambda_prev) < eps:
            break

        lambda_prev = lambda_next
        e = temp.copy()

    print("---------------------------------------------------------------")
    print(f"Largest lambda = {format_number(lambda_next)}\n")


# ----------------------------------------------------------------------
# 3. Power Method
# ----------------------------------------------------------------------
def power_method(A, n, eps):
    x = [1] * n
    x_next = [0] * n

    lambda_prev = 0
    lambda_next = 0
    iteration = 0

    print("\n-------------------------- Power Method -------------------------")
    print(f"{'Iter':>4} | {'x1':>10} {'x2':>10} {'x3':>10} {'x4':>10} {'x5':>10} | {'lambda':>12} | {'|lambda-lambda_prev|':>14}")
    print("---------------------------------------------------------------")

    while True:
        iteration += 1

        for i in range(n):
            x_next[i] = sum(A[i][j] * x[j] for j in range(n))

        ratios = [abs(x_next[i] / x[i]) for i in range(n)]
        lambda_next = max(ratios)

        print(f"{iteration:6} |", end="")
        for i in range(n):
            print(f" {shorten_large_number(x_next[i]):>10}", end="")
        print(f" | {format_number(lambda_next):>12} | {format_number(abs(lambda_next - lambda_prev)):>14}")

        if abs(lambda_next - lambda_prev) < eps:
            break

        lambda_prev = lambda_next
        x = x_next.copy()

    print("---------------------------------------------------------------")
    print(f"Largest lambda = {format_number(lambda_next)}\n")


# ----------------------------------------------------------------------
# Main menu
# ----------------------------------------------------------------------
print("Choose method:")
print("1 - Scalar Product Method")
print("2 - Modified Scalar Product Method (normalized)")
print("3 - Power Method")
choice = input("Enter choice (1, 2 or 3): ")

if choice == "1":
    scalar_product_method(A)
elif choice == "2":
    modified_scalar_product(A, n, eps)
elif choice == "3":
    power_method(A, n, eps)
else:
    print("Invalid choice. Please enter 1, 2 or 3.")






