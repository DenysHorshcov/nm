import math

def f(x):
    return x**3 - 3*x**2 - 14*x - 8

# φ(x) для методу простої ітерації
def phi(x):
    return (3*x*x + 14*x + 8)**(1/3)

def iterative_method():
    a = 5
    b = 6
    x0 = (a + b) / 2
    e = 10**-8

    print(f"\n{'n':<5} | {'x':<20} | {'f(x)':<20} | {'|xn-xp|':<20}")
    print("-" * 80)

    x = x0
    print(f"{0:<5} | {x:<20} | {f(x):<20} | {'-':<20}")

    for n in range(1, 1000):  # максимум 1000 ітерацій на випадок розбігу
        xp = x
        x = phi(xp)
        diff = abs(x - xp)

        print(f"{n:<5} | {x:<20} | {f(x):<20} | {diff:<20}")

        if diff < e:
            break

    print(f"\nFinal approximation: x ≈ {x}")
    print(f"Iterations: {n}\n")


def bisection_method():
    a = 5
    b = 6
    e = 10**-8

    n_apr = math.ceil(math.log((b - a) / e) / math.log(2))

    print(f"\n{'n':<5} | {'a':<15} | {'b':<15} | {'x':<20} | {'f(x)':<20}")
    print("-" * 90)

    for i in range(1, n_apr + 1):
        x = (a + b) / 2
        fx = f(x)

        print(f"{i:<5} | {a:<15} | {b:<15} | {x:<20} | {fx:<20}")

        # вибір нового проміжку
        if fx * f(a) < 0:
            b = x
        else:
            a = x

        if (b - a) < 2 * e:
            break

    print(f"\nFinal approximation: x ≈ {x}")
    print(f"Iterations: {i}\n")


while True:
    mode = input(
        "\nSelect algorithm to solve x^3 - 3x^2 - 14x - 8 = 0:\n"
        " 1: Iterative method\n"
        " 2: Bisection method\n"
        " 3: Exit program\n"
        " > "
    )

    if mode == "1":
        iterative_method()

    elif mode == "2":
        bisection_method()

    elif mode == "3":
        print("Exiting program...\n")
        break

    else:
        print("Invalid choice, try again.\n")
