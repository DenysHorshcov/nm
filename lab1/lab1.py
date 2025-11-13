import math

def f(x):
    return x**3 - 3*x**2 - 14*x - 8

def phi(x):
    tau = 2 / 89
    return x - tau * f(x)

def ao(q, xn, xp):
    return abs(xn - xp) / (1 - q)

def iterative_method():
    a = 5
    b = 6
    x0 = (a + b) / 2
    e = 10**-4
    q = 27 / 89
    n = int(
        math.log(abs(phi(x0) - x0) / ((1 - q) * e)) / math.log(1 / q)
    ) + 1
    x = x0

    print(f"\n{'n':<5} | {'x':<20} | {'f(x)':<20} | {'ao':<20}")
    print("-" * 70)
    print(f"{0:<5} | {x:<20} | {f(x):<20} | {'-':<20}")

    for i in range(n):
        xp = x
        x = phi(x)
        ao_value = ao(q, x, xp)
        print(f"{i+1:<5} | {x:<20} | {f(x):<20} | {ao_value:<20}")
    print()

def bisection_method():
    a = 5
    b = 6
    e = 10**-4
    n_apr = math.ceil(math.log((b - a) / e) / math.log(2))

    print(f"\n{'n':<5} | {'x':<20} | {'f(x)':<20} | {'ao':<20}")
    print("-" * 70)

    for i in range(1, n_apr + 1):
        x = (a + b) / 2
        fx = f(x)
        ao_value = (b - a) / 2
        print(f"{i:<5} | {x:<20} | {fx:<20} | {ao_value:<20}")

        if fx * f(a) < 0:
            b = x
        else:
            a = x

        if (b - a) < 2 * e:
            break
    print()


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
