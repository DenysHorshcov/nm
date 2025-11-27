import math 

def f(x):
    return x**3 - 3*x**2 - 14*x - 8

def phi(x):
    return (3*x*x + 14*x + 8)**(1/3)



def iterative_method():
    a = 5
    b = 6
    x0 = (a + b) / 2
    e = 10**-8
    n_apr = 13   # теоретично достатньо 13 ітерацій

    print(f"\nМетод простої ітерації (13 кроків)")
    print(f"{'n':<5} | {'x':<20} | {'f(x)':<20} | {'|xn-xp|':<20}")
    print("-" * 80)

    x = x0
    print(f"{0:<5} | {x:<20} | {f(x):<20} | {'-':<20}")

    for n in range(1, n_apr + 1):
        xp = x
        x = phi(xp)
        diff = abs(x - xp)

        print(f"{n:<5} | {x:<20} | {f(x):<20} | {diff:<20}")

    print(f"\nFinal approximation after {n_apr} steps: x ≈ {x}\n")


def bisection_method():
    a = 5
    b = 6
    e = 10**-8

    n_apr = 14  # використовуємо саме априорну оцінку

    print(f"\nМетод бісекції (14 кроків)")
    print(f"{'n':<5} | {'a':<15} | {'b':<15} | {'x':<20} | {'f(x)':<20}")
    print("-" * 90)

    for i in range(1, n_apr + 1):
        x = (a + b) / 2
        fx = f(x)

        print(f"{i:<5} | {a:<15} | {b:<15} | {x:<20} | {fx:<20}")

        if fx * f(a) < 0:
            b = x
        else:
            a = x

    print(f"\nFinal approximation after {n_apr} steps: x ≈ {x}\n")



while True:
    mode = input(
        "\nSelect algorithm:\n"
        " 1: Iterative method\n"
        " 2: Bisection method\n"
        " 3: Exit\n"
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
