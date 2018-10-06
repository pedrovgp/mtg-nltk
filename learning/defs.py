def prime_factor(value):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors

import time

def foo(my_number):
    square = my_number * my_number
    time.sleep(1)
    return square 