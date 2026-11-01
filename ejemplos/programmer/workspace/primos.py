from math import isqrt


def obtener_primos(hasta: int) -> list[int]:
    """Devuelve todos los números primos menores o iguales a `hasta`."""
    if hasta < 2:
        return []

    es_primo = [True] * (hasta + 1)
    es_primo[0] = es_primo[1] = False

    for numero in range(2, isqrt(hasta) + 1):
        if es_primo[numero]:
            for multiplo in range(numero * numero, hasta + 1, numero):
                es_primo[multiplo] = False

    return [numero for numero, primo in enumerate(es_primo) if primo]


if __name__ == "__main__":
    limite = int(input("Obtener números primos hasta: "))
    print(obtener_primos(limite))
