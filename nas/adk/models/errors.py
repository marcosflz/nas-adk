from __future__ import annotations


class LlmError(Exception):
    """Error base de un proveedor LLM.

    ``retryable`` indica si conviene repetir la misma petición; ``fallback_allowed``
    indica si tiene sentido probar el siguiente modelo de la cadena.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        fallback_allowed: bool = False,
        body: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.fallback_allowed = fallback_allowed
        self.body = body
        self.cause = cause


class LlmAuthError(LlmError):
    """API key ausente o inválida. No se reintenta ni se usa fallback."""


class LlmQuotaError(LlmError):
    """Saldo o cuota agotados para el modelo actual."""


class LlmRateLimitError(LlmError):
    """Límite de peticiones alcanzado."""


class LlmModelError(LlmError):
    """Modelo inexistente, no permitido o no disponible."""


class LlmServerError(LlmError):
    """Error 5xx, timeout o fallo de red."""


class LlmResponseError(LlmError):
    """Respuesta no parseable o que no cumple el contrato esperado."""


__all__ = [
    "LlmAuthError",
    "LlmError",
    "LlmModelError",
    "LlmQuotaError",
    "LlmRateLimitError",
    "LlmResponseError",
    "LlmServerError",
]
