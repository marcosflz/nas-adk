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


class LlmRequestError(LlmError):
    """Petición inválida detectada localmente o rechazada por el proveedor (400/413)."""


class LlmUnsupportedError(LlmError, NotImplementedError):
    """Capacidad no soportada por este proveedor o por esta configuración."""


class LlmPermissionError(LlmModelError):
    """La credencial no tiene permiso para esta operación (403)."""


class LlmTimeoutError(LlmServerError):
    """La operación superó el tiempo máximo."""


class LlmConnectionError(LlmServerError):
    """Fallo de transporte o de red."""


class LlmTruncatedError(LlmResponseError):
    """La salida se cortó por límite de tokens; ``partial_content`` conserva lo recibido."""

    def __init__(self, message: str, *, partial_content: str | None = None, **kwargs: object) -> None:
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.partial_content = partial_content


class LlmRefusalError(LlmResponseError):
    """El modelo se negó a responder (``category`` según el proveedor)."""

    def __init__(self, message: str, *, category: str | None = None, **kwargs: object) -> None:
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
        self.category = category


class LlmProtocolError(LlmResponseError):
    """La salida no cumple el protocolo interno de NAS (decisión/herramientas emuladas)."""


class LlmProcessError(LlmError):
    """Ejecutable ausente, versión incompatible o proceso local fallido."""


class LlmBillingModeError(LlmAuthError):
    """La configuración podría facturar por API cuando se pidió modo suscripción."""


class LlmIsolationError(LlmError):
    """El proceso local arrancó con capacidades que el adaptador exige desactivadas."""


__all__ = [
    "LlmAuthError",
    "LlmBillingModeError",
    "LlmConnectionError",
    "LlmError",
    "LlmIsolationError",
    "LlmModelError",
    "LlmPermissionError",
    "LlmProcessError",
    "LlmProtocolError",
    "LlmQuotaError",
    "LlmRateLimitError",
    "LlmRefusalError",
    "LlmRequestError",
    "LlmResponseError",
    "LlmServerError",
    "LlmTimeoutError",
    "LlmTruncatedError",
    "LlmUnsupportedError",
]
