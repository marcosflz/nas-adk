"""«Sign in with ChatGPT» para que cada persona use su propio plan de ChatGPT.

Sigue la guía oficial de OpenAI para apps de terceros (OAuth 2 con PKCE, sin secreto de
cliente). La redirección es siempre de loopback (``http://127.0.0.1:1455/auth/callback``). Si la
app no corre en el mismo equipo que el navegador, esa página no carga: la persona copia su
dirección y la app la pasa a :func:`exchange_code`.

Funciones puras + httpx; quien llama guarda las credenciales (cifradas y por persona) y
serializa los refrescos de una misma cuenta, porque el token de refresco rota en cada uso.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from nas.adk.models.chatgpt.llm import API_BASE, parse_models
from nas.adk.models.errors import LlmAuthError, LlmPermissionError, LlmServerError

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
OPENID_CONFIG_URL = "https://auth.openai.com/.well-known/openid-configuration"
REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"
RESOURCE = API_BASE
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REQUIRED_SCOPE = "chatgpt.tokens.use.direct"
REGISTRATION_CLIENT = "dynamic_agent_client"
TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)


@dataclass(frozen=True)
class PendingLogin:
    """Lo que hay que recordar entre abrir la URL y recibir la redirección. No sale del servidor."""

    state: str
    nonce: str
    verifier: str
    client_id: str
    url: str


@dataclass(frozen=True)
class ChatGptCredentials:
    client_id: str
    access_token: str
    refresh_token: str
    expires_at: float
    """Segundos epoch en los que caduca ``access_token``."""
    account: str | None = None
    """Correo (o ``sub``) de la cuenta, solo para mostrarlo."""

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, raw: str) -> ChatGptCredentials:
        data = json.loads(raw)
        return cls(
            client_id=str(data["client_id"]),
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),
            account=data.get("account"),
        )

    def expires_within(self, seconds: float) -> bool:
        return time.time() + seconds >= self.expires_at


def new_host_id() -> str:
    """Identificador opaco del host (``ext_agent_host_id``): uno por instalación, nunca por persona."""
    import uuid

    return f"urn:uuid:{uuid.uuid4()}"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def start_login(*, host_id: str, app_name: str, client_id: str | None = None) -> PendingLogin:
    """URL de autorización (PKCE S256). ``client_id`` es el que OpenAI emitió antes a esta persona."""
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    params = {
        "client_id": client_id or REGISTRATION_CLIENT,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if not client_id:
        params["agent_name_hint"] = app_name
        params["ext_agent_host_id"] = host_id
    return PendingLogin(
        state=state,
        nonce=nonce,
        verifier=verifier,
        client_id=client_id or REGISTRATION_CLIENT,
        url=f"{AUTHORIZE_URL}?{urlencode(params)}",
    )


def parse_redirect(pasted: str, pending: PendingLogin) -> tuple[str, str]:
    """Extrae ``code`` y el ``client_id`` emitido de la dirección pegada. Valida ``state``."""
    parts = urlsplit(pasted.strip())
    query = parse_qs(parts.query)
    if f"{parts.scheme}://{parts.netloc}{parts.path}" != REDIRECT_URI:
        raise LlmAuthError("That is not the address of the ChatGPT sign-in page (127.0.0.1:1455)")
    if "error" in query:
        raise LlmPermissionError(f"ChatGPT sign-in was not completed: {query['error'][0]}")
    state = (query.get("state") or [""])[0]
    if not secrets.compare_digest(state, pending.state):
        raise LlmAuthError("The address is from another sign-in attempt; start again")
    code = (query.get("code") or [""])[0]
    if not code:
        raise LlmAuthError("The address does not contain an authorization code")
    scope = (query.get("scope") or [""])[0]
    if scope and REQUIRED_SCOPE not in scope.split():
        raise LlmPermissionError("ChatGPT plan usage was not enabled for this app")
    client_id = (query.get("client_id") or [pending.client_id])[0]
    return code, client_id


def _jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError) as error:
        raise LlmAuthError("OpenAI returned an invalid ID token") from error
    if not isinstance(claims, dict):
        raise LlmAuthError("OpenAI returned an invalid ID token")
    return claims


async def _token_request(client: httpx.AsyncClient, form: dict[str, str]) -> dict[str, Any]:
    try:
        response = await client.post(
            TOKEN_URL,
            data=form,
            headers={"Accept": "application/json"},
            timeout=30,
        )
    except httpx.HTTPError as error:
        raise LlmServerError("Could not reach OpenAI sign-in", retryable=True, cause=error) from error
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code >= 400 or not isinstance(data, dict) or "access_token" not in data:
        code = str(data.get("error") or "") if isinstance(data, dict) else ""
        if code in TERMINAL_REFRESH_ERRORS or response.status_code in (400, 401):
            raise LlmAuthError(f"OpenAI rejected the sign-in ({code or response.status_code})", body=code or None)
        raise LlmServerError(f"OpenAI sign-in failed ({response.status_code})", retryable=True)
    return data


def _credentials(data: dict[str, Any], client_id: str, account: str | None) -> ChatGptCredentials:
    granted = str(data.get("scope") or "")
    if granted and REQUIRED_SCOPE not in granted.split():
        raise LlmPermissionError("ChatGPT plan usage was not enabled for this app")
    if not data.get("refresh_token"):
        raise LlmAuthError("OpenAI did not return a refresh token")
    return ChatGptCredentials(
        client_id=client_id,
        access_token=str(data["access_token"]),
        refresh_token=str(data["refresh_token"]),
        expires_at=time.time() + float(data.get("expires_in") or 3600),
        account=account,
    )


async def exchange_code(
    pasted_redirect: str,
    pending: PendingLogin,
    *,
    client: httpx.AsyncClient | None = None,
) -> ChatGptCredentials:
    """Canjea el código de la dirección pegada por credenciales de esa persona."""
    code, client_id = parse_redirect(pasted_redirect, pending)
    async with _client(client) as http:
        data = await _token_request(
            http,
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": pending.verifier,
                "redirect_uri": REDIRECT_URI,
                "resource": RESOURCE,
            },
        )
    account = None
    if data.get("id_token"):
        claims = _jwt_claims(str(data["id_token"]))
        # La firma la garantiza TLS con auth.openai.com (respuesta directa al canje con PKCE);
        # aquí se comprueban los vínculos con esta petición.
        if claims.get("nonce") != pending.nonce:
            raise LlmAuthError("The ID token does not match this sign-in attempt")
        if float(claims.get("exp") or 0) < time.time():
            raise LlmAuthError("The ID token has expired")
        account = str(claims.get("email") or claims.get("sub") or "") or None
    return _credentials(data, client_id, account)


async def refresh(
    credentials: ChatGptCredentials,
    *,
    client: httpx.AsyncClient | None = None,
) -> ChatGptCredentials:
    """Nuevo token de acceso. Devuelve también el token de refresco rotado: guárdalo siempre.

    Un ``LlmAuthError`` significa que hay que volver a iniciar sesión.
    """
    async with _client(client) as http:
        data = await _token_request(
            http,
            {
                "grant_type": "refresh_token",
                "client_id": credentials.client_id,
                "refresh_token": credentials.refresh_token,
                "resource": RESOURCE,
            },
        )
    return _credentials(data, credentials.client_id, credentials.account)


async def revoke(credentials: ChatGptCredentials, *, client: httpx.AsyncClient | None = None) -> None:
    """Revoca el token de refresco (mejor esfuerzo)."""
    async with _client(client) as http:
        try:
            config = (await http.get(OPENID_CONFIG_URL, timeout=15)).json()
            endpoint = config.get("revocation_endpoint") if isinstance(config, dict) else None
            if isinstance(endpoint, str) and endpoint.startswith("https://auth.openai.com/"):
                await http.post(
                    endpoint,
                    data={
                        "token": credentials.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": credentials.client_id,
                    },
                    timeout=15,
                )
        except (httpx.HTTPError, ValueError):
            pass


async def list_models(access_token: str, *, client: httpx.AsyncClient | None = None) -> list[dict[str, str]]:
    """Modelos del plan: ``[{id, label}]`` en el orden que da OpenAI."""
    async with _client(client) as http:
        try:
            response = await http.get(
                f"{API_BASE}/models",
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
                timeout=20,
            )
        except httpx.HTTPError as error:
            raise LlmServerError("Could not reach OpenAI", retryable=True, cause=error) from error
    if response.status_code == 401:
        raise LlmAuthError("OpenAI rejected the ChatGPT access token")
    if response.status_code >= 400:
        raise LlmServerError(f"OpenAI returned {response.status_code} listing models", retryable=True)
    data = response.json()
    return parse_models(data if isinstance(data, dict) else {})


class _client:
    """Usa el cliente dado o crea uno temporal."""

    def __init__(self, client: httpx.AsyncClient | None) -> None:
        self._given = client
        self._own: httpx.AsyncClient | None = None

    async def __aenter__(self) -> httpx.AsyncClient:
        if self._given is not None:
            return self._given
        self._own = httpx.AsyncClient(follow_redirects=False)
        return self._own

    async def __aexit__(self, *exc: object) -> None:
        if self._own is not None:
            await self._own.aclose()
