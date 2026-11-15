"""Modelos que anuncia el CLI de Claude Code 2.1.319 (la versión probada) con un login completo.

Con un token de ``claude setup-token`` el CLI no conoce el plan de la cuenta y solo anuncia los
alias (default, opus, sonnet, haiku, fable). Estos ids concretos se aceptan igualmente en la
inferencia, así que se ofrecen como «conocidos»: puede que alguno no esté en el plan de la persona.
Actualízalo al subir de versión el CLI (``ClaudeCodeLlm().list_models()`` con un login completo).
"""

from __future__ import annotations

KNOWN_MODELS: tuple[tuple[str, str, str], ...] = (
    ("claude-fable-5-2", "Fable 5.2", "For your toughest challenges"),
    ("claude-fable-5", "Fable 5", "Most capable for your hardest and longest-running tasks"),
    ("claude-opus-5", "Opus 5", "Best for everyday, complex tasks"),
    ("claude-sonnet-5", "Sonnet 5", "Efficient for routine tasks"),
    ("claude-opus-4-8", "Opus 4.8", "Best for everyday, complex tasks"),
    ("claude-opus-4-7", "Opus 4.7", "Best for everyday, complex tasks"),
    ("claude-opus-4-6", "Opus 4.6", "Best for everyday, complex tasks"),
    ("claude-sonnet-4-6", "Sonnet 4.6", "Efficient for routine tasks"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5", "Fastest for quick answers"),
)
