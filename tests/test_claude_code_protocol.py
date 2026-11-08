from __future__ import annotations

import json

import pytest
from jsonschema import Draft7Validator
from pydantic import BaseModel

from nas.adk.models import LlmProtocolError
from nas.adk.models.claude_code import protocol as proto
from nas.adk.tools.function_tool import FunctionTool


class Item(BaseModel):
    sku: str
    qty: int


def pedido(items: list[Item], nota: str | None = None) -> str:
    """Crea un pedido."""
    return "ok"


def registry() -> dict[str, dict]:
    tool = FunctionTool(pedido)
    return proto.tool_registry([tool.to_llm_tool()])


def test_discriminated_schema_lifts_defs_and_validates() -> None:
    reg = registry()
    assert "$defs" in reg["pedido"]  # pydantic genera $defs/$ref
    schema, mode = proto.build_output_schema(reg, None)
    assert mode == "discriminated"
    Draft7Validator.check_schema(schema)
    # La API rechaza anyOf/oneOf/allOf en la raíz del input_schema:
    assert schema["type"] == "object" and not {"anyOf", "oneOf", "allOf"} & set(schema)
    assert "$defs" not in json.dumps(schema)
    validator = Draft7Validator(schema)
    good = {
        "protocol_version": 1,
        "kind": "tool_calls",
        "calls": [{"name": "pedido", "arguments": {"items": [{"sku": "a", "qty": 1}]}}],
    }
    assert validator.is_valid(good)
    bad = json.loads(json.dumps(good))
    bad["calls"][0]["arguments"]["items"][0]["qty"] = "x"
    assert not validator.is_valid(bad)  # el $ref reescrito sigue validando el interior
    assert not validator.is_valid({**good, "extra": 1})
    assert not validator.is_valid({**good, "calls": [{"name": "otra", "arguments": {}}]})


def test_falls_back_to_transport_schema_when_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proto, "MAX_SCHEMA_BYTES", 50)
    schema, mode = proto.build_output_schema(registry(), None)
    assert mode == "transport"
    assert schema["properties"]["calls"]["items"]["properties"]["name"] == {"enum": ["pedido"]}
    # La validación local sigue siendo la estricta aunque el esquema sea simple:
    with pytest.raises(LlmProtocolError, match="Invalid arguments"):
        proto.parse_decision(
            {"protocol_version": 1, "kind": "tool_calls", "calls": [{"name": "pedido", "arguments": {"items": [{"sku": 1}]}}]},
            registry(),
            None,
            max_calls=3,
        )


def test_parse_valid_calls_resolves_refs_locally() -> None:
    decision = proto.parse_decision(
        {
            "protocol_version": 1,
            "kind": "tool_calls",
            "calls": [{"name": "pedido", "arguments": {"items": [{"sku": "a", "qty": 2}], "nota": None}}],
        },
        registry(),
        None,
        max_calls=3,
    )
    assert decision.calls[0].function["name"] == "pedido"


@pytest.mark.parametrize(
    "payload",
    [
        {"protocol_version": 1, "kind": "tool_calls", "calls": []},
        {"protocol_version": 1, "kind": "tool_calls", "calls": [{"name": "pedido"}]},
        {"protocol_version": 1, "kind": "tool_calls", "calls": [{"name": "pedido", "arguments": [], "x": 1}]},
        {"protocol_version": 1, "kind": "final"},
    ],
)
def test_malformed_payloads(payload: dict) -> None:
    with pytest.raises(LlmProtocolError):
        proto.parse_decision(payload, registry(), None, max_calls=3)


def test_no_tools_schema_is_final_only() -> None:
    schema, _ = proto.build_output_schema({}, None)
    assert schema["properties"]["kind"] == {"enum": ["final"]}
