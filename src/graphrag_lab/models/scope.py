"""Scope: the hierarchical addressing model for memories, entities, and relations."""

import re
from typing import cast

from pydantic import RootModel, field_validator

ScopeValue = str | int | float | bool

# tenant_id ends up inside Neo4j label and index names.
_SAFE_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class Scope(RootModel[dict[str, ScopeValue]]):
    """Where a memory lives: a `tenant_id` plus any other keys (user_id,
    project, ...).

    Each tenant has its own label and indexes in Neo4j (ADR 0004). The other
    keys are filtered in Python after retrieval. A scope includes a narrower
    one: `{tenant_id: acme, project: alpha}` includes
    `{tenant_id: acme, project: alpha, agent_id: researcher_1}`.
    """

    @field_validator("root")
    @classmethod
    def _validate(cls, value: dict[str, ScopeValue]) -> dict[str, ScopeValue]:
        if "tenant_id" not in value:
            raise ValueError("scope must contain a 'tenant_id' key")
        tenant_id = value["tenant_id"]
        if not isinstance(tenant_id, str) or not _SAFE_TENANT_ID.match(tenant_id):
            raise ValueError(
                f"tenant_id {tenant_id!r} must be a non-empty string matching "
                "^[A-Za-z0-9_-]+$ (it is embedded in Neo4j label and index names)"
            )
        for key in value:
            if not key:
                raise ValueError("scope keys must be non-empty strings")
        return value

    @property
    def tenant_id(self) -> str:
        return cast(str, self.root["tenant_id"])

    def includes(self, other: "Scope") -> bool:
        """True if `other` has every key-value pair this scope has."""
        if self.tenant_id != other.tenant_id:
            return False
        return all(other.root.get(key) == val for key, val in self.root.items())

    def to_query_dict(self) -> dict[str, ScopeValue]:
        return dict(self.root)

    @classmethod
    def from_dict(cls, data: dict[str, ScopeValue]) -> "Scope":
        return cls(data)
