from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True, extra="forbid")
