# pose/config.py

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def __getitem__(self, item):
        return self.raw[item]
