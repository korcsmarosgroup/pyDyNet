from dataclasses import dataclass
from typing_extensions import ParamSpec

@dataclass(init=True)
class PyDynetParameters:
    Param_a: str
    Param_b: int