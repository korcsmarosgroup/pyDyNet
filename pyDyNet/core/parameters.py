from dataclasses import dataclass

@dataclass()
class PyDynetParameters:
    Param_a: str = DefaultVal('Param_a')
    Param_b: int = DefaultVal(0)
