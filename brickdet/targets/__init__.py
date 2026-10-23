"""
Export targets, one module per runtime. A target module provides:

  IO = {'input': name, 'output': name}   tensor names in the exported model
  export(net, x, out) -> [paths]         write the model for torch module `net` and example input `x`
  load(out) -> run                       run(x: float32 [1, 3, H, W]) -> float32 [1, N, C]
"""
import pkgutil

TARGETS = [m.name for m in pkgutil.iter_modules(__path__)]
