<p align="center">
  <img src="static/logo.svg" alt="warmpool" width="480">
</p>

<p align="center">
  <strong>A single-worker subprocess pool that can actually kill C extensions.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/warmpool/"><img alt="PyPI" src="https://img.shields.io/pypi/v/warmpool?color=ff6b35"></a>
  <a href="https://github.com/slopden/warmpool/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/slopden/warmpool?color=ff3d00"></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/warmpool?color=1a1a2e">
</p>

---

A "ProcessPool-like-executor" with hard-kill timeouts and import warming. The basic problem is that if you freeze up deep in a C-extension Python timeout-handling stuff doesn't work. `warmpool` runs functions in a spawned subprocess, and if they exceed their timeout it SIGTERM+SIGKILL the process and all children if the C extension has spawned anything.

- It calls a "warming function" in each new process, so you can have it keep a process warmed with `import scipy, numpy, etc` which can easily be 2+ seconds.
- The timeouts actually work regardless of what happens in the function.
- It has an option to keep a spare process warm in the background so it can rotate cleanly without eating an import period.
- It sends logs back to the parent through a pipe.

```python
import time
from warmpool import WarmPool

def warm_imports():
    import numpy
    import scipy.linalg

def eigh_huge(n=5000):
    """Stuck in LAPACK C code — only SIGKILL works."""
    import numpy as np
    from scipy import linalg
    a = np.random.rand(n, n)
    a = a + a.T
    return linalg.eigh(a)

def add(a=0, b=0):
    return a + b

pool = WarmPool(warming=warm_imports)

# numpy+scipy are already imported — no 2s wait
start = time.perf_counter()
try:
    pool.run(eigh_huge, timeout=0.5, n=5000)
except TimeoutError:
    print(f"killed after {time.perf_counter() - start:.2f}s")

# pool recovered via spare
result = pool.run(add, timeout=5.0, a=2, b=3)
print(f"recovered: add(2, 3) = {result}")
pool.shutdown()
```

```
killed after 0.53s
recovered: add(2, 3) = 5
```
