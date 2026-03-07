# warmpool

A "ProcessPool-like-executor" with hard-kill timeouts and import warming. The basic problem is if you freeze up deep in a C-extension most of the Python timeout-handling stuff doesn't work. `warmpool` runs functions in a SINGLE subprocess, and if they exceed their timeout it SIGTERM+SIGKILL's them and all of their children (if the C extension has spawned anything). 

- It has a "module warming" mode, so you can have it keep a process warmed with `import scipy, numpy, etc` which can easily be 2+ seconds. 
- It has an option to keep a spare process warm so it can rotate cleanly without eating an import period.
- It sends logs back to the parent through a pipe.
