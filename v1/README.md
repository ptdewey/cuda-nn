# V1
Version one switches from the base math exponential function `exp()` to `expf()` (the faster cuda version) in the sigmoid function.
<!-- TODO: test speed of just this function -->

In initial profiling, the binary cross-entropy (BCE) kernel showed to be the slowest one, so it was my first optimization target.
- I modified the function to switch from an atomic add to a rudimentary shared memory tree reduction.

