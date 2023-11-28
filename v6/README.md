# V6
Version 6 is an alternative to version 4, switching from the warp-based shuffle reduction to a simple shared memory tree reduction.
As shown in binary classification results, the shared memory reduction is faster than the shuffle version, I wanted to test if that would still be the case in a multiclass situation.
