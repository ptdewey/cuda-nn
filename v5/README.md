# V5
Version 5 finally tries to tackle speeding up the matrix multiplication kernels (linear_layer.cu), namely forward and backward propagation, as well as the weight and bias update kernels.
To do this, I used the cuBLAS library sGemm function. However, it ended up being slower (might be user error) than my naive implementation.
