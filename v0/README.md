# V0
This code serves as the starting point for my project and was sourced from 
[this guide](https://luniak.io/cuda-neural-network-implementation-part-1/)
and [this repo](https://github.com/pwlnk/cuda-neural-network)

Later Note: A fair amount of modifications were made to it later for expansion, profiling, and testing purposes.  

These modifications included:
- Implementation of an abstract class for cost functions in [cost.hh](nn_utils/cost.hh) (To allow easy switching of cost functions) and other adaptations necessary for it to function.
- Addition of an alternative dataset option (MNIST handwritten digits, 0s and 1s).
- Ability to import and run a testing dataset (for better quantifying accuracy).
- Addition of a makefile to allow for easier compilation and cleanup
- Modification of the `computeAccuracy` function to allow for other datasets and multiclass classification (this is mainly for version 3+)

