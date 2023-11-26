# V4
Version 4 expands upon version 3, mainly focusing on optimizing the mean square error cost function and its derivative. 
These additions resulted in a very noticeable speedup.
Additionally, I originally intended to speed up the `exp_sum()` kernel in the softmax layer, but I ended up just letting each thread compute its own exponential row sum (what exp_sum was doing) within the softmax kernels. This improved the speed by quite a bit, likely due to just removing some kernel launching overhead (went from ~5 microseconds to ~3).

Version 4 is also when I added the cost function interface, to make switching between cost functions substantially easier, which was very helpful for testing. (This change has been extended to all previous versions as well)
