# V2.5
Version 2.5 tries to deal with the lack of improvement from v1 to v2 by reducing the number of shuffle operations performed, which should improve performance in cases with very small thread blocks.

(Changes to [bce_cost.cu](nn_utils/bce_cost.cu))
