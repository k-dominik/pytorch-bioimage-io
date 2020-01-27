# TODO would be nice to have something similar to inferno's batch_function etc.
# functionality, but more flexible w.r.t implementing different dimensions.

import pybio.core.transformations


class Transformation(pybio.core.transformations.Transformation):
    pass

class CombinedTransformation(pybio.core.transformations.CombinedTransformation):
    pass

class SynchronizedTransformation(pybio.core.transformations.SynchronizedTransformation):
    pass
