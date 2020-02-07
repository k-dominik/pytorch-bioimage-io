# TODO would be nice to have something similar to inferno's batch_function etc.
# functionality, but more flexible w.r.t implementing different dimensions.

import pybio.core.transformations


class PyBioTorchTransformation(pybio.core.transformations.PyBioTransformation):
    pass


class PyBioTorchCombinedTransformation(pybio.core.transformations.CombinedPyBioTransformation):
    pass


class PyBioTorchSynchronizedTransformation(pybio.core.transformations.SynchronizedPyBioTransformation):
    pass
