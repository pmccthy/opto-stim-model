"""
Models classes for opsin-expressing V1 neurons.
Author: patrick.mccarthy@dpag.ox.ac.uk
"""

import brian2 as b2
import numpy as np

class Model:
    
    """
    Need to be able to specify:
    - num exc neurons
    - num inh neurons
    - delay distribution
    - weight distribution
    - connection probability
    - opsin input model (current based, single exp conductance-based or bi-exp)
    - synapse models
    - synapse time constants
    - membrane time constant
    """
    pass