from collections import namedtuple

import numpy as np

from pysc2.lib import actions
from pysc2.lib import features


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])

FLAT_FEATURES = [
  FlatFeature(0,  features.FeatureType.SCALAR, 3, 'message'),
  FlatFeature(1,  features.FeatureType.SCALAR, 1, 'prev_action'),
]