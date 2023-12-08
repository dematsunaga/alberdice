from .optidice_learner import OptiDiceLearner
from .alberdice_learner import AlberDiceLearner

REGISTRY = {}

REGISTRY["optidice_learner"] = OptiDiceLearner
REGISTRY["alberdice_learner"] = AlberDiceLearner


