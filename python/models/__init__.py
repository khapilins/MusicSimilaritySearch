from .CBankGRUModel import CBankGRUModel
from .ConvModel import ConvModel


def get_model(model_name):
    return globals()[model_name]
