from .CBankGRUModel import CBankGRUModel


def get_model(model_name):
    return globals()[model_name]
