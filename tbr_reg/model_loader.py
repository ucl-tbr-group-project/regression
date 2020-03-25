import os


def create_factories():
    factories = []

    from .models.svm import SupportVectorFactory
    factories.append(SupportVectorFactory())

    from .models.rf import RandomForestFactory
    factories.append(RandomForestFactory())

    from .models.knn import NearestNeighboursFactory
    factories.append(NearestNeighboursFactory())

    from .models.gpr import GaussianProcessFactory
    factories.append(GaussianProcessFactory())

    from .models.kriging import KrigingFactory
    factories.append(KrigingFactory())

    from .models.rbf import RBFFactory
    factories.append(RBFFactory())

    from .models.idw import IDWFactory
    factories.append(IDWFactory())

    from .models.rmts import RMTSFactory
    factories.append(RMTSFactory())

    from .models.ridge import RidgeFactory
    factories.append(RidgeFactory())

    from .models.sgd import SGDFactory
    factories.append(SGDFactory())

    from .models.nn import NeuralNetworkSavedModelFactory, NeuralNetworkCheckpointFactory
    factories.append(NeuralNetworkSavedModelFactory())
    factories.append(NeuralNetworkCheckpointFactory())

    return factories


def get_model_factory():
    '''Get dictionary with model names and initializers.'''
    return {factory.extension: factory.init_model for factory in create_factories()}


def load_model_from_file(filename):
    '''Instantiate a trained model from a file previously created using .save().'''

    suffix_to_loader = {factory.extension: factory.load_model
                        for factory in create_factories()}

    loaded_model_name, loaded_model = None, None
    for suffix, loader in suffix_to_loader.items():
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            loaded_model_name = os.path.basename(filename)
            loaded_model = loader(filename)

    return loaded_model_name, loaded_model
