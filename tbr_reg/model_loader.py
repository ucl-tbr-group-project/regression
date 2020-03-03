import os


def get_model_factory():
    def init_nn(args):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(args))

    return {
        'nn': init_nn
    }


def load_model_from_file(filename):
    def load_nn_full(fname):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel.load(
            model='%s.nn.h5' % fname,
            scaler='%s.scaler.pkl' % fname)

    suffix_to_loader = {
        '.nn.h5': load_nn_full
    }

    loaded_model_name, loaded_model = None, None
    for suffix, loader in suffix_to_loader.items():
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            loaded_model_name = os.path.basename(filename)
            loaded_model = loader(filename)

    return loaded_model_name, loaded_model
