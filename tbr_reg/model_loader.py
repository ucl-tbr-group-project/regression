import os


def get_model_factory():
    '''Get dictionary with model names and initializers.'''

    def init_nn(args):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(args))

    def init_svm(args):
        from models.svm import SupportVectorModel
        return SupportVectorModel(**SupportVectorModel.parse_cli_args(args))

    return {
        'nn': init_nn,
        'svm': init_svm
    }


def load_model_from_file(filename):
    '''Instantiate a trained model from a file previously created using .save().'''

    def load_nn_full(fname):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel.load(
            model='%s.nn.h5' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_nn_cp_arch(fname):
        from models.nn import NeuralNetworkModel
        fname_without_util = os.path.join(os.path.dirname(
            fname), os.path.basename(fname).split('.')[0])
        return NeuralNetworkModel.load(
            weights='%s.nncp.h5' % fname,
            arch='%s.arch.yml' % fname_without_util,
            scaler='%s.scaler.pkl' % fname_without_util)

    suffix_to_loader = {
        '.nn.h5': load_nn_full,
        '.nncp.h5': load_nn_cp_arch
    }

    loaded_model_name, loaded_model = None, None
    for suffix, loader in suffix_to_loader.items():
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            loaded_model_name = os.path.basename(filename)
            loaded_model = loader(filename)

    return loaded_model_name, loaded_model
