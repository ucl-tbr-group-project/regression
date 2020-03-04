import os


def get_model_factory():
    '''Get dictionary with model names and initializers.'''

    def init_nn(args):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(args))

    def init_svm(args):
        from models.svm import SupportVectorModel
        return SupportVectorModel(**SupportVectorModel.parse_cli_args(args))

    def init_knn(args):
        from models.knn import NearestNeighboursModel
        return NearestNeighboursModel(**NearestNeighboursModel.parse_cli_args(args))

    def init_gpr(args):
        from models.gpr import GaussianProcessModel
        return GaussianProcessModel(**GaussianProcessModel.parse_cli_args(args))

    def init_krg(args):
        from models.kriging import KrigingModel
        return KrigingModel(**KrigingModel.parse_cli_args(args))

    def init_rbf(args):
        from models.rbf import RBFModel
        return RBFModel(**RBFModel.parse_cli_args(args))

    def init_idw(args):
        from models.idw import IDWModel
        return IDWModel(**IDWModel.parse_cli_args(args))

    def init_rmts(args):
        from models.rmts import RMTSModel
        return RMTSModel(**RMTSModel.parse_cli_args(args))

    def init_ridge(args):
        from models.ridge import RidgeModel
        return RidgeModel(**RidgeModel.parse_cli_args(args))

    def init_sgd(args):
        from models.sgd import SGDModel
        return SGDModel(**SGDModel.parse_cli_args(args))

    return {
        'nn': init_nn,
        'svm': init_svm,
        'knn': init_knn,
        'gpr': init_gpr,
        'krg': init_krg,
        'rbf': init_rbf,
        'idw': init_idw,
        'rmts': init_rmts,
        'ridge': init_ridge,
        'sgd': init_sgd
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

    def load_svm(fname):
        from models.svm import SupportVectorModel
        return SupportVectorModel.load(
            '%s.svm.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_knn(fname):
        from models.knn import NearestNeighboursModel
        return NearestNeighboursModel.load(
            '%s.knn.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_gpr(fname):
        from models.gpr import GaussianProcessModel
        return GaussianProcessModel.load(
            '%s.gpr.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_krg(fname):
        from models.kriging import KrigingModel
        return KrigingModel.load(
            '%s.krg.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_rbf(fname):
        from models.rbf import RBFModel
        return RBFModel.load(
            '%s.rbf.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_idw(fname):
        from models.idw import IDWModel
        return IDWModel.load(
            '%s.idw.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_rmts(fname):
        from models.rmts import RMTSModel
        return RMTSModel.load(
            '%s.rmts.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_ridge(fname):
        from models.ridge import RidgeModel
        return RidgeModel.load(
            '%s.ridge.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    def load_sgd(fname):
        from models.sgd import SGDModel
        return SGDModel.load(
            '%s.sgd.pkl' % fname,
            scaler='%s.scaler.pkl' % fname)

    suffix_to_loader = {
        '.nn.h5': load_nn_full,
        '.nncp.h5': load_nn_cp_arch,
        '.svm.pkl': load_svm,
        '.knn.pkl': load_knn,
        '.gpr.pkl': load_gpr,
        '.krg.pkl': load_krg,
        '.rbf.pkl': load_rbf,
        '.idw.pkl': load_idw,
        '.rmts.pkl': load_rmts,
        '.ridge.pkl': load_ridge,
        '.sgd.pkl': load_sgd
    }

    loaded_model_name, loaded_model = None, None
    for suffix, loader in suffix_to_loader.items():
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
            loaded_model_name = os.path.basename(filename)
            loaded_model = loader(filename)

    return loaded_model_name, loaded_model
