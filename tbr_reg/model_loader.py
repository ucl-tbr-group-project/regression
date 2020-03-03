def get_model_factory():
    def init_nn(args):
        from models.nn import NeuralNetworkModel
        return NeuralNetworkModel(**NeuralNetworkModel.parse_cli_args(args))

    return {
        'nn': init_nn
    }
