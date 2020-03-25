def create_factories():
    factories = []

    from .metrics.mae import MAEFactory
    factories.append(MAEFactory())

    from .metrics.r2 import R2Factory
    factories.append(R2Factory())

    return factories


def get_metric_factory():
    '''Get dictionary with metric names and initializers.'''
    return {factory.id: factory.init_metric for factory in create_factories()}
