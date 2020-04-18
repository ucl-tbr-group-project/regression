from setuptools import setup, find_packages

setup(
    name='tbr_reg',
    version='1.0.0',
    author='Petr Mánek, Graham Van Goffrier',
    packages=find_packages(exclude=['*tests']),
    install_requires=['numpy', 'matplotlib',
                      'pandas', 'scikit-learn',
                      'keras', 'pyqt5', 'joblib',
                      'scikit-optimize', 'smt',
                      'tensorflow'],
    entry_points={
        'console_scripts': [
            'tbr_train = tbr_reg.endpoints.training:main',
            'tbr_ae = tbr_reg.endpoints.autoencoder:main',
            'tbr_split = tbr_reg.endpoints.split_batches:main',
            'tbr_eval = tbr_reg.endpoints.evaluation:main',
            'tbr_eval_ho = tbr_reg.endpoints.evaluation_hyperopt:main',
            'tbr_eval_benchmark = tbr_reg.endpoints.evaluation_benchmark:main',
            'tbr_search = tbr_reg.endpoints.search:main',
            'tbr_search_benchmark = tbr_reg.endpoints.search_benchmark:main',
            'tbr_qass1 = tbr_reg.endpoints.qass_v1:main',
            'tbr_qass2 = tbr_reg.endpoints.qass_v2:main',
            'tbr_qass3 = tbr_reg.endpoints.qass_v3:main',
            'tbr_fakeqass = tbr_reg.endpoints.qass_fake:main'
        ],
        'gui_scripts': [
            'tbr_visualizer = tbr_reg.endpoints.visualizer:main'
        ]}
)
