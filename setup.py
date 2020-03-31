from setuptools import setup, find_packages

setup(
    name='tbr_reg',
    version='1.0.0',
    author='Petr Mánek, Graham Van Goffrier',
    packages=find_packages(exclude=['*tests']),
    install_requires=['numpy', 'matplotlib',
                      'pandas', 'scikit-learn',
                      'keras', 'pyqt5', 'joblib',
                      'scikit-optimize', 'smt'],
    entry_points={
        'console_scripts': [
            'tbr_train = tbr_reg.endpoints.training:main',
            'tbr_ae = tbr_reg.endpoints.autoencoder:main',
            'tbr_split = tbr_reg.endpoints.split_batches:main',
            'tbr_eval = tbr_reg.endpoints.evaluation:main',
            'tbr_search = tbr_reg.endpoints.search:main',
            'tbr_qass = tbr_reg.endpoints.qass:main'
        ],
        'gui_scripts': [
            'tbr_visualizer = tbr_reg.endpoints.visualizer:main'
        ]}
)
