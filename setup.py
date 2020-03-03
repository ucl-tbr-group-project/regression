from setuptools import setup, find_packages

setup(
    name='tbr_reg',
    version='1.0.0',
    author='Petr Mánek, Graham Van Goffrier',
    packages=find_packages(exclude=['*tests']),
    install_requires=['numpy', 'matplotlib',
                      'pandas', 'scikit-learn', 'keras', 'pyqt5', 'joblib'],
    entry_points={
        'console_scripts': [
            'tbr_train = tbr_reg.run_training:main'
        ]}
)
