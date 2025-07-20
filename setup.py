from setuptools import setup, find_packages

setup(
    name='purkinje-learning-myocardial-mesh',
    version='0.1.0',
    description='Ventricular myocardial mesh construction for ECG and FIM simulation',
    author='Ricardo G.',
    author_email='your.email@example.com',
    url='https://github.com/ricardogr07/purkinje-learning-myocardial-mesh',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'vtk',
        'pyvista',
        'meshio',
        'fimpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
