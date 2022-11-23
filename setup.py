from setuptools import setup

setup(
    name='powderworld',
    version='1.0.0',    
    description='A example Python package',
    url='https://github.com/kvfrans/powderworld',
    author='Kevin Frans',
    author_email='kevinfrans2@gmail.com',
    license='MIT',
    zip_safe=False,
    packages=['powderworld'],
    include_package_data=True,
    install_requires=['torch',
                      'numpy',
                      'stable_baselines3',
                      'gym',
                      'matplotlib',
                      'scikit-image',
                      ],
)
