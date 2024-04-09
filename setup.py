from setuptools import setup, find_packages

setup(
    name='robonlu',  # Replace with your package name
    version='1.0',  # Replace with your package version
    description='The RoboNLU package for Natural Command Understanding',  # Replace with your package description
    author='Maëlic Neau & Sinuo Wang',  # Replace with your name
    author_email='neau@enib.fr',  # Replace with your email
    url='https://github.com/RoboBreizh-RoboCup-Home/RoboNLU',  # Replace with the URL of your package's source code
    packages=find_packages(exclude=("data", "logs",)),  # Automatically find all packages and subpackages
    install_requires=[
        # List all dependencies here
        # For example:
        # 'numpy',
        # 'pandas',
    ],
    classifiers=[
        # Classifiers help users find your project by categorizing it
        # For a list of valid classifiers, see https://pypi.org/classifiers/
        'Development Status :: 5',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)