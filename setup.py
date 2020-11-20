""" Packaging
"""
import setuptools

with open('requirements.txt') as fp:
    requirements = fp.readlines()

setuptools.setup(
    name="keras-list-mapper",
    version="0.0.1",
    author="Pedro Marques",
    author_email="pedro.r.marques@gmail.com",
    description="Keras RaggedTensor Mapper",
    url="https://github.com/pedro-r-marques/keras-list-mapper",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)
