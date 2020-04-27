import setuptools

setuptools.setup(
    name="torchdyn",
    version="0.1.1",
    author="DiffEqML",
    author_email="polimic03@gmail.com, massaroli@robot.t.u-tokyo.ac.jp",
    description="PyTorch package for all things neural differential equations",
    url="https://github.com/Zymrael/torchdyn",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.4.1',
                      'pytorch-lightning>=0.7.3',
                      'dgl>=0.4.1',
                      'torchdiffeq>=0.0.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)