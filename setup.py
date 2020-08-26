# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

setuptools.setup(
    name="torchdyn",
    version="0.2.0",
    author="DiffEqML",
    author_email="polimic03@gmail.com, massaroli@robot.t.u-tokyo.ac.jp",
    description="PyTorch package for all things neural differential equations",
    url="https://github.com/DiffEqML/torchdyn",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.6.0',
                      'pytorch-lightning>=0.8.4',
                      'dgl>=0.4.1',
                      'torchdiffeq>=0.0.1',
                      'matplotlib',
                      'torchvision',
                      'scikit-learn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
