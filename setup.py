import io
import os
import sys

from setuptools import setup
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(name='musket_text',
      version='0.443',
      description='Text support for Musket ML',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/musket-ml/musket_text',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/aleju/imgaug'],
      install_requires=["musket_core","seqeval"],
      zip_safe=False)
