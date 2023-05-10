from setuptools import setup

setup(name='up_ibacop',
      version='0.0.1',
      description='up_ibacop',
      author='Isabel Cenamor and Tomas de la Rosa and Fernando Fernandez',
      author_email='icenamorg@gmail.com',
      packages=["up_ibacop"],
      install_requires=["platform_system == 'Linux'"],
      python_requires='>=3.7',
      license='APACHE')
