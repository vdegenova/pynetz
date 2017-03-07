from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pynetz',
      version='0.1',
      description='A simple yet powerful Python Neural Network Implementation',
      url='http://github.com/vdegenova/pynetz',
      author='Vinny DeGenova',
      author_email='vdegenova@gmail.com',
      license='MIT',
      packages=['pynetz'],
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      zip_safe=False,
      include_package_data=True)