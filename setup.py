from setuptools import setup, find_packages

classfiers = [
    'Development Status :: In production :: Pre-release',
    'Programming language :: Python :: 3.8',
    'Operating System :: OS independent'
]

setup(
    name='mlmnemonist',
    version='0.1.0',
    description='A light-weight framework to store the progress you made'
                'on your ML operations with the ability to smartly cache your models'
                'and retrieve it even when your session crashes.',
    long_description=open('README.rst').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/HamidrezaKmK/ML-Mnemonist',
    author='Hamidreza Kamkari',
    author_email='hamidrezakamkari@gmail.com',
    license='MIT',
    classfiers=classfiers,
    keywords='MLOps, Caching',
    packages=find_packages(),
    install_requires=['python-dotenv', 'yacs']
)