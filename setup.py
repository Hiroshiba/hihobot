from setuptools import setup, find_packages

setup(
    name='hihobot',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/hihobot',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    install_requires=[
        'chainer',
        'ndjson',
        'gensim',
        'janome',
        'tensorboard-chainer',
        'pillow',
    ],
)
