# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'Keras>=2.2.0',
    'pronouncing>=0.2.0',
    'scikit-learn>=0.20.0'
]

setup(
    name='pincelate',
    version='0.0.1',
    description="Easy to use ML model for spelling and sounding out words",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Allison Parrish",
    author_email='allison@decontextualize.com',
    url='https://github.com/aparrish/pincelate',
    packages=['pincelate'],
    package_data={
        '': ['LICENSE'],
        'pincelate': ['models/*']
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='pincelate',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Artistic Software",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    test_suite='tests'
)
