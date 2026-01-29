#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===============================
HtmlTestRunner
===============================


.. image:: https://img.shields.io/pypi/v/peakpyp.svg
        :target: https://pypi.python.org/pypi/peakpyp
.. image:: https://img.shields.io/travis/pthemis5/peakpyp.svg
        :target: https://travis-ci.org/pthemis5/peakpyp

Containing Codes that were used for analyses, simulations, measurements and tests of the PeakSat mission


Links:
---------
* `Github <https://github.com/pthemis5/peakpyp>`_
"""

from setuptools import setup, find_packages

requirements = [ ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Themis Poultourtzidis",
    author_email='poultourd@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Containing Codes that were used for analyses, simulations, measurements and tests of the PeakSat mission",
    install_requires=requirements,
    license="MIT license",
    long_description=__doc__,
    include_package_data=True,
    keywords='peakpyp',
    name='peakpyp',
    packages=find_packages(include=['peakpyp']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pthemis5/peakpyp',
    version='0.1.0',
    zip_safe=False,
)
