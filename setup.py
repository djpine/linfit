#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The Lesser GNU Public License Version 2 (or later)

from setuptools import setup, find_packages

setup(
    author="David Pine",
    description="Least squares linear fit for numpy library of Python",
    license="LGPL2",
    name="linfit",
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    url="https://github.com/djpine/linfit",
    download_url="https://github.com/djpine/linfit",
    version="2013.11.29",
)
