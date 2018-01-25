#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__author__ = 'Yoshi Ri'


setup(
    name='ImRegPOC',  
    version='1.0',  
    description='Image Registration Tool using 2D FFT',  
    author='Yoshi Ri', 
    author_email='yoshiyoshidetteiu@gmail.com',  
    url='https://github.com/YoshiRi/ImRegPOC', 
    classifiers=[ 
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(), 
    include_package_data=True,  
    keywords=['Image Registration', 'POC', 'FFT'], 
    license='BSD License', 
    install_requires=[ 
        'opencv-python',
        'opencv-contrib-python',
        'numpy',
        'matplotlib',
    ],
    entry_points="""  # コマンドラインにするときのエントリーポイント、pitchpx/__init__.pyの関数をエントリーポイントにしました.
        [console_scripts]
        imregpoc = imregpoc:main
    """,
)