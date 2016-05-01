#! /usr/bin/env python
from setuptools import setup

descr = """IO and analysis package built to streamline
    and standardize the data handling, analysis, and
    visualization of electrophysiology and calcium imaging data.
    """

DISTNAME = 'neurphys'
DESCRIPTION = descr
MAINTAINER = 'Chad Estep, Dan Galtieri'
MAINTAINER_EMAIL = 'chadestep@u.northwestern.edu'
LICENSE = 'GPL'
DOWNLOAD_URL = 'https://github.com/surmeierlab/neurphys.git'
VERSION = '0.0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          platforms='any',
          packages=['neurphys'],
          )
