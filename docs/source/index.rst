======================================
Welcome to mckit-meshes documentation!
======================================

.. note::

   This documentation is currently under active development.

Overview
========

The package provides methods to read, write and process MCNP tally and weight meshes.

That includes: ::

    - comparison
    - merge
    - inversion and conversion tally to weight mesh
    - normalization
    - rebinning over spatial and energy bins
    - extraction data for specific spatial points and slices




Installation
============

**From PyPI (Recommended):**

.. code-block:: bash
   
   pip install mckit-meshes
   # or 
   uv pip install mckit-meshes

**With package manager (as a dependency):**

.. code-block:: bash
   
   # uv
   uv add mckit-nuclides
   
   # pixi
   pixi add --pypi mckit-meshes

   # poetry
   poetry add mckit-meshes

   # ...

From source:

.. code-block:: bash

   uv pip install https://github.com/MC-kit/mckit-meshes.git
   # or
   pip install https://github.com/MC-kit/mckit-meshes.git

Details
=======

.. toctree::
   :maxdepth: 2

   readme
   modules
   license
   todo



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
