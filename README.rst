==============================================================================
*mckit_meshes*: to work with MCNP mesh tallies and weight meshes
==============================================================================



|Maintained| |License| |Versions| |PyPI| |Docs|

.. contents::


Description
-----------

The module implements methods to read and manipulate (merge, inverse, scale, etc.)
MCNP mesh tallies and weight meshes.

More details in documentation_.


Contributing
------------

.. image:: https://github.com/MC-kit/mckit-meshes/workflows/Tests/badge.svg
   :target: https://github.com/MC-kit/mckit-meshes/actions?workflow=Tests
   :alt: Tests

..  check why the codecov site is not accessible

.. image:: https://codecov.io/gh/MC-kit/mckit-meshes/branch/master/graph/badge.svg?token=wlqoa368k8
   :target: https://codecov.io/gh/MC-kit/mckit-meshes
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
   :target: https://github.com/astral-sh/uv


Some specific: in development environment we use uv_, just_, ruff_.

To setup development environment, run:

.. code-block:: shell

  just install | reinstall

To build documentation, run:

.. code-block:: shell

   just docs        # - for local online docs rendering, while editing 
   just docs-build  # - to build documentation 

To release, run:

.. code-block:: shell

  just bump [major|minor|patch]  # - in `devel` branch
  
Then merge devel to master (via Pull Request) and if all the checks are passed create Release. Manually.


.. ... with /home/dvp/.julia/dev/Tools.jl/scripts/extract-half-lives.jl (nice script by the way).

References
----------

.. todo::

   dvp: add references to f4enix, mckit, etc

.. Links

.. _documentation: https://mckit-meshes.readthedocs.io/en/latest
.. _uv: https://github.com/astral-sh/uv
.. _just: https://github.com/casey/just
.. _ruff: https://github.com/astral-sh/ruff


.. Substitutions

.. |Maintained| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/MC-kit/mckit-meshes/graphs/commit-activity
.. |Tests| image:: https://github.com/MC-kit/mckit-meshes/workflows/Tests/badge.svg
   :target: https://github.com/MC-kit/mckit-meshes/actions?workflow=Tests
   :alt: Tests
.. |License| image:: https://img.shields.io/github/license/MC-kit/mckit-meshes
   :target: https://github.com/MC-kit/mckit-meshes
.. |Versions| image:: https://img.shields.io/pypi/pyversions/mckit-meshes
   :alt: PyPI - Python Version
.. |PyPI| image:: https://img.shields.io/pypi/v/mckit-meshes
   :target: https://pypi.org/project/mckit-meshes/
   :alt: PyPI
.. |Docs| image:: https://readthedocs.org/projects/mckit-meshes/badge/?version=latest
   :target: https://mckit_meshes.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
