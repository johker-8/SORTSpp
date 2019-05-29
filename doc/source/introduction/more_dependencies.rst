Optional dependencies
===========================

.. include:: ../../../pip_req_optional.txt
   :literal:

Basemap
----------

To install **basemap**:

Install proj-bin:

.. code-block:: bash

   sudo apt-get install proj-bin

Get the basemap source

.. code-block:: bash

   wget --no-check-certificate https://github.com/matplotlib/basemap/archive/master.tar.gz

Un-tar the basemap version X.Y.Z source tar.gz file, and enter the basemap-X.Y.Z directory

.. code-block:: bash

   export GEOS_DIR=<where you want the libs and headers to go>

Then go to the geos distribution in the un-tar'ed basemap and run

.. code-block:: bash

   ./configure --prefix=$GEOS_DIR
   make; make install

Lastly be sure to use the python bin in the virtualenv to install basemap as:: bash

   /..../SORTSpp/env2.7/bin/python setup.py install


Pyglow
--------

Follow the installation guide on `Pyglow <https://github.com/timduly4/pyglow>`_.


