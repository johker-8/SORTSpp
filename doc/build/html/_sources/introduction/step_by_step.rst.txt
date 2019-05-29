
Step by step guides
========================

Step by step: fresh Ubuntu 16.04 LTS
------------------------------------------

If needed:

.. code-block:: bash

   sudo dpkg --configure -a

Proceed to:

.. code-block:: bash

   sudo apt-get install git
   cd /my/projects_dir/
   git clone https://gitlab.irf.se/danielk/SORTSpp.git
   cd SORTSpp/

Check your currently installed python versions:

.. code-block:: bash

   python --version

If this does NOT return `Python 2.7.x`:

.. code-block:: bash

   sudo apt-get install python-dev

Check that pip is installed and bound to your python 2.7:

.. code-block:: bash

   pip --version

If pip is NOT installed:

.. code-block:: bash

   sudo apt-get install python-pip

At this stage: DO NOT UPGRADE PIP. Do this after the virtualenv is installed and activated.

Then install and create a virtualenv, here the name "env2.7" is used since this name is included in the .gitignore file and will not be detected by git:

.. code-block:: bash

   pip install virtualenv
   virtualenv --version
   virtualenv env2.7

Activate virtualenv:

.. code-block:: bash

   source env2.7/bin/activate

If needed (should already be latest version), upgrade the pip inside the virtualenv:

.. code-block:: bash

   pip install --upgrade pip

Then make sure additional requirements are fulfilled:

* Used by matplotlib

.. code-block:: bash

   sudo apt-get install libfreetype6-dev
   sudo apt-get install libpng12-dev
   sudo apt-get install python-tk

* Used by mpi4py

.. code-block:: bash

   sudo apt-get install libopenmpi-dev

Then install the dependency requirement for SORTS++

.. code-block:: bash

   pip install -r pip_req.txt


Then test the installation following the test section below.

