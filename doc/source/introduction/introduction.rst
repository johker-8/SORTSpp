Introduction
===================================

What is SORTS++
-----------------
SORTS++ stands for Next-generation (++) Space Object Radar Tracking Simulator (SORTS). It is a collection of modules designed for research purposes concerning the tracking and detection of objects in space. Its ultimate goal is to simulate the tracking and discovery of objects in space using radar systems in a very general fashion. 


Install
-----------------

System requirements
~~~~~~~~~~~~~~~~~~~~~~

* Unix (tested on Ubuntu-16.04 LTS, Ubuntu-server-16.04 LTS)
* Python 2.7

Dependencies
~~~~~~~~~~~~~~~~~~~~~~

.. include:: ../../../pip_req.txt
   :literal:


"I'm feeling lucky" Install
-------------------------------

THIS SHOULD BE A MAKEFILE
All of the series of install instructions can also be performed by running the :code:`build.sh` file from the :code:`SORTSpp/` folder after cloning the repository:

make <dependancy>

.. code-block:: bash

   make all
   make install



Test installation
------------------

Modules
~~~~~~~~~~~~~~~~

Simply navigate to the :code:`SORTSpp` directory and run:

.. code-block:: bash

   pytest

And it will automatically use the :code:`pytest.ini` file to discover and run all tests.


Simulation
~~~~~~~~~~~~

To test the simulation capabilities (usage of all modules simultaniusly):
 * Look in the **SIMULATIONS/** folder
 * Configure the files ending with :code:`*_test.py` to output data to desired paths.

Remember, To run the simulation with MPI the file must be executable. To test a capability run the corresponding :code:`test_*` file with 

.. code-block:: bash

   python ./SIMULATIONS/test_simulation.py

or with 

.. code-block:: bash

   mpirun -np 8 ./SIMULATIONS/test_simulation.py

if you wish to test the MPI implementation of the simulation. The *-np* specifies how many processes should be launched and should not be larger then the number of cores available.


General simulation
-------------------

To perform a general simulation using SORTS++ you need to use the :class:`~simulation.simulation` class.

To construct a simulation class instance you need:

:Radar instance:
   Manually constructed from :class:`~radar_config.radar_system` or a preset instance from :mod:`radar_library`.

:Population instance:
   Manually constructed from :class:`~population.population` or a preset instance from :mod:`population`.

:Simulation root:
   A designated root folder where simulation files will be stored.

These are the bare minimum, it is also recommended to have:

:Radar scan:
   Manually constructed from :class:`~radar_scans.radar_scan` or a preset instance from :mod:`radar_scan_library`. The radar_scan instance should be set in the radar_system instance using the :func:`~radar_config.radar_system.set_scan` method.

:Scheduler instance:
   A scheduler instance is a function declaration located in :mod:`scheduler_library`. Different schedulers need different configurations and auxiliary functions and instances.


Below is an example simulation of catalogue maintenance:

.. literalinclude:: ../../../SIMULATION_EXAMPLES/simple_sim.py
   :language: python
   :linenos:

The simulation can be found in :code:`SIMULATION_EXAMPLES/simple_sim.py`, remember to change the simulation root before running, it can be run using

.. code-block:: bash

   mpirun -np 4 ./SIMULATION_EXAMPLES/simple_sim.py


License
------------------

.. include:: ../../../LICENSE

