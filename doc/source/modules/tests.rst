unit-tests
===========================

List of tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../../tests/test_list.txt
   :linenos:


Visual-tests
===========================

Visual tests produce some amount of plots that are inspected to make a rough estimate of validation.

Test propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These test files will take a propagator and produce 3 different orbit plots:

1. Circular orbit in equatorial place
2. Elliptic orbit in equatorial plane
3. Elliptic polar orbit in ECEF over several periods

Most scientists that work with orbits daily will have a good grasp of how these scenarios should look visually and can make early detection of errors by inspecting the output of the below code.

.. literalinclude:: ../../../tests/test_propagator_base.py
   :language: python
   :linenos:

Simulation-tests
===========================

test


