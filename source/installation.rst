Installation
======================================================================

Run from the hiNiche project directory. The StarDist file supports Linux, macOS, and Windows; the main hiNiche environment targets Linux with an NVIDIA GPU and CUDA 13.2.

Step 1: StarDist
----------------------------------------------------------------------

For nuclei segmentation.

.. code-block:: bash

   conda env create -f environment_stardist.yml
   conda activate hiNiche_stardist
   python -m ipykernel install --user --name hiNiche_stardist --display-name "StarDist"

Select the **StarDist** kernel for nuclei segmentation notebooks.

:download:`environment_stardist.yml <environments/environment_stardist.yml>`

Step 2: hiNiche
----------------------------------------------------------------------

For reference preparation, training, and downstream analysis.

.. code-block:: bash

   conda deactivate
   conda env create -f environment.yml
   conda activate hiNiche
   python -m ipykernel install --user --name hiNiche --display-name "hiNiche"
   jupyter lab

Select the **hiNiche** kernel for all remaining notebooks.

:download:`environment.yml <environments/environment.yml>`
