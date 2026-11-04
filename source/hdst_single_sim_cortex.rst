Simulated cortex
================================================================================

:download:`Download original notebook <examples/HDST_sim_cortex_run_hiNiche.ipynb>`

Code and saved outputs are reproduced from this notebook. No cells are executed during documentation builds.

.. rubric:: hiNiche: Simulated mouse cortex (simulation benchmark)

.. raw:: html

   <p>Multi-scale (spot / cell / niche) cell-type deconvolution and niche discovery on spatial transcriptomics data simulated from a mouse cortex reference.</p>

.. raw:: html

   <p>The simulated AnnData provides the ground truth used for evaluation:</p>

.. raw:: html

   <p>- <code>obsm[&quot;S_s2c&quot;]</code>: spot-to-cell assignment</p>

.. raw:: html

   <p>- <code>obs[&quot;celltype&quot;]</code>: ground-truth cell type of each spot</p>

.. raw:: html

   <p>- <code>obs[&quot;niche_id&quot;]</code>: ground-truth niche of each spot</p>

1. Setup and imports
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   import os
   import sys
   # cupy (used by rapids_singlecell) compiles GPU kernels at runtime and needs the CUDA headers shipped inside the conda environment
   os.environ.setdefault("CUDA_PATH", os.path.join(sys.prefix, "targets", "x86_64-linux"))
   
   # --- Project modules ---
   import config
   import utils
   from train import run_train_seq
   
   # --- Standard library ---
   import warnings
   
   # --- Data handling ---
   import numpy as np
   import pandas as pd
   import anndata as ad
   import scanpy as sc
   import scipy.sparse as sp
   
   # --- Deep learning ---
   import torch
   import torch.nn.functional as F
   
   # --- Clustering and evaluation ---
   import rapids_singlecell as rsc     # GPU-accelerated neighbors / UMAP
   from sklearn.cluster import KMeans
   from sklearn.metrics import accuracy_score, f1_score, classification_report
   from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
   
   # --- Plotting ---
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches
   
   %load_ext autoreload
   %autoreload 2
   
   warnings.filterwarnings("ignore")

.. container:: cell-output

   .. code-block:: text

      The autoreload extension is already loaded. To reload it, use:
        %reload_ext autoreload

2. Parameter configuration
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   SAMPLE = "Simulation2_HD"
   
   # Data paths
   h5_path     = f"DATA/{SAMPLE}/sim_cortex.h5ad"
   refile_path = f"DATA/{SAMPLE}/lambda_matrix.csv"
   
   # Patch save path
   PATCH_LOC = f"DATA/{SAMPLE}/mini_batch_cache"
   
   # Model save path (an existing checkpoint is loaded instead of retraining)
   MODEL_SAVE_DIR = f"DATA/{SAMPLE}/checkpoint.pt"
   
   # Color maps defined in config.py
   celltype_color = getattr(config, "colormap_mouse_brain_41")
   umap_color     = getattr(config, "umap_40")
   
   # Model / training hyperparameters
   graph_knn   = 12
   niche_knn   = 20
   lr          = 0.005
   num_epochs  = 500
   edge_drop   = 0.5
   member_drop = 0.5
   neg_samples = 5
   tau         = 1
   alpha       = 1
   beta        = 1
   seed        = 42
   device      = "cuda:0"
   
   # Niche clustering
   n_clusters = 7    # number of KMeans clusters (= number of simulated niches)
   emb_k      = 50   # n_neighbors of the kNN graph used for UMAP only

3. Data loading and preprocessing
--------------------------------------------------------------------------------

3.1 Load simulated data and reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Reference expression profiles (genes x cell types) and simulated spot-level counts
   ref_df = pd.read_csv(refile_path, index_col=0)
   adata = sc.read(h5_path)
   print(adata)

.. container:: cell-output

   .. code-block:: text

      AnnData object with n_obs × n_vars = 360000 × 5170
          obs: 'spotid', 'celltype', 'x', 'y', 'cell_id', 'niche_id', 'niche_name'
          uns: 'cells', 'celltype_names', 'niche_compositions', 'niche_names', 'pattern', 'spatial_neighbors'
          obsm: 'S_s2c', 'spatial_grid'
          layers: 'counts_clean'
          obsp: 'spatial_connectivities', 'spatial_distances'

3.2 Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Keep raw counts before normalization
   adata.layers["counts"] = adata.X.copy()
   adata.var_names_make_unique()
   
   # Library-size normalization + log transform
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)

3.3 Align genes with the reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Restrict to genes shared between the data and the reference
   overlap_gs = adata.var.index.intersection(ref_df.index)
   print(f"[Log] num of gene overlap {len(overlap_gs)}")
   
   adata.obsm["X_mle"] = adata.layers["counts"][:, adata.var_names.isin(overlap_gs)]   # raw counts
   adata.obsm["X_input"] = adata[:, adata.var_names.isin(overlap_gs)].X                # normalized model input
   
   print(f"[Log] input gene number is {len(overlap_gs)}")
   ref_matrix = ref_df.loc[overlap_gs].values
   ref_matrix = np.array(ref_matrix)
   
   celltypes = list(ref_df.columns)
   print(f"[Log] rna ref size is {ref_matrix.shape}")

.. container:: cell-output

   .. code-block:: text

      [Log] num of gene overlap 5170
      [Log] input gene number is 5170
      [Log] rna ref size is (5170, 41)

4. Build the spot / cell / niche hierarchy
--------------------------------------------------------------------------------

4.1 Spot-to-cell and spot-to-niche assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Spot -> cell assignment taken from the simulation (ground-truth cell segmentation)
   S_s2c_all, cell_background_mask = utils.spot2cell_from_assignment(adata.obsm["S_s2c"], max_spots_per_cell=200, min_spots_per_cell=1)
   print(f"[Log] {S_s2c_all.shape[1]} cells are detected")
   
   # Spot -> niche assignment: each niche is a cell plus its nearest neighbors (including itself)
   S_s2n_all, niche_background_mask, active_niche_indices, S_c2n_all = utils.spot2niche_by_cell_knn(adata.obsm["spatial_grid"], S_s2c_all, k_niche=niche_knn)
   background_mask = {"cell": cell_background_mask, "niche": niche_background_mask}

.. container:: cell-output

   .. code-block:: text

      [Log] Active Cells (before filtering): 8034
      [Log] Valid Cells (1 <= spots <= 200): 8034
      [Log] Filtered Cells (< 1 spots): 0
      [Log] Filtered Cells (> 200 spots): 0
      [Log] 8034 cells are detected
      [Log] Active Niches: 8034, Cells per Niche (k): 20

4.2 Spatial KNN graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Spatial KNN graphs at spot and cell level
   spot_edges = utils.build_spot_knn_graph(adata.obsm["spatial_grid"], graph_knn)
   cell_edges = utils.build_cell_knn_graph(adata.obsm["spatial_grid"], S_s2c_all, graph_knn)
   # Niche graph: reuses the cell KNN graph (one niche per cell), same k as spot/cell graphs
   niche_edges = cell_edges

.. container:: cell-output

   .. code-block:: text

      [Log] 4330462 edges generated from 360000 nodes (k=12)
      [Log] 107196 edges generated from 8034 nodes (k=12)

4.3 Spatial blocking and subgraph caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # Split the slide into overlapping spatial blocks (core + halo) for mini-batch training
   batch_mask, block_ids = utils.make_block_masks(adata.obsm["spatial_grid"], S_s2c=S_s2c_all, S_s2n=S_s2n_all, single_patch = True)
   
   # Build and cache per-block subgraph data to disk
   utils.build_subgraph_data(adata, ref_matrix, spot_edges, cell_edges, niche_edges,
                             S_s2c_all, S_s2n_all, S_c2n_all, background_mask, batch_mask, block_ids, PATCH_LOC)
   print(f"[Log] Original ST is patchfied to {len(block_ids)} blocks with avg {batch_mask['spot_level_core'].sum(dim=1).float().mean().item():.0f} spots, {batch_mask['cell_level_core'].sum(dim=1).float().mean().item():.0f} cells and {batch_mask['niche_level_core'].sum(dim=1).float().mean().item():.0f} niches per block")

.. container:: cell-output

   .. code-block:: text

      [Log] Single-patch mode: the whole slide is one block

.. container:: cell-output

   .. code-block:: text

      [Log] Building & saving mini-batches: 100%|██████████| 1/1 [00:13<00:00, 13.95s/it]

.. container:: cell-output

   .. code-block:: text

      [Log] Saved 1 blocks → DATA/Simulation2_HD/mini_batch_cache
      [Log] Original ST is patchfied to 1 blocks with avg 360000 spots, 8034 cells and 8034 niches per block

5. Model training
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   emb_all, slice_id = run_train_seq(total_unit_number_per_level = {"spot": S_s2c_all.shape[0], "cell": S_s2c_all.shape[1], "niche": S_s2n_all.shape[1]},
                                     PATCH_LOC = PATCH_LOC,
                                     MODEL_SAVE_DIR = MODEL_SAVE_DIR,
                                     ref_matrix = ref_matrix,
                                     lr = lr,
                                     in_dim = len(overlap_gs),
                                     hidden_dim = len(celltypes),
                                     num_epochs = num_epochs,
                                     edge_drop = edge_drop,
                                     member_drop = member_drop,
                                     neg_samples = neg_samples,
                                     tau = tau,
                                     alpha = alpha,
                                     beta = beta,
                                     seed = seed,
                                     device = device)

.. container:: cell-output

   .. code-block:: text

      Epoch 1: 100%|██████████| 1/1 [00:02<00:00,  2.06s/it, cl=5.2715, nll=616.9104, entp=-7.2189]
      Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  1.21it/s, cl=5.5756, nll=605.0949, entp=-7.2093]
      Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  1.23it/s, cl=5.6859, nll=602.8158, entp=-7.1123]
      Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.7109, nll=601.6625, entp=-7.0274]
      Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.7146, nll=600.6374, entp=-6.9499]
      Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.7114, nll=599.6086, entp=-6.8781]
      Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.7066, nll=598.5803, entp=-6.8077]
      Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.7000, nll=597.5720, entp=-6.7340]
      Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.6927, nll=596.5898, entp=-6.6519]
      Epoch 10: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.6846, nll=595.6810, entp=-6.5616]
      Epoch 11: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.6755, nll=594.8501, entp=-6.4677]
      Epoch 12: 100%|██████████| 1/1 [00:00<00:00,  1.23it/s, cl=5.6648, nll=594.0970, entp=-6.3765]
      Epoch 13: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.6541, nll=593.3975, entp=-6.2902]
      Epoch 14: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.6420, nll=592.7529, entp=-6.2112]
      Epoch 15: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.6283, nll=592.1348, entp=-6.1388]
      Epoch 16: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.6142, nll=591.5625, entp=-6.0738]
      Epoch 17: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.6001, nll=591.0020, entp=-6.0116]
      Epoch 18: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.5866, nll=590.4802, entp=-5.9514]
      Epoch 19: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.5708, nll=589.9893, entp=-5.8915]
      Epoch 20: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.5569, nll=589.5203, entp=-5.8299]
      Epoch 21: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.5426, nll=589.0916, entp=-5.7705]
      Epoch 22: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.5276, nll=588.6820, entp=-5.7120]
      Epoch 23: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.5132, nll=588.2814, entp=-5.6558]
      Epoch 24: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.4980, nll=587.8965, entp=-5.6011]
      Epoch 25: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.4844, nll=587.5311, entp=-5.5467]
      Epoch 26: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.4729, nll=587.1909, entp=-5.4970]
      Epoch 27: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.4579, nll=586.8766, entp=-5.4512]
      Epoch 28: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.4453, nll=586.5653, entp=-5.4086]
      Epoch 29: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.4331, nll=586.2718, entp=-5.3720]
      Epoch 30: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.4198, nll=585.9739, entp=-5.3419]
      Epoch 31: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.4097, nll=585.6921, entp=-5.3187]
      Epoch 32: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3999, nll=585.4170, entp=-5.3057]
      Epoch 33: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3873, nll=585.1392, entp=-5.3043]
      Epoch 34: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.3807, nll=584.8701, entp=-5.3081]
      Epoch 35: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3719, nll=584.6378, entp=-5.3090]
      Epoch 36: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3615, nll=584.4299, entp=-5.2991]
      Epoch 37: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.3488, nll=584.1790, entp=-5.2814]
      Epoch 38: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3383, nll=583.9152, entp=-5.2569]
      Epoch 39: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3251, nll=583.6711, entp=-5.2269]
      Epoch 40: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3138, nll=583.4426, entp=-5.1948]
      Epoch 41: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.3025, nll=583.2379, entp=-5.1645]
      Epoch 42: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2901, nll=583.0507, entp=-5.1413]
      Epoch 43: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2792, nll=582.8600, entp=-5.1217]
      Epoch 44: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2709, nll=582.6976, entp=-5.1089]
      Epoch 45: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2607, nll=582.5212, entp=-5.1003]
      Epoch 46: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2506, nll=582.3741, entp=-5.0894]
      Epoch 47: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.2430, nll=582.2382, entp=-5.0827]
      Epoch 48: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2362, nll=582.0936, entp=-5.0707]
      Epoch 49: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2282, nll=581.9777, entp=-5.0613]
      Epoch 50: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2199, nll=581.8556, entp=-5.0490]
      Epoch 51: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.2144, nll=581.7336, entp=-5.0362]
      Epoch 52: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2066, nll=581.6323, entp=-5.0211]
      Epoch 53: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.2005, nll=581.5191, entp=-5.0086]
      Epoch 54: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1930, nll=581.4042, entp=-4.9926]
      Epoch 55: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.1875, nll=581.3157, entp=-4.9758]
      Epoch 56: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1814, nll=581.2241, entp=-4.9636]
      Epoch 57: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1759, nll=581.1235, entp=-4.9475]
      Epoch 58: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1700, nll=581.0454, entp=-4.9382]
      Epoch 59: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1640, nll=580.9743, entp=-4.9282]
      Epoch 60: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1590, nll=580.8884, entp=-4.9172]
      Epoch 61: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1530, nll=580.8185, entp=-4.9085]
      Epoch 62: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1480, nll=580.7419, entp=-4.9015]
      Epoch 63: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.1420, nll=580.6841, entp=-4.8960]
      Epoch 64: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1368, nll=580.6284, entp=-4.8932]
      Epoch 65: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1331, nll=580.5652, entp=-4.8892]
      Epoch 66: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1249, nll=580.4961, entp=-4.8870]
      Epoch 67: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.1222, nll=580.4503, entp=-4.8883]
      Epoch 68: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1155, nll=580.3904, entp=-4.8917]
      Epoch 69: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1105, nll=580.3240, entp=-4.8984]
      Epoch 70: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1090, nll=580.2866, entp=-4.9072]
      Epoch 71: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.1034, nll=580.2329, entp=-4.9201]
      Epoch 72: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0980, nll=580.1805, entp=-4.9360]
      Epoch 73: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0941, nll=580.1289, entp=-4.9540]
      Epoch 74: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0906, nll=580.0685, entp=-4.9832]
      Epoch 75: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0868, nll=580.0087, entp=-5.0146]
      Epoch 76: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0813, nll=579.9286, entp=-5.0528]
      Epoch 77: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0793, nll=579.8473, entp=-5.0978]
      Epoch 78: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0761, nll=579.7493, entp=-5.1470]
      Epoch 79: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0733, nll=579.6666, entp=-5.1988]
      Epoch 80: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0739, nll=579.5910, entp=-5.2394]
      Epoch 81: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0725, nll=579.5202, entp=-5.2717]
      Epoch 82: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0707, nll=579.4675, entp=-5.2997]
      Epoch 83: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0707, nll=579.3788, entp=-5.3198]
      Epoch 84: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0665, nll=579.2612, entp=-5.3395]
      Epoch 85: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0638, nll=579.1145, entp=-5.3542]
      Epoch 86: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0582, nll=578.9756, entp=-5.3716]
      Epoch 87: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0540, nll=578.8566, entp=-5.3841]
      Epoch 88: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0484, nll=578.7574, entp=-5.3949]
      Epoch 89: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0431, nll=578.6361, entp=-5.4023]
      Epoch 90: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0372, nll=578.5062, entp=-5.4088]
      Epoch 91: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0290, nll=578.4288, entp=-5.4139]
      Epoch 92: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0227, nll=578.3365, entp=-5.4226]
      Epoch 93: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=5.0165, nll=578.2386, entp=-5.4314]
      Epoch 94: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0097, nll=578.1254, entp=-5.4383]
      Epoch 95: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=5.0054, nll=578.0208, entp=-5.4515]
      Epoch 96: 100%|██████████| 1/1 [00:00<00:00,  1.26it/s, cl=5.0008, nll=577.9306, entp=-5.4542]
      Epoch 97: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9953, nll=577.8436, entp=-5.4569]
      Epoch 98: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9904, nll=577.7679, entp=-5.4606]
      Epoch 99: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9836, nll=577.6680, entp=-5.4617]
      Epoch 100: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9762, nll=577.6078, entp=-5.4647]
      Epoch 101: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9725, nll=577.5269, entp=-5.4723]
      Epoch 102: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9663, nll=577.4857, entp=-5.4769]
      Epoch 103: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9597, nll=577.4147, entp=-5.4853]
      Epoch 104: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9583, nll=577.3644, entp=-5.4916]
      Epoch 105: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9555, nll=577.3029, entp=-5.4938]
      Epoch 106: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9533, nll=577.2622, entp=-5.5006]
      Epoch 107: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9495, nll=577.2165, entp=-5.5032]
      Epoch 108: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9459, nll=577.1781, entp=-5.5057]
      Epoch 109: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9459, nll=577.1377, entp=-5.5084]
      Epoch 110: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9447, nll=577.1014, entp=-5.5142]
      Epoch 111: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9404, nll=577.0579, entp=-5.5156]
      Epoch 112: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9379, nll=577.0206, entp=-5.5188]
      Epoch 113: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9369, nll=577.0034, entp=-5.5205]
      Epoch 114: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9349, nll=576.9625, entp=-5.5201]
      Epoch 115: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9348, nll=576.9528, entp=-5.5198]
      Epoch 116: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9317, nll=576.9175, entp=-5.5183]
      Epoch 117: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9314, nll=576.8730, entp=-5.5184]
      Epoch 118: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9288, nll=576.8524, entp=-5.5195]
      Epoch 119: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9275, nll=576.8307, entp=-5.5193]
      Epoch 120: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9277, nll=576.8077, entp=-5.5185]
      Epoch 121: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9243, nll=576.7690, entp=-5.5140]
      Epoch 122: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.9210, nll=576.7549, entp=-5.5164]
      Epoch 123: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9218, nll=576.7399, entp=-5.5155]
      Epoch 124: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9165, nll=576.7229, entp=-5.5189]
      Epoch 125: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9161, nll=576.6970, entp=-5.5185]
      Epoch 126: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9127, nll=576.6804, entp=-5.5194]
      Epoch 127: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9126, nll=576.6628, entp=-5.5226]
      Epoch 128: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9095, nll=576.6559, entp=-5.5211]
      Epoch 129: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9076, nll=576.6439, entp=-5.5239]
      Epoch 130: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9030, nll=576.6259, entp=-5.5276]
      Epoch 131: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.9011, nll=576.6005, entp=-5.5307]
      Epoch 132: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8996, nll=576.5927, entp=-5.5296]
      Epoch 133: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8988, nll=576.5790, entp=-5.5344]
      Epoch 134: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8966, nll=576.5786, entp=-5.5354]
      Epoch 135: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8950, nll=576.5566, entp=-5.5380]
      Epoch 136: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8927, nll=576.5413, entp=-5.5409]
      Epoch 137: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8929, nll=576.5115, entp=-5.5450]
      Epoch 138: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8923, nll=576.5140, entp=-5.5484]
      Epoch 139: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8904, nll=576.4974, entp=-5.5526]
      Epoch 140: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8882, nll=576.4846, entp=-5.5538]
      Epoch 141: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8883, nll=576.4855, entp=-5.5566]
      Epoch 142: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8858, nll=576.4727, entp=-5.5603]
      Epoch 143: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8847, nll=576.4600, entp=-5.5654]
      Epoch 144: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8814, nll=576.4413, entp=-5.5670]
      Epoch 145: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8815, nll=576.4299, entp=-5.5686]
      Epoch 146: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8810, nll=576.4130, entp=-5.5722]
      Epoch 147: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8774, nll=576.4008, entp=-5.5727]
      Epoch 148: 100%|██████████| 1/1 [00:00<00:00,  1.26it/s, cl=4.8771, nll=576.3887, entp=-5.5748]
      Epoch 149: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8758, nll=576.3704, entp=-5.5786]
      Epoch 150: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8730, nll=576.3470, entp=-5.5806]
      Epoch 151: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8721, nll=576.3298, entp=-5.5852]
      Epoch 152: 100%|██████████| 1/1 [00:00<00:00,  1.26it/s, cl=4.8699, nll=576.3397, entp=-5.5897]
      Epoch 153: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8690, nll=576.3013, entp=-5.5930]
      Epoch 154: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.8690, nll=576.3125, entp=-5.5982]
      Epoch 155: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8658, nll=576.2648, entp=-5.6051]
      Epoch 156: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8668, nll=576.2430, entp=-5.6085]
      Epoch 157: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8647, nll=576.2112, entp=-5.6152]
      Epoch 158: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8644, nll=576.1742, entp=-5.6250]
      Epoch 159: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8628, nll=576.1370, entp=-5.6348]
      Epoch 160: 100%|██████████| 1/1 [00:00<00:00,  1.26it/s, cl=4.8650, nll=576.0815, entp=-5.6445]
      Epoch 161: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8656, nll=576.0308, entp=-5.6549]
      Epoch 162: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8645, nll=575.9732, entp=-5.6598]
      Epoch 163: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8623, nll=575.9314, entp=-5.6650]
      Epoch 164: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8604, nll=575.8845, entp=-5.6697]
      Epoch 165: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8602, nll=575.8788, entp=-5.6726]
      Epoch 166: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8598, nll=575.8351, entp=-5.6753]
      Epoch 167: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8552, nll=575.7995, entp=-5.6776]
      Epoch 168: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8524, nll=575.7763, entp=-5.6788]
      Epoch 169: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8496, nll=575.7338, entp=-5.6789]
      Epoch 170: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8476, nll=575.7103, entp=-5.6808]
      Epoch 171: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8430, nll=575.6854, entp=-5.6788]
      Epoch 172: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8394, nll=575.6307, entp=-5.6766]
      Epoch 173: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8357, nll=575.6144, entp=-5.6773]
      Epoch 174: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8326, nll=575.5598, entp=-5.6766]
      Epoch 175: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8295, nll=575.5419, entp=-5.6788]
      Epoch 176: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8254, nll=575.5184, entp=-5.6797]
      Epoch 177: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8249, nll=575.4898, entp=-5.6821]
      Epoch 178: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8249, nll=575.4694, entp=-5.6827]
      Epoch 179: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8243, nll=575.4394, entp=-5.6874]
      Epoch 180: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8233, nll=575.4348, entp=-5.6933]
      Epoch 181: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8218, nll=575.3938, entp=-5.7004]
      Epoch 182: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8200, nll=575.3912, entp=-5.7099]
      Epoch 183: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8220, nll=575.3657, entp=-5.7213]
      Epoch 184: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8208, nll=575.3520, entp=-5.7356]
      Epoch 185: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8208, nll=575.3240, entp=-5.7497]
      Epoch 186: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8208, nll=575.2903, entp=-5.7655]
      Epoch 187: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8188, nll=575.2681, entp=-5.7818]
      Epoch 188: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8204, nll=575.2090, entp=-5.8014]
      Epoch 189: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8209, nll=575.1628, entp=-5.8127]
      Epoch 190: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8244, nll=575.1195, entp=-5.8192]
      Epoch 191: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8216, nll=575.0803, entp=-5.8222]
      Epoch 192: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8211, nll=575.0386, entp=-5.8241]
      Epoch 193: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8199, nll=574.9727, entp=-5.8282]
      Epoch 194: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8161, nll=574.9020, entp=-5.8365]
      Epoch 195: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8144, nll=574.7579, entp=-5.8458]
      Epoch 196: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8088, nll=574.6073, entp=-5.8577]
      Epoch 197: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.8077, nll=574.3307, entp=-5.8745]
      Epoch 198: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8070, nll=574.0322, entp=-5.8918]
      Epoch 199: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8023, nll=573.8068, entp=-5.9088]
      Epoch 200: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8019, nll=573.6957, entp=-5.9198]
      Epoch 201: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8025, nll=573.6388, entp=-5.9273]
      Epoch 202: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8045, nll=573.6216, entp=-5.9326]
      Epoch 203: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8039, nll=573.6307, entp=-5.9377]
      Epoch 204: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8029, nll=573.5798, entp=-5.9399]
      Epoch 205: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8025, nll=573.5518, entp=-5.9442]
      Epoch 206: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8018, nll=573.5054, entp=-5.9451]
      Epoch 207: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7994, nll=573.4711, entp=-5.9480]
      Epoch 208: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7989, nll=573.4156, entp=-5.9442]
      Epoch 209: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7981, nll=573.3719, entp=-5.9433]
      Epoch 210: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7974, nll=573.3058, entp=-5.9390]
      Epoch 211: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7973, nll=573.2867, entp=-5.9346]
      Epoch 212: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7935, nll=573.2438, entp=-5.9276]
      Epoch 213: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7961, nll=573.1812, entp=-5.9229]
      Epoch 214: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7961, nll=573.1578, entp=-5.9163]
      Epoch 215: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7985, nll=573.1241, entp=-5.9115]
      Epoch 216: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7956, nll=573.1091, entp=-5.9058]
      Epoch 217: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7965, nll=573.0593, entp=-5.9016]
      Epoch 218: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7993, nll=573.0344, entp=-5.8987]
      Epoch 219: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7992, nll=573.0035, entp=-5.8919]
      Epoch 220: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8014, nll=572.9970, entp=-5.8903]
      Epoch 221: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8021, nll=572.9818, entp=-5.8907]
      Epoch 222: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8028, nll=572.9518, entp=-5.8879]
      Epoch 223: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8044, nll=572.9023, entp=-5.8888]
      Epoch 224: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.8063, nll=572.9010, entp=-5.8923]
      Epoch 225: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8083, nll=572.8791, entp=-5.8964]
      Epoch 226: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8060, nll=572.8599, entp=-5.8978]
      Epoch 227: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8072, nll=572.8226, entp=-5.9039]
      Epoch 228: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8069, nll=572.7808, entp=-5.9102]
      Epoch 229: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8068, nll=572.7682, entp=-5.9160]
      Epoch 230: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8090, nll=572.7186, entp=-5.9190]
      Epoch 231: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8079, nll=572.6923, entp=-5.9250]
      Epoch 232: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8096, nll=572.6472, entp=-5.9269]
      Epoch 233: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8104, nll=572.6022, entp=-5.9283]
      Epoch 234: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8096, nll=572.5629, entp=-5.9303]
      Epoch 235: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8079, nll=572.4916, entp=-5.9327]
      Epoch 236: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8100, nll=572.4149, entp=-5.9321]
      Epoch 237: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8084, nll=572.3618, entp=-5.9318]
      Epoch 238: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8073, nll=572.2864, entp=-5.9323]
      Epoch 239: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8091, nll=572.2483, entp=-5.9337]
      Epoch 240: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8057, nll=572.1778, entp=-5.9333]
      Epoch 241: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8055, nll=572.1400, entp=-5.9317]
      Epoch 242: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8041, nll=572.1147, entp=-5.9342]
      Epoch 243: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8033, nll=572.0852, entp=-5.9344]
      Epoch 244: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8024, nll=572.0674, entp=-5.9373]
      Epoch 245: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8032, nll=572.0350, entp=-5.9387]
      Epoch 246: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.8004, nll=572.0294, entp=-5.9421]
      Epoch 247: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7982, nll=572.0067, entp=-5.9440]
      Epoch 248: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7992, nll=572.0088, entp=-5.9440]
      Epoch 249: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7959, nll=571.9852, entp=-5.9456]
      Epoch 250: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7952, nll=571.9652, entp=-5.9475]
      Epoch 251: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7954, nll=571.9468, entp=-5.9473]
      Epoch 252: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7941, nll=571.9651, entp=-5.9472]
      Epoch 253: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7916, nll=571.9231, entp=-5.9495]
      Epoch 254: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7925, nll=571.9120, entp=-5.9473]
      Epoch 255: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7919, nll=571.8928, entp=-5.9489]
      Epoch 256: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7933, nll=571.8987, entp=-5.9473]
      Epoch 257: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7910, nll=571.8714, entp=-5.9475]
      Epoch 258: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7909, nll=571.8500, entp=-5.9456]
      Epoch 259: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7897, nll=571.8540, entp=-5.9461]
      Epoch 260: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7889, nll=571.8377, entp=-5.9445]
      Epoch 261: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7870, nll=571.8248, entp=-5.9434]
      Epoch 262: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7878, nll=571.8190, entp=-5.9448]
      Epoch 263: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7887, nll=571.8170, entp=-5.9409]
      Epoch 264: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7878, nll=571.7986, entp=-5.9436]
      Epoch 265: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7856, nll=571.7853, entp=-5.9423]
      Epoch 266: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7845, nll=571.7887, entp=-5.9453]
      Epoch 267: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7838, nll=571.7639, entp=-5.9446]
      Epoch 268: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7844, nll=571.7443, entp=-5.9450]
      Epoch 269: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7847, nll=571.7654, entp=-5.9454]
      Epoch 270: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7814, nll=571.7425, entp=-5.9430]
      Epoch 271: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7813, nll=571.7368, entp=-5.9449]
      Epoch 272: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7810, nll=571.7228, entp=-5.9419]
      Epoch 273: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7790, nll=571.7303, entp=-5.9422]
      Epoch 274: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7818, nll=571.7117, entp=-5.9418]
      Epoch 275: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7824, nll=571.7037, entp=-5.9403]
      Epoch 276: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7805, nll=571.7079, entp=-5.9394]
      Epoch 277: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7792, nll=571.6884, entp=-5.9416]
      Epoch 278: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7822, nll=571.6602, entp=-5.9392]
      Epoch 279: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7816, nll=571.6761, entp=-5.9408]
      Epoch 280: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7817, nll=571.6723, entp=-5.9423]
      Epoch 281: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7792, nll=571.6622, entp=-5.9389]
      Epoch 282: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7789, nll=571.6711, entp=-5.9397]
      Epoch 283: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7803, nll=571.6606, entp=-5.9412]
      Epoch 284: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7798, nll=571.6431, entp=-5.9420]
      Epoch 285: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7817, nll=571.6442, entp=-5.9415]
      Epoch 286: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7789, nll=571.6207, entp=-5.9404]
      Epoch 287: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7774, nll=571.6209, entp=-5.9406]
      Epoch 288: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7771, nll=571.6234, entp=-5.9404]
      Epoch 289: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7785, nll=571.6136, entp=-5.9393]
      Epoch 290: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7789, nll=571.6143, entp=-5.9411]
      Epoch 291: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7759, nll=571.6115, entp=-5.9409]
      Epoch 292: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7766, nll=571.6109, entp=-5.9419]
      Epoch 293: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7746, nll=571.6097, entp=-5.9412]
      Epoch 294: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7763, nll=571.5940, entp=-5.9421]
      Epoch 295: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7739, nll=571.5911, entp=-5.9414]
      Epoch 296: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7771, nll=571.5774, entp=-5.9426]
      Epoch 297: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7765, nll=571.5624, entp=-5.9416]
      Epoch 298: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7753, nll=571.5924, entp=-5.9430]
      Epoch 299: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7755, nll=571.5677, entp=-5.9437]
      Epoch 300: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7748, nll=571.5535, entp=-5.9427]
      Epoch 301: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7740, nll=571.5664, entp=-5.9409]
      Epoch 302: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7734, nll=571.5489, entp=-5.9428]
      Epoch 303: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7730, nll=571.5524, entp=-5.9418]
      Epoch 304: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7748, nll=571.5406, entp=-5.9401]
      Epoch 305: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7736, nll=571.5490, entp=-5.9426]
      Epoch 306: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7738, nll=571.5416, entp=-5.9420]
      Epoch 307: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7721, nll=571.5189, entp=-5.9436]
      Epoch 308: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7721, nll=571.5182, entp=-5.9444]
      Epoch 309: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7718, nll=571.5088, entp=-5.9410]
      Epoch 310: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7719, nll=571.5273, entp=-5.9439]
      Epoch 311: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7710, nll=571.5093, entp=-5.9428]
      Epoch 312: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7709, nll=571.5209, entp=-5.9423]
      Epoch 313: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7710, nll=571.5115, entp=-5.9409]
      Epoch 314: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7708, nll=571.4954, entp=-5.9434]
      Epoch 315: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7704, nll=571.5022, entp=-5.9445]
      Epoch 316: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7704, nll=571.4911, entp=-5.9437]
      Epoch 317: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7685, nll=571.4981, entp=-5.9450]
      Epoch 318: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7692, nll=571.4952, entp=-5.9449]
      Epoch 319: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7673, nll=571.4940, entp=-5.9446]
      Epoch 320: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7668, nll=571.4990, entp=-5.9432]
      Epoch 321: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7690, nll=571.4824, entp=-5.9439]
      Epoch 322: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7678, nll=571.4691, entp=-5.9433]
      Epoch 323: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7682, nll=571.4744, entp=-5.9426]
      Epoch 324: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7651, nll=571.4641, entp=-5.9437]
      Epoch 325: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7680, nll=571.4630, entp=-5.9427]
      Epoch 326: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7674, nll=571.4556, entp=-5.9445]
      Epoch 327: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7674, nll=571.4599, entp=-5.9425]
      Epoch 328: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7668, nll=571.4463, entp=-5.9430]
      Epoch 329: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7648, nll=571.4620, entp=-5.9434]
      Epoch 330: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7662, nll=571.4523, entp=-5.9433]
      Epoch 331: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7639, nll=571.4631, entp=-5.9453]
      Epoch 332: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7638, nll=571.4383, entp=-5.9460]
      Epoch 333: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7643, nll=571.4465, entp=-5.9457]
      Epoch 334: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7630, nll=571.4207, entp=-5.9441]
      Epoch 335: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7639, nll=571.4420, entp=-5.9461]
      Epoch 336: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7640, nll=571.4310, entp=-5.9460]
      Epoch 337: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7635, nll=571.4150, entp=-5.9439]
      Epoch 338: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7623, nll=571.4271, entp=-5.9465]
      Epoch 339: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7618, nll=571.4231, entp=-5.9459]
      Epoch 340: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7623, nll=571.4218, entp=-5.9460]
      Epoch 341: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7610, nll=571.4136, entp=-5.9467]
      Epoch 342: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7615, nll=571.4215, entp=-5.9469]
      Epoch 343: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7591, nll=571.4157, entp=-5.9477]
      Epoch 344: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7604, nll=571.4121, entp=-5.9481]
      Epoch 345: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7595, nll=571.4111, entp=-5.9459]
      Epoch 346: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7603, nll=571.3956, entp=-5.9439]
      Epoch 347: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7613, nll=571.3967, entp=-5.9461]
      Epoch 348: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7597, nll=571.3809, entp=-5.9439]
      Epoch 349: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7603, nll=571.3990, entp=-5.9460]
      Epoch 350: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7597, nll=571.3876, entp=-5.9450]
      Epoch 351: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7594, nll=571.3879, entp=-5.9470]
      Epoch 352: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7582, nll=571.3713, entp=-5.9462]
      Epoch 353: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7600, nll=571.3771, entp=-5.9476]
      Epoch 354: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7570, nll=571.3660, entp=-5.9457]
      Epoch 355: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7576, nll=571.3701, entp=-5.9460]
      Epoch 356: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7594, nll=571.3784, entp=-5.9455]
      Epoch 357: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7572, nll=571.3698, entp=-5.9463]
      Epoch 358: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7556, nll=571.3698, entp=-5.9457]
      Epoch 359: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7574, nll=571.3682, entp=-5.9463]
      Epoch 360: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7570, nll=571.3563, entp=-5.9461]
      Epoch 361: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7553, nll=571.3663, entp=-5.9487]
      Epoch 362: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7564, nll=571.3490, entp=-5.9477]
      Epoch 363: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7561, nll=571.3551, entp=-5.9479]
      Epoch 364: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7554, nll=571.3444, entp=-5.9458]
      Epoch 365: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7546, nll=571.3559, entp=-5.9466]
      Epoch 366: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7540, nll=571.3400, entp=-5.9468]
      Epoch 367: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7538, nll=571.3375, entp=-5.9472]
      Epoch 368: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7533, nll=571.3303, entp=-5.9473]
      Epoch 369: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7546, nll=571.3217, entp=-5.9469]
      Epoch 370: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7551, nll=571.3264, entp=-5.9455]
      Epoch 371: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7531, nll=571.3311, entp=-5.9479]
      Epoch 372: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7529, nll=571.3147, entp=-5.9461]
      Epoch 373: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7529, nll=571.3150, entp=-5.9469]
      Epoch 374: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7509, nll=571.3121, entp=-5.9466]
      Epoch 375: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7523, nll=571.3113, entp=-5.9455]
      Epoch 376: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7530, nll=571.3165, entp=-5.9459]
      Epoch 377: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7508, nll=571.3094, entp=-5.9474]
      Epoch 378: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7492, nll=571.3154, entp=-5.9476]
      Epoch 379: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7501, nll=571.3007, entp=-5.9478]
      Epoch 380: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7486, nll=571.3118, entp=-5.9482]
      Epoch 381: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7484, nll=571.3094, entp=-5.9500]
      Epoch 382: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7506, nll=571.2860, entp=-5.9496]
      Epoch 383: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7482, nll=571.2911, entp=-5.9497]
      Epoch 384: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7488, nll=571.3031, entp=-5.9475]
      Epoch 385: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7484, nll=571.2959, entp=-5.9486]
      Epoch 386: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7493, nll=571.2780, entp=-5.9492]
      Epoch 387: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7490, nll=571.2721, entp=-5.9473]
      Epoch 388: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7477, nll=571.2737, entp=-5.9473]
      Epoch 389: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7480, nll=571.2550, entp=-5.9475]
      Epoch 390: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7471, nll=571.2678, entp=-5.9474]
      Epoch 391: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7483, nll=571.2637, entp=-5.9485]
      Epoch 392: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7484, nll=571.2580, entp=-5.9468]
      Epoch 393: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7469, nll=571.2550, entp=-5.9463]
      Epoch 394: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7462, nll=571.2586, entp=-5.9457]
      Epoch 395: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7459, nll=571.2311, entp=-5.9473]
      Epoch 396: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7439, nll=571.2275, entp=-5.9483]
      Epoch 397: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7464, nll=571.2108, entp=-5.9468]
      Epoch 398: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7448, nll=571.1844, entp=-5.9486]
      Epoch 399: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7446, nll=571.1669, entp=-5.9464]
      Epoch 400: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7436, nll=571.1448, entp=-5.9460]
      Epoch 401: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7447, nll=571.0924, entp=-5.9478]
      Epoch 402: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7446, nll=571.0229, entp=-5.9496]
      Epoch 403: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7442, nll=570.9329, entp=-5.9527]
      Epoch 404: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7452, nll=570.7516, entp=-5.9589]
      Epoch 405: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7464, nll=570.5239, entp=-5.9674]
      Epoch 406: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7465, nll=570.1013, entp=-5.9820]
      Epoch 407: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7490, nll=569.5319, entp=-6.0035]
      Epoch 408: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7510, nll=568.8563, entp=-6.0247]
      Epoch 409: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7545, nll=568.3647, entp=-6.0354]
      Epoch 410: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7625, nll=568.0595, entp=-6.0414]
      Epoch 411: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7656, nll=567.9060, entp=-6.0472]
      Epoch 412: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7697, nll=567.8465, entp=-6.0520]
      Epoch 413: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7731, nll=567.7849, entp=-6.0539]
      Epoch 414: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7778, nll=567.7676, entp=-6.0590]
      Epoch 415: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7786, nll=567.7698, entp=-6.0599]
      Epoch 416: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7835, nll=567.7613, entp=-6.0614]
      Epoch 417: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7840, nll=567.7567, entp=-6.0633]
      Epoch 418: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7846, nll=567.7401, entp=-6.0647]
      Epoch 419: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7847, nll=567.7429, entp=-6.0677]
      Epoch 420: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7857, nll=567.7261, entp=-6.0677]
      Epoch 421: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7855, nll=567.7159, entp=-6.0667]
      Epoch 422: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7864, nll=567.7111, entp=-6.0683]
      Epoch 423: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7869, nll=567.6805, entp=-6.0664]
      Epoch 424: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7841, nll=567.7061, entp=-6.0676]
      Epoch 425: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7843, nll=567.7139, entp=-6.0662]
      Epoch 426: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7847, nll=567.6611, entp=-6.0630]
      Epoch 427: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7825, nll=567.6711, entp=-6.0638]
      Epoch 428: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7829, nll=567.6513, entp=-6.0623]
      Epoch 429: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7808, nll=567.6378, entp=-6.0596]
      Epoch 430: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7774, nll=567.6174, entp=-6.0589]
      Epoch 431: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7778, nll=567.6112, entp=-6.0579]
      Epoch 432: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7742, nll=567.5818, entp=-6.0558]
      Epoch 433: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7739, nll=567.5714, entp=-6.0547]
      Epoch 434: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7737, nll=567.5674, entp=-6.0511]
      Epoch 435: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7706, nll=567.5715, entp=-6.0517]
      Epoch 436: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7707, nll=567.5646, entp=-6.0493]
      Epoch 437: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7682, nll=567.5370, entp=-6.0467]
      Epoch 438: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7677, nll=567.5315, entp=-6.0457]
      Epoch 439: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7667, nll=567.5222, entp=-6.0453]
      Epoch 440: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7645, nll=567.5101, entp=-6.0425]
      Epoch 441: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7630, nll=567.5013, entp=-6.0422]
      Epoch 442: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7621, nll=567.4992, entp=-6.0406]
      Epoch 443: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7622, nll=567.4827, entp=-6.0402]
      Epoch 444: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7592, nll=567.4690, entp=-6.0374]
      Epoch 445: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7601, nll=567.4640, entp=-6.0374]
      Epoch 446: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7580, nll=567.4412, entp=-6.0371]
      Epoch 447: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7566, nll=567.4317, entp=-6.0355]
      Epoch 448: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7580, nll=567.4260, entp=-6.0353]
      Epoch 449: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7548, nll=567.3967, entp=-6.0369]
      Epoch 450: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7562, nll=567.3857, entp=-6.0358]
      Epoch 451: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7557, nll=567.3387, entp=-6.0362]
      Epoch 452: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7540, nll=567.3195, entp=-6.0363]
      Epoch 453: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7546, nll=567.2529, entp=-6.0393]
      Epoch 454: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7536, nll=567.1669, entp=-6.0410]
      Epoch 455: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7537, nll=567.0832, entp=-6.0410]
      Epoch 456: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7546, nll=566.9774, entp=-6.0436]
      Epoch 457: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7535, nll=566.9355, entp=-6.0426]
      Epoch 458: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7542, nll=566.8861, entp=-6.0433]
      Epoch 459: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7552, nll=566.8591, entp=-6.0434]
      Epoch 460: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7538, nll=566.8674, entp=-6.0427]
      Epoch 461: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7541, nll=566.8242, entp=-6.0450]
      Epoch 462: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7538, nll=566.8224, entp=-6.0463]
      Epoch 463: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7538, nll=566.8280, entp=-6.0468]
      Epoch 464: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7537, nll=566.8037, entp=-6.0475]
      Epoch 465: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7516, nll=566.8054, entp=-6.0473]
      Epoch 466: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7507, nll=566.8008, entp=-6.0479]
      Epoch 467: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7492, nll=566.7947, entp=-6.0471]
      Epoch 468: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7516, nll=566.7841, entp=-6.0489]
      Epoch 469: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7501, nll=566.7943, entp=-6.0493]
      Epoch 470: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7476, nll=566.7843, entp=-6.0485]
      Epoch 471: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7494, nll=566.7948, entp=-6.0498]
      Epoch 472: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7474, nll=566.7702, entp=-6.0499]
      Epoch 473: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7466, nll=566.7646, entp=-6.0485]
      Epoch 474: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7488, nll=566.7653, entp=-6.0493]
      Epoch 475: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7435, nll=566.7834, entp=-6.0497]
      Epoch 476: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7455, nll=566.7620, entp=-6.0491]
      Epoch 477: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7441, nll=566.7537, entp=-6.0500]
      Epoch 478: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7429, nll=566.7539, entp=-6.0497]
      Epoch 479: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7418, nll=566.7448, entp=-6.0506]
      Epoch 480: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7440, nll=566.7499, entp=-6.0505]
      Epoch 481: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7432, nll=566.7354, entp=-6.0509]
      Epoch 482: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7407, nll=566.7330, entp=-6.0516]
      Epoch 483: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7392, nll=566.7234, entp=-6.0512]
      Epoch 484: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7371, nll=566.7451, entp=-6.0509]
      Epoch 485: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7385, nll=566.7228, entp=-6.0510]
      Epoch 486: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7393, nll=566.7242, entp=-6.0483]
      Epoch 487: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7382, nll=566.7092, entp=-6.0498]
      Epoch 488: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7369, nll=566.7284, entp=-6.0513]
      Epoch 489: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7344, nll=566.7174, entp=-6.0506]
      Epoch 490: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7340, nll=566.7117, entp=-6.0508]
      Epoch 491: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7344, nll=566.7081, entp=-6.0487]
      Epoch 492: 100%|██████████| 1/1 [00:00<00:00,  1.24it/s, cl=4.7328, nll=566.7065, entp=-6.0496]
      Epoch 493: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7320, nll=566.7138, entp=-6.0508]
      Epoch 494: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7344, nll=566.7120, entp=-6.0509]
      Epoch 495: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7315, nll=566.6941, entp=-6.0501]
      Epoch 496: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7317, nll=566.6995, entp=-6.0497]
      Epoch 497: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7306, nll=566.6767, entp=-6.0497]
      Epoch 498: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7341, nll=566.6902, entp=-6.0471]
      Epoch 499: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7319, nll=566.6925, entp=-6.0484]
      Epoch 500: 100%|██████████| 1/1 [00:00<00:00,  1.25it/s, cl=4.7331, nll=566.6818, entp=-6.0475]

.. container:: cell-output

   .. code-block:: text

      Training finished!
      Model saved to DATA/Simulation2_HD/checkpoint.pt

.. container:: cell-output

   .. code-block:: text

      generating embeddings (slice 0)...: 100%|██████████| 1/1 [00:00<00:00,  1.59it/s]

6. Cell-typing evaluation
--------------------------------------------------------------------------------

6.1 Spot- and cell-level cell-type maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=600)
   
   keys = ["spot", "cell"]
   labels_all = {}
   for i, ax in enumerate(axes):
       check = keys[i]
       x_raw, y_raw = adata.obsm["spatial_grid"][:, 0], adata.obsm["spatial_grid"][:, 1]
   
       # Cell-type probabilities
       prob = F.softmax(emb_all[check], dim=1)
       max_prob_raw, max_idx_raw = torch.max(prob, dim=1)
   
       if check == "cell":
           # Broadcast cell-level predictions back to their spots (spot -> cell mapping)
           S = S_s2c_all.coalesce()
           rows, cols = S.indices()[0], S.indices()[1]
           spot2target = torch.full((S.size(0),), -1, dtype=torch.long, device=S.device)
           spot2target[rows] = cols
           max_idx = torch.where(spot2target >= 0, max_idx_raw[spot2target], torch.tensor(-1, device=S.device))
           max_prob = torch.where(spot2target >= 0, max_prob_raw[spot2target], torch.tensor(0.0, device=S.device))
       else:
           max_prob, max_idx = max_prob_raw, max_idx_raw
   
       # --- Label assignment ---
       cell_labels = []
       for j in range(len(max_prob)):
           # max_idx == -1: the spot belongs to no valid cell
           if max_idx[j] == -1:
               cell_labels.append("filtered")
           elif max_prob[j] < 0.1:
               cell_labels.append("ambiguous")
           else:
               cell_labels.append(celltypes[max_idx[j]])
       labels_all[check] = cell_labels
   
       colors = [celltype_color[name] for name in cell_labels]
       ax.scatter(x_raw, y_raw, c=colors, s=5, marker='.', linewidths=0)
       ax.set_title(f"{check} level", fontsize=8)
       ax.invert_yaxis()
       ax.axis("off")
   
   plt.tight_layout()
   plt.savefig("spot_cell_type.png")
   plt.show()
   
   # Legend of all cell types in the color map
   handles = [mpatches.Patch(color=color, label=label) for label, color in celltype_color.items()]
   ncols = 7
   nrows = (len(handles) + ncols - 1) // ncols
   fig, ax = plt.subplots(figsize=(ncols * 3, nrows * 0.1))
   ax.legend(handles=handles, loc='upper center', ncol=ncols, frameon=False, handlelength=1.5, fontsize=10)
   ax.axis('off')
   plt.tight_layout()
   plt.show()

.. image:: _images/544190e5503eeb17ab389b2b2efcc26f920ef6fbf076137513b061751f55f4a7.png
   :alt: Saved result from Simulated cortex
   :width: 100%

.. image:: _images/993f2651001b41f974baf06e5d4af554e2f8f4262089bb77ceb35f04f64591ed.png
   :alt: Saved result from Simulated cortex
   :width: 100%

6.2 Cell-type accuracy against the ground truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Each cell&#x27;s ground-truth type is the majority <code>celltype</code> of its spots; the prediction is the argmax of the cell-level output.</p>

.. raw:: html

   <p>Cells without ground truth (<code>NA</code>) or labeled <code>ambiguous</code> are excluded.</p>

.. code-block:: python
   :linenos:

   # --- Prediction: argmax cell type of each cell ---
   pred_idx = F.softmax(emb_all["cell"], dim=1).argmax(1).cpu().numpy()
   pred_ct  = np.asarray(celltypes)[pred_idx]
   
   # --- Ground truth: spot cell type -> majority vote per cell ---
   S = S_s2c_all.coalesce()
   s_rows = S.indices()[0].cpu().numpy()
   c_cols = S.indices()[1].cpu().numpy()
   n_cell = S.size(1)
   gt_spot = adata.obs["celltype"].astype(str).values
   maj = pd.DataFrame({"cell": c_cols, "ct": gt_spot[s_rows]}).groupby("cell")["ct"].agg(lambda x: x.mode().iat[0])
   gt_ct = np.array(["NA"] * n_cell, dtype=object)
   gt_ct[maj.index.values] = maj.values
   m = gt_ct != "NA"
   
   # --- Evaluation label set: types present in the ground truth, excluding "ambiguous" ---
   eval_labels = sorted(set(gt_ct[m]) - {"ambiguous"})
   keep = m & np.isin(gt_ct, eval_labels)
   print(f"evaluated cells: {int(keep.sum())}/{n_cell} (excluding NA and ambiguous)")
   
   # Cells predicted as a type that does not exist in the ground truth (reported separately)
   spurious = ~np.isin(pred_ct[keep], eval_labels)
   print(f"cells predicted as a type absent from the ground truth: {spurious.sum()} ({spurious.mean():.2%}), "
         f"types: {sorted(set(pred_ct[keep][spurious]))}")
   
   print("Accuracy     :", round(accuracy_score(gt_ct[keep], pred_ct[keep]), 4))
   print("F1 (macro)   :", round(f1_score(gt_ct[keep], pred_ct[keep], labels=eval_labels,
                                          average="macro", zero_division=0), 4))
   print("F1 (weighted):", round(f1_score(gt_ct[keep], pred_ct[keep], labels=eval_labels,
                                          average="weighted", zero_division=0), 4))
   print()
   print(classification_report(gt_ct[keep], pred_ct[keep], labels=eval_labels, zero_division=0))

.. container:: cell-output

   .. code-block:: text

      evaluated cells: 8034/8034 (excluding NA and ambiguous)
      cells predicted as a type absent from the ground truth: 708 (8.81%), types: [np.str_('CA1-ProS'), np.str_('Car3'), np.str_('DG'), np.str_('L2/3 IT PPP'), np.str_('L5/6 IT TPE-ENT'), np.str_('NP PPP'), np.str_('SUB-ProS')]
      Accuracy     : 0.8559
      F1 (macro)   : 0.7565
      F1 (weighted): 0.8475
      
                    precision    recall  f1-score   support
      
             Astro       0.99      1.00      0.99      1033
                CR       0.00      0.00      0.00       201
              Endo       0.99      1.00      0.99        72
       L2/3 IT CTX       1.00      0.93      0.96       904
       L4/5 IT CTX       0.93      1.00      0.96       526
         L5 IT CTX       0.98      1.00      0.99       514
         L5 PT CTX       0.99      0.98      0.98       401
       L5/6 NP CTX       0.91      0.96      0.93       275
         L6 CT CTX       0.85      0.35      0.50       535
         L6 IT CTX       0.00      0.00      0.00       403
           L6b CTX       0.72      1.00      0.84       332
             Lamp5       0.83      1.00      0.91       298
         Micro-PVM       0.99      1.00      0.99       231
             Oligo       0.98      1.00      0.99       883
             Pvalb       0.95      1.00      0.98       287
          SMC-Peri       0.96      1.00      0.98        74
              Sncg       0.00      0.00      0.00        56
               Sst       0.87      1.00      0.93       511
         Sst Chodl       0.00      0.00      0.00        58
              VLMC       0.99      0.99      0.99       148
               Vip       0.94      1.00      0.97       292
      
         micro avg       0.94      0.86      0.90      8034
         macro avg       0.76      0.77      0.76      8034
      weighted avg       0.86      0.86      0.85      8034

7. Niche detection
--------------------------------------------------------------------------------

7.1 Niche clustering (UMAP + spatial map) and NMI / ARI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   torch.cuda.empty_cache()
   
   # --- 1. Niche embedding: KMeans clustering + UMAP ---
   X_niche = emb_all["niche"].numpy()
   niche_label = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X_niche)
   
   adata_tmp = ad.AnnData(X_niche)
   rsc.pp.neighbors(adata_tmp, use_rep='X', n_neighbors=emb_k)
   rsc.tl.umap(adata_tmp, random_state=seed)
   niche_umap = adata_tmp.obsm["X_umap"]
   
   clusters_id = {"niche": niche_label}
   print(f"niche: {len(np.unique(niche_label))} clusters")
   
   # --- 2. Map niche clusters back to their center cells ---
   S = S_s2c_all.coalesce()
   rows, cols = S.indices()[0], S.indices()[1]
   n_cell = S.size(1)
   
   cell_niche = torch.full((n_cell,), -1, dtype=torch.long, device=S.device)
   active = torch.as_tensor(active_niche_indices, device=S.device).long()
   cell_niche[active] = torch.as_tensor(niche_label, device=S.device).long()
   
   # --- 3. Broadcast to spots (each spot takes the niche of its cell) ---
   spot2cell = torch.full((S.size(0),), -1, dtype=torch.long, device=S.device)
   spot2cell[rows] = cols
   spot_niche = torch.where(spot2cell >= 0,
                            cell_niche[spot2cell],
                            torch.tensor(-1, dtype=torch.long, device=S.device)).cpu().numpy()
   
   xy = np.asarray(adata.obsm["spatial_grid"])
   keep_spot = spot_niche >= 0
   print(f"spots with a niche label: {keep_spot.sum()} / {len(xy)}")
   
   # --- 4. Left: niche UMAP; right: spot-level niche spatial distribution ---
   fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
   
   axes[0].scatter(niche_umap[:, 0], niche_umap[:, 1],
                   c=[umap_color[f"Cluster_{v % 40}"] for v in niche_label], s=2, linewidths=0)
   axes[0].set_title(f"niche UMAP (n={n_clusters})")
   axes[0].set_xticks([]); axes[0].set_yticks([])
   axes[0].set_xlabel("UMAP1"); axes[0].set_ylabel("UMAP2")
   
   axes[1].scatter(xy[keep_spot, 0], xy[keep_spot, 1],
                   c=[umap_color[f"Cluster_{v % 40}"] for v in spot_niche[keep_spot]],
                   s=5, marker='.', linewidths=0)
   axes[1].set_title("niche spatial distribution (spot level)")
   axes[1].invert_yaxis(); axes[1].set_aspect("equal"); axes[1].axis("off")
   
   plt.tight_layout()
   plt.show()
   
   # ==================== NMI / ARI (cell-level evaluation) ====================
   s_rows = S.indices()[0].cpu().numpy()
   c_cols = S.indices()[1].cpu().numpy()
   
   # --- Ground truth: spot niche -> majority vote per cell ---
   gt_spot = pd.factorize(adata.obs["niche_id"].values)[0]          # integer labels
   maj = pd.DataFrame({"cell": c_cols, "gt": gt_spot[s_rows]}).groupby("cell")["gt"].agg(lambda x: np.bincount(x).argmax())
   gt_cell = np.full(n_cell, -1, dtype=np.int64)
   gt_cell[maj.index.values] = maj.values
   
   # --- Prediction: niche cluster of each cell ---
   pred_cell = cell_niche.cpu().numpy()
   
   mask = (pred_cell >= 0) & (gt_cell >= 0)
   print(f"evaluated cells: {mask.sum()} / {n_cell}")
   print(f"GT niches: {len(np.unique(gt_cell[mask]))},  pred niches: {len(np.unique(pred_cell[mask]))}")
   print(f"NMI = {NMI(gt_cell[mask], pred_cell[mask]):.4f}")
   print(f"ARI = {ARI(gt_cell[mask], pred_cell[mask]):.4f}")

.. container:: cell-output

   .. code-block:: text

      niche: 7 clusters
      spots with a niche label: 319899 / 360000

.. image:: _images/92f94e8d28c5f3b9c2ee9bbd1ce4a327b7f3c5e2003ebd28912930703f305031.png
   :alt: Saved result from Simulated cortex
   :width: 100%

.. container:: cell-output

   .. code-block:: text

      evaluated cells: 8034 / 8034
      GT niches: 7,  pred niches: 7
      NMI = 0.9004
      ARI = 0.9141

7.2 Predicted cell-type composition of each niche cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   # --- Predicted cell-type proportions of each niche's center cell (softmax of the cell-level output) ---
   prop = F.softmax(emb_all["cell"], dim=1).cpu().numpy()[np.asarray(active_niche_indices)]   # [n_niche, n_celltype]
   prop_df = pd.DataFrame(prop, columns=celltypes)
   
   # --- Mean composition per niche cluster, stacked bar chart ---
   lab  = np.asarray(clusters_id["niche"])
   comp = prop_df.groupby(lab).mean()                           # [n_cluster, n_celltype]
   
   ax = comp.plot(kind="bar", stacked=True, figsize=(1 + 0.6*len(comp), 5),
                  color=[celltype_color[c] for c in comp.columns], width=0.9)
   for tick, cid in zip(ax.get_xticklabels(), comp.index):      # color x tick labels by niche cluster
       tick.set_color(umap_color[f"Cluster_{cid % 40}"]); tick.set_fontweight("bold")
   ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
             title="cell type", ncol=1 + len(celltypes)//20)
   ax.set_xlabel("niche cluster"); ax.set_ylabel("mean cell type proportion")
   ax.set_title("Niche cell-type composition (deconvolution, softmax)")
   plt.tight_layout(); plt.show()

.. image:: _images/4d2c2b5de0563800bf6b64b82f6c181a6d09c00f22444484949d8d9439ff0378.png
   :alt: Saved result from Simulated cortex
   :width: 100%

