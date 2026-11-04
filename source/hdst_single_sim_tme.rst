Simulated tumor microenvironment
================================================================================

:download:`Download original notebook <examples/HDST_sim_TME_run_hiNiche.ipynb>`

Code and saved outputs are reproduced from this notebook. No cells are executed during documentation builds.

.. rubric:: hiNiche: Simulated colorectal cancer TME (simulation benchmark)

.. raw:: html

   <p>Multi-scale (spot / cell / niche) cell-type deconvolution and niche discovery on spatial transcriptomics data simulated from a colorectal cancer tumor microenvironment (TME) reference.</p>

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

      /hpc/home/wz220/miniconda3/envs/hiNiche/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
        from .autonotebook import tqdm as notebook_tqdm

2. Parameter configuration
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   SAMPLE = "Simulation1_HD"
   
   # Data paths
   h5_path     = f"DATA/{SAMPLE}/sim_TME.h5ad"
   refile_path = f"DATA/{SAMPLE}/lambda_matrix.csv"
   
   # Patch save path
   PATCH_LOC = f"DATA/{SAMPLE}/mini_batch_cache"
   
   # Model save path (an existing checkpoint is loaded instead of retraining)
   MODEL_SAVE_DIR = f"DATA/{SAMPLE}/checkpoint.pt"
   
   # Color maps defined in config.py
   celltype_color = getattr(config, "colormap_coloncatlas_18")
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
   n_clusters = 5    # number of KMeans clusters (= number of simulated niches)
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

      AnnData object with n_obs × n_vars = 360000 × 6008
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

      [Log] num of gene overlap 6008
      [Log] input gene number is 6008
      [Log] rna ref size is (6008, 18)

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

      [Log] Active Cells (before filtering): 8030
      [Log] Valid Cells (1 <= spots <= 200): 8030
      [Log] Filtered Cells (< 1 spots): 0
      [Log] Filtered Cells (> 200 spots): 0
      [Log] 8030 cells are detected
      [Log] Active Niches: 8030, Cells per Niche (k): 20

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
      [Log] 107126 edges generated from 8030 nodes (k=12)

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

      [Log] Building & saving mini-batches: 100%|██████████| 1/1 [00:15<00:00, 15.87s/it]

.. container:: cell-output

   .. code-block:: text

      [Log] Saved 1 blocks → DATA/Simulation1_HD/mini_batch_cache
      [Log] Original ST is patchfied to 1 blocks with avg 360000 spots, 8030 cells and 8030 niches per block

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

      Epoch 1: 100%|██████████| 1/1 [00:02<00:00,  2.30s/it, cl=5.3526, nll=812.9199, entp=-5.3385]
      Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  1.10it/s, cl=5.6618, nll=789.8260, entp=-5.1718]
      Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.6578, nll=779.3951, entp=-4.9870]
      Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.6309, nll=773.7040, entp=-4.7568]
      Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.6074, nll=770.1455, entp=-4.5855]
      Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.5836, nll=767.3323, entp=-4.5112]
      Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.5583, nll=764.5542, entp=-4.5215]
      Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.5329, nll=761.5325, entp=-4.5851]
      Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.5062, nll=758.3461, entp=-4.6587]
      Epoch 10: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.4858, nll=755.2198, entp=-4.7049]
      Epoch 11: 100%|██████████| 1/1 [00:00<00:00,  1.09it/s, cl=5.4634, nll=752.2742, entp=-4.7085]
      Epoch 12: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.4379, nll=749.4686, entp=-4.6856]
      Epoch 13: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.4067, nll=746.7878, entp=-4.6576]
      Epoch 14: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.3749, nll=744.2272, entp=-4.6306]
      Epoch 15: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.3395, nll=741.8354, entp=-4.6064]
      Epoch 16: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.3062, nll=739.6917, entp=-4.5840]
      Epoch 17: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.2746, nll=737.7842, entp=-4.5608]
      Epoch 18: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.2462, nll=736.1412, entp=-4.5355]
      Epoch 19: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.2209, nll=734.7562, entp=-4.5068]
      Epoch 20: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.1992, nll=733.5441, entp=-4.4752]
      Epoch 21: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.1797, nll=732.5952, entp=-4.4411]
      Epoch 22: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.1653, nll=731.7027, entp=-4.4047]
      Epoch 23: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.1516, nll=730.9031, entp=-4.3672]
      Epoch 24: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.1429, nll=730.1756, entp=-4.3296]
      Epoch 25: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.1318, nll=729.4441, entp=-4.2910]
      Epoch 26: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.1239, nll=728.8054, entp=-4.2543]
      Epoch 27: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.1178, nll=728.1505, entp=-4.2207]
      Epoch 28: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.1104, nll=727.6463, entp=-4.1891]
      Epoch 29: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.1027, nll=727.0778, entp=-4.1585]
      Epoch 30: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0981, nll=726.6256, entp=-4.1333]
      Epoch 31: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0931, nll=726.2007, entp=-4.1082]
      Epoch 32: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0907, nll=725.8009, entp=-4.0872]
      Epoch 33: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0882, nll=725.4782, entp=-4.0677]
      Epoch 34: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0834, nll=725.1411, entp=-4.0515]
      Epoch 35: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0819, nll=724.8298, entp=-4.0369]
      Epoch 36: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0801, nll=724.5319, entp=-4.0252]
      Epoch 37: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0759, nll=724.3387, entp=-4.0151]
      Epoch 38: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0764, nll=724.0809, entp=-4.0063]
      Epoch 39: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0726, nll=723.9104, entp=-3.9994]
      Epoch 40: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0693, nll=723.6923, entp=-3.9935]
      Epoch 41: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0680, nll=723.4999, entp=-3.9897]
      Epoch 42: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0675, nll=723.3044, entp=-3.9866]
      Epoch 43: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0636, nll=723.1033, entp=-3.9843]
      Epoch 44: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0615, nll=722.9436, entp=-3.9846]
      Epoch 45: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0580, nll=722.7919, entp=-3.9866]
      Epoch 46: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0564, nll=722.6364, entp=-3.9898]
      Epoch 47: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0525, nll=722.4691, entp=-3.9958]
      Epoch 48: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0503, nll=722.3060, entp=-4.0017]
      Epoch 49: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0467, nll=722.1383, entp=-4.0101]
      Epoch 50: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0435, nll=721.9168, entp=-4.0183]
      Epoch 51: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0407, nll=721.7793, entp=-4.0289]
      Epoch 52: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0394, nll=721.6292, entp=-4.0405]
      Epoch 53: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0337, nll=721.4958, entp=-4.0517]
      Epoch 54: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0334, nll=721.2750, entp=-4.0643]
      Epoch 55: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0291, nll=721.1783, entp=-4.0745]
      Epoch 56: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0291, nll=721.0201, entp=-4.0850]
      Epoch 57: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0265, nll=720.9127, entp=-4.0933]
      Epoch 58: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0223, nll=720.7924, entp=-4.1026]
      Epoch 59: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0219, nll=720.6948, entp=-4.1090]
      Epoch 60: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0194, nll=720.5866, entp=-4.1158]
      Epoch 61: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0188, nll=720.5527, entp=-4.1211]
      Epoch 62: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0197, nll=720.4198, entp=-4.1241]
      Epoch 63: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0163, nll=720.3402, entp=-4.1272]
      Epoch 64: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0152, nll=720.2394, entp=-4.1302]
      Epoch 65: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0146, nll=720.1638, entp=-4.1320]
      Epoch 66: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0136, nll=720.0781, entp=-4.1335]
      Epoch 67: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0116, nll=719.9964, entp=-4.1327]
      Epoch 68: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0127, nll=719.9026, entp=-4.1323]
      Epoch 69: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0093, nll=719.8142, entp=-4.1319]
      Epoch 70: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0072, nll=719.7574, entp=-4.1307]
      Epoch 71: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0047, nll=719.6639, entp=-4.1299]
      Epoch 72: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0047, nll=719.5667, entp=-4.1303]
      Epoch 73: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=5.0027, nll=719.4982, entp=-4.1268]
      Epoch 74: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=5.0001, nll=719.4257, entp=-4.1277]
      Epoch 75: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9991, nll=719.3697, entp=-4.1259]
      Epoch 76: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9971, nll=719.2684, entp=-4.1259]
      Epoch 77: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9947, nll=719.2272, entp=-4.1292]
      Epoch 78: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9930, nll=719.1243, entp=-4.1282]
      Epoch 79: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9909, nll=719.0768, entp=-4.1314]
      Epoch 80: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9885, nll=719.0030, entp=-4.1333]
      Epoch 81: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9853, nll=718.9241, entp=-4.1368]
      Epoch 82: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9852, nll=718.8873, entp=-4.1412]
      Epoch 83: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9833, nll=718.8145, entp=-4.1477]
      Epoch 84: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9796, nll=718.7367, entp=-4.1541]
      Epoch 85: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9780, nll=718.7256, entp=-4.1619]
      Epoch 86: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9734, nll=718.6085, entp=-4.1710]
      Epoch 87: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9713, nll=718.5483, entp=-4.1827]
      Epoch 88: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9688, nll=718.4721, entp=-4.1942]
      Epoch 89: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9657, nll=718.4639, entp=-4.2068]
      Epoch 90: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9613, nll=718.3104, entp=-4.2224]
      Epoch 91: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9570, nll=718.2993, entp=-4.2378]
      Epoch 92: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9528, nll=718.2053, entp=-4.2542]
      Epoch 93: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9471, nll=718.1406, entp=-4.2717]
      Epoch 94: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9422, nll=718.0732, entp=-4.2897]
      Epoch 95: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9403, nll=718.0431, entp=-4.3068]
      Epoch 96: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9372, nll=717.9548, entp=-4.3218]
      Epoch 97: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9311, nll=717.8939, entp=-4.3351]
      Epoch 98: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9290, nll=717.8359, entp=-4.3469]
      Epoch 99: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9258, nll=717.7274, entp=-4.3569]
      Epoch 100: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9256, nll=717.7240, entp=-4.3618]
      Epoch 101: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9260, nll=717.6066, entp=-4.3686]
      Epoch 102: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9251, nll=717.5067, entp=-4.3711]
      Epoch 103: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9251, nll=717.4597, entp=-4.3743]
      Epoch 104: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9284, nll=717.3746, entp=-4.3761]
      Epoch 105: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9295, nll=717.2898, entp=-4.3777]
      Epoch 106: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9297, nll=717.1984, entp=-4.3769]
      Epoch 107: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9324, nll=717.1489, entp=-4.3797]
      Epoch 108: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9335, nll=717.0866, entp=-4.3809]
      Epoch 109: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9310, nll=717.0262, entp=-4.3839]
      Epoch 110: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9323, nll=716.9454, entp=-4.3834]
      Epoch 111: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9304, nll=716.9236, entp=-4.3869]
      Epoch 112: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9323, nll=716.8126, entp=-4.3876]
      Epoch 113: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9314, nll=716.7863, entp=-4.3898]
      Epoch 114: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9312, nll=716.7135, entp=-4.3899]
      Epoch 115: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9304, nll=716.6785, entp=-4.3933]
      Epoch 116: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9308, nll=716.6393, entp=-4.3945]
      Epoch 117: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9314, nll=716.5912, entp=-4.3943]
      Epoch 118: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9311, nll=716.5122, entp=-4.3950]
      Epoch 119: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9319, nll=716.5065, entp=-4.3960]
      Epoch 120: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9321, nll=716.4979, entp=-4.3981]
      Epoch 121: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9315, nll=716.3921, entp=-4.3990]
      Epoch 122: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9320, nll=716.3645, entp=-4.4001]
      Epoch 123: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9316, nll=716.3232, entp=-4.4007]
      Epoch 124: 100%|██████████| 1/1 [00:00<00:00,  1.10it/s, cl=4.9328, nll=716.2675, entp=-4.4017]
      Epoch 125: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9334, nll=716.2231, entp=-4.4030]
      Epoch 126: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9347, nll=716.2263, entp=-4.4039]
      Epoch 127: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9353, nll=716.1748, entp=-4.4042]
      Epoch 128: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9346, nll=716.0996, entp=-4.4061]
      Epoch 129: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9362, nll=716.0842, entp=-4.4072]
      Epoch 130: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9372, nll=716.0636, entp=-4.4095]
      Epoch 131: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9381, nll=715.9886, entp=-4.4107]
      Epoch 132: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9365, nll=715.9918, entp=-4.4116]
      Epoch 133: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9377, nll=715.9472, entp=-4.4146]
      Epoch 134: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9395, nll=715.9362, entp=-4.4132]
      Epoch 135: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9384, nll=715.9005, entp=-4.4161]
      Epoch 136: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9384, nll=715.8727, entp=-4.4179]
      Epoch 137: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9371, nll=715.8189, entp=-4.4192]
      Epoch 138: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9379, nll=715.8013, entp=-4.4213]
      Epoch 139: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9370, nll=715.7780, entp=-4.4232]
      Epoch 140: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9387, nll=715.7253, entp=-4.4241]
      Epoch 141: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9391, nll=715.7222, entp=-4.4267]
      Epoch 142: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9393, nll=715.6783, entp=-4.4292]
      Epoch 143: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9401, nll=715.6865, entp=-4.4308]
      Epoch 144: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9408, nll=715.6261, entp=-4.4327]
      Epoch 145: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9388, nll=715.6111, entp=-4.4356]
      Epoch 146: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9390, nll=715.5815, entp=-4.4373]
      Epoch 147: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9392, nll=715.5494, entp=-4.4387]
      Epoch 148: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9393, nll=715.5618, entp=-4.4415]
      Epoch 149: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9393, nll=715.5031, entp=-4.4446]
      Epoch 150: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9412, nll=715.5131, entp=-4.4479]
      Epoch 151: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9399, nll=715.4840, entp=-4.4498]
      Epoch 152: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9416, nll=715.4425, entp=-4.4523]
      Epoch 153: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9404, nll=715.4496, entp=-4.4546]
      Epoch 154: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9399, nll=715.3829, entp=-4.4577]
      Epoch 155: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9409, nll=715.3804, entp=-4.4592]
      Epoch 156: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9412, nll=715.3618, entp=-4.4640]
      Epoch 157: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9418, nll=715.3103, entp=-4.4650]
      Epoch 158: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9400, nll=715.3035, entp=-4.4678]
      Epoch 159: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9410, nll=715.2856, entp=-4.4709]
      Epoch 160: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9399, nll=715.2762, entp=-4.4733]
      Epoch 161: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9394, nll=715.2422, entp=-4.4768]
      Epoch 162: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9416, nll=715.1974, entp=-4.4770]
      Epoch 163: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9409, nll=715.2416, entp=-4.4805]
      Epoch 164: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9410, nll=715.1633, entp=-4.4829]
      Epoch 165: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9413, nll=715.1855, entp=-4.4846]
      Epoch 166: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9421, nll=715.1392, entp=-4.4850]
      Epoch 167: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9398, nll=715.1305, entp=-4.4878]
      Epoch 168: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9408, nll=715.0890, entp=-4.4909]
      Epoch 169: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9425, nll=715.0770, entp=-4.4923]
      Epoch 170: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9410, nll=715.0696, entp=-4.4956]
      Epoch 171: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9414, nll=715.0471, entp=-4.4966]
      Epoch 172: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9396, nll=715.0264, entp=-4.4973]
      Epoch 173: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9417, nll=714.9924, entp=-4.4992]
      Epoch 174: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9410, nll=714.9964, entp=-4.5011]
      Epoch 175: 100%|██████████| 1/1 [00:00<00:00,  1.10it/s, cl=4.9407, nll=714.9621, entp=-4.5022]
      Epoch 176: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9421, nll=714.9515, entp=-4.5054]
      Epoch 177: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9427, nll=714.9541, entp=-4.5072]
      Epoch 178: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9412, nll=714.9543, entp=-4.5079]
      Epoch 179: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9424, nll=714.8833, entp=-4.5086]
      Epoch 180: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9422, nll=714.8697, entp=-4.5112]
      Epoch 181: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9408, nll=714.8793, entp=-4.5131]
      Epoch 182: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9422, nll=714.8097, entp=-4.5166]
      Epoch 183: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9421, nll=714.7949, entp=-4.5163]
      Epoch 184: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9419, nll=714.7927, entp=-4.5190]
      Epoch 185: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9426, nll=714.7816, entp=-4.5213]
      Epoch 186: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9429, nll=714.7527, entp=-4.5231]
      Epoch 187: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9426, nll=714.7203, entp=-4.5241]
      Epoch 188: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9413, nll=714.7446, entp=-4.5233]
      Epoch 189: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9426, nll=714.7295, entp=-4.5268]
      Epoch 190: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9398, nll=714.7135, entp=-4.5267]
      Epoch 191: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9421, nll=714.6541, entp=-4.5294]
      Epoch 192: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9414, nll=714.6477, entp=-4.5296]
      Epoch 193: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9413, nll=714.6396, entp=-4.5309]
      Epoch 194: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9427, nll=714.6482, entp=-4.5317]
      Epoch 195: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9418, nll=714.6276, entp=-4.5318]
      Epoch 196: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9408, nll=714.6032, entp=-4.5341]
      Epoch 197: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9396, nll=714.5724, entp=-4.5345]
      Epoch 198: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9415, nll=714.5762, entp=-4.5356]
      Epoch 199: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9424, nll=714.5787, entp=-4.5371]
      Epoch 200: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9406, nll=714.5585, entp=-4.5375]
      Epoch 201: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9430, nll=714.5292, entp=-4.5376]
      Epoch 202: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9423, nll=714.5522, entp=-4.5390]
      Epoch 203: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9441, nll=714.4783, entp=-4.5395]
      Epoch 204: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9426, nll=714.4908, entp=-4.5403]
      Epoch 205: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9401, nll=714.4327, entp=-4.5415]
      Epoch 206: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9424, nll=714.4500, entp=-4.5411]
      Epoch 207: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9431, nll=714.4561, entp=-4.5422]
      Epoch 208: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9438, nll=714.4050, entp=-4.5442]
      Epoch 209: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9418, nll=714.4188, entp=-4.5436]
      Epoch 210: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9414, nll=714.3778, entp=-4.5444]
      Epoch 211: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9419, nll=714.3857, entp=-4.5451]
      Epoch 212: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9426, nll=714.3590, entp=-4.5473]
      Epoch 213: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9423, nll=714.3816, entp=-4.5467]
      Epoch 214: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9424, nll=714.3337, entp=-4.5490]
      Epoch 215: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9413, nll=714.3569, entp=-4.5482]
      Epoch 216: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9422, nll=714.2922, entp=-4.5500]
      Epoch 217: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9420, nll=714.3126, entp=-4.5498]
      Epoch 218: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9416, nll=714.2985, entp=-4.5504]
      Epoch 219: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9424, nll=714.2905, entp=-4.5514]
      Epoch 220: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9407, nll=714.2532, entp=-4.5521]
      Epoch 221: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9397, nll=714.2793, entp=-4.5533]
      Epoch 222: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9403, nll=714.2552, entp=-4.5533]
      Epoch 223: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9420, nll=714.2427, entp=-4.5537]
      Epoch 224: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9413, nll=714.2145, entp=-4.5546]
      Epoch 225: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9403, nll=714.2080, entp=-4.5546]
      Epoch 226: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9399, nll=714.1885, entp=-4.5556]
      Epoch 227: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9391, nll=714.1717, entp=-4.5571]
      Epoch 228: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9413, nll=714.1700, entp=-4.5581]
      Epoch 229: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9405, nll=714.1600, entp=-4.5589]
      Epoch 230: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9404, nll=714.1304, entp=-4.5597]
      Epoch 231: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9398, nll=714.1193, entp=-4.5598]
      Epoch 232: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9377, nll=714.1469, entp=-4.5608]
      Epoch 233: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9388, nll=714.1139, entp=-4.5619]
      Epoch 234: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9388, nll=714.1183, entp=-4.5627]
      Epoch 235: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9361, nll=714.1000, entp=-4.5643]
      Epoch 236: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9375, nll=714.0834, entp=-4.5649]
      Epoch 237: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9372, nll=714.0721, entp=-4.5672]
      Epoch 238: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9360, nll=714.0645, entp=-4.5672]
      Epoch 239: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9350, nll=714.0668, entp=-4.5695]
      Epoch 240: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9368, nll=714.0690, entp=-4.5691]
      Epoch 241: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9351, nll=714.0508, entp=-4.5729]
      Epoch 242: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9330, nll=714.0377, entp=-4.5744]
      Epoch 243: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9328, nll=714.0659, entp=-4.5748]
      Epoch 244: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9342, nll=714.0072, entp=-4.5769]
      Epoch 245: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9331, nll=713.9889, entp=-4.5779]
      Epoch 246: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9327, nll=714.0048, entp=-4.5810]
      Epoch 247: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9322, nll=714.0292, entp=-4.5832]
      Epoch 248: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9322, nll=713.9971, entp=-4.5841]
      Epoch 249: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9298, nll=713.9617, entp=-4.5851]
      Epoch 250: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9297, nll=713.9801, entp=-4.5887]
      Epoch 251: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9290, nll=713.9579, entp=-4.5912]
      Epoch 252: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9283, nll=713.9832, entp=-4.5932]
      Epoch 253: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9289, nll=713.9434, entp=-4.5966]
      Epoch 254: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9276, nll=713.9344, entp=-4.5985]
      Epoch 255: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9267, nll=713.9421, entp=-4.6014]
      Epoch 256: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9260, nll=713.9253, entp=-4.6042]
      Epoch 257: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9254, nll=713.9302, entp=-4.6064]
      Epoch 258: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9245, nll=713.9020, entp=-4.6089]
      Epoch 259: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9243, nll=713.9143, entp=-4.6138]
      Epoch 260: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9241, nll=713.9141, entp=-4.6179]
      Epoch 261: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9230, nll=713.8820, entp=-4.6220]
      Epoch 262: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9227, nll=713.8969, entp=-4.6245]
      Epoch 263: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9197, nll=713.8845, entp=-4.6282]
      Epoch 264: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9202, nll=713.8705, entp=-4.6318]
      Epoch 265: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9193, nll=713.8248, entp=-4.6348]
      Epoch 266: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9195, nll=713.8442, entp=-4.6394]
      Epoch 267: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9179, nll=713.8600, entp=-4.6424]
      Epoch 268: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9198, nll=713.8530, entp=-4.6445]
      Epoch 269: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9179, nll=713.8478, entp=-4.6488]
      Epoch 270: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9169, nll=713.8099, entp=-4.6526]
      Epoch 271: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9170, nll=713.8248, entp=-4.6553]
      Epoch 272: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9168, nll=713.8134, entp=-4.6587]
      Epoch 273: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9151, nll=713.8080, entp=-4.6640]
      Epoch 274: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9141, nll=713.8140, entp=-4.6679]
      Epoch 275: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9138, nll=713.7979, entp=-4.6697]
      Epoch 276: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9132, nll=713.7967, entp=-4.6753]
      Epoch 277: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9120, nll=713.7905, entp=-4.6806]
      Epoch 278: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9132, nll=713.7759, entp=-4.6842]
      Epoch 279: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9137, nll=713.7631, entp=-4.6896]
      Epoch 280: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9125, nll=713.7708, entp=-4.6953]
      Epoch 281: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9101, nll=713.7750, entp=-4.7014]
      Epoch 282: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9102, nll=713.7675, entp=-4.7055]
      Epoch 283: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9114, nll=713.6995, entp=-4.7125]
      Epoch 284: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9088, nll=713.7140, entp=-4.7180]
      Epoch 285: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9089, nll=713.6935, entp=-4.7229]
      Epoch 286: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9116, nll=713.6868, entp=-4.7290]
      Epoch 287: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9104, nll=713.6510, entp=-4.7320]
      Epoch 288: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9084, nll=713.6554, entp=-4.7350]
      Epoch 289: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9079, nll=713.6252, entp=-4.7400]
      Epoch 290: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9087, nll=713.6370, entp=-4.7445]
      Epoch 291: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9075, nll=713.6423, entp=-4.7458]
      Epoch 292: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9092, nll=713.5857, entp=-4.7502]
      Epoch 293: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9086, nll=713.5961, entp=-4.7526]
      Epoch 294: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9086, nll=713.5759, entp=-4.7546]
      Epoch 295: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9086, nll=713.5441, entp=-4.7582]
      Epoch 296: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9111, nll=713.5425, entp=-4.7610]
      Epoch 297: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9081, nll=713.5103, entp=-4.7626]
      Epoch 298: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9094, nll=713.5283, entp=-4.7660]
      Epoch 299: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9085, nll=713.4922, entp=-4.7681]
      Epoch 300: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9092, nll=713.4567, entp=-4.7702]
      Epoch 301: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9076, nll=713.4618, entp=-4.7728]
      Epoch 302: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9078, nll=713.4537, entp=-4.7772]
      Epoch 303: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9072, nll=713.4324, entp=-4.7772]
      Epoch 304: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9073, nll=713.4158, entp=-4.7794]
      Epoch 305: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9082, nll=713.4409, entp=-4.7819]
      Epoch 306: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=713.3998, entp=-4.7832]
      Epoch 307: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=713.3986, entp=-4.7854]
      Epoch 308: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9089, nll=713.3745, entp=-4.7875]
      Epoch 309: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9093, nll=713.3813, entp=-4.7904]
      Epoch 310: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9074, nll=713.3636, entp=-4.7933]
      Epoch 311: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9076, nll=713.3375, entp=-4.7947]
      Epoch 312: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=713.3224, entp=-4.7977]
      Epoch 313: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9078, nll=713.3088, entp=-4.7988]
      Epoch 314: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9069, nll=713.2898, entp=-4.8017]
      Epoch 315: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9069, nll=713.3020, entp=-4.8019]
      Epoch 316: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=713.2895, entp=-4.8054]
      Epoch 317: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9070, nll=713.2974, entp=-4.8073]
      Epoch 318: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9075, nll=713.2542, entp=-4.8092]
      Epoch 319: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=713.2441, entp=-4.8110]
      Epoch 320: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=713.2419, entp=-4.8143]
      Epoch 321: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9071, nll=713.1882, entp=-4.8158]
      Epoch 322: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9069, nll=713.2117, entp=-4.8167]
      Epoch 323: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9053, nll=713.2264, entp=-4.8188]
      Epoch 324: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9065, nll=713.2134, entp=-4.8196]
      Epoch 325: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=713.2151, entp=-4.8221]
      Epoch 326: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=713.2034, entp=-4.8236]
      Epoch 327: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9060, nll=713.1593, entp=-4.8253]
      Epoch 328: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9061, nll=713.1331, entp=-4.8266]
      Epoch 329: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9074, nll=713.1461, entp=-4.8266]
      Epoch 330: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9068, nll=713.1528, entp=-4.8283]
      Epoch 331: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9064, nll=713.1158, entp=-4.8295]
      Epoch 332: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9069, nll=713.1101, entp=-4.8313]
      Epoch 333: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9061, nll=713.1019, entp=-4.8320]
      Epoch 334: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9075, nll=713.1082, entp=-4.8321]
      Epoch 335: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9064, nll=713.0739, entp=-4.8328]
      Epoch 336: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9065, nll=713.0455, entp=-4.8330]
      Epoch 337: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9075, nll=713.0781, entp=-4.8347]
      Epoch 338: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=713.0415, entp=-4.8333]
      Epoch 339: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9087, nll=713.0374, entp=-4.8353]
      Epoch 340: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=713.0519, entp=-4.8369]
      Epoch 341: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9074, nll=713.0310, entp=-4.8371]
      Epoch 342: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9066, nll=713.0355, entp=-4.8374]
      Epoch 343: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=713.0135, entp=-4.8373]
      Epoch 344: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=713.0068, entp=-4.8385]
      Epoch 345: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9055, nll=712.9954, entp=-4.8396]
      Epoch 346: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9072, nll=712.9888, entp=-4.8391]
      Epoch 347: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9080, nll=712.9561, entp=-4.8397]
      Epoch 348: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9067, nll=713.0011, entp=-4.8404]
      Epoch 349: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9076, nll=712.9711, entp=-4.8404]
      Epoch 350: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9061, nll=712.9642, entp=-4.8407]
      Epoch 351: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9055, nll=712.9641, entp=-4.8413]
      Epoch 352: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9063, nll=712.9094, entp=-4.8407]
      Epoch 353: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9072, nll=712.9244, entp=-4.8410]
      Epoch 354: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9068, nll=712.9102, entp=-4.8428]
      Epoch 355: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9067, nll=712.9437, entp=-4.8407]
      Epoch 356: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9076, nll=712.9194, entp=-4.8416]
      Epoch 357: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9074, nll=712.9150, entp=-4.8413]
      Epoch 358: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9066, nll=712.9054, entp=-4.8420]
      Epoch 359: 100%|██████████| 1/1 [00:00<00:00,  1.10it/s, cl=4.9085, nll=712.8655, entp=-4.8423]
      Epoch 360: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9078, nll=712.9028, entp=-4.8428]
      Epoch 361: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9083, nll=712.8564, entp=-4.8430]
      Epoch 362: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9066, nll=712.8499, entp=-4.8437]
      Epoch 363: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9086, nll=712.8766, entp=-4.8431]
      Epoch 364: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9084, nll=712.8755, entp=-4.8432]
      Epoch 365: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9089, nll=712.8525, entp=-4.8436]
      Epoch 366: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9075, nll=712.8541, entp=-4.8445]
      Epoch 367: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=712.8438, entp=-4.8450]
      Epoch 368: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9088, nll=712.8243, entp=-4.8445]
      Epoch 369: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9084, nll=712.8520, entp=-4.8451]
      Epoch 370: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=712.8312, entp=-4.8439]
      Epoch 371: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9059, nll=712.8405, entp=-4.8458]
      Epoch 372: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=712.8209, entp=-4.8457]
      Epoch 373: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9080, nll=712.7994, entp=-4.8451]
      Epoch 374: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9056, nll=712.7997, entp=-4.8449]
      Epoch 375: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9065, nll=712.7982, entp=-4.8451]
      Epoch 376: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9062, nll=712.7795, entp=-4.8456]
      Epoch 377: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9056, nll=712.8108, entp=-4.8456]
      Epoch 378: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9069, nll=712.7710, entp=-4.8451]
      Epoch 379: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9068, nll=712.7902, entp=-4.8450]
      Epoch 380: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9077, nll=712.7776, entp=-4.8456]
      Epoch 381: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9070, nll=712.7800, entp=-4.8449]
      Epoch 382: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9073, nll=712.7737, entp=-4.8448]
      Epoch 383: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9070, nll=712.7660, entp=-4.8453]
      Epoch 384: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9042, nll=712.7711, entp=-4.8455]
      Epoch 385: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9066, nll=712.7343, entp=-4.8457]
      Epoch 386: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9054, nll=712.7227, entp=-4.8448]
      Epoch 387: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9059, nll=712.7380, entp=-4.8455]
      Epoch 388: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9048, nll=712.7372, entp=-4.8464]
      Epoch 389: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9054, nll=712.7415, entp=-4.8465]
      Epoch 390: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9062, nll=712.7435, entp=-4.8458]
      Epoch 391: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9048, nll=712.7252, entp=-4.8468]
      Epoch 392: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9036, nll=712.7258, entp=-4.8467]
      Epoch 393: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9063, nll=712.6869, entp=-4.8466]
      Epoch 394: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9037, nll=712.6885, entp=-4.8477]
      Epoch 395: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9044, nll=712.6801, entp=-4.8480]
      Epoch 396: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9040, nll=712.7024, entp=-4.8471]
      Epoch 397: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9052, nll=712.7158, entp=-4.8486]
      Epoch 398: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9035, nll=712.6893, entp=-4.8475]
      Epoch 399: 100%|██████████| 1/1 [00:00<00:00,  1.10it/s, cl=4.9030, nll=712.6812, entp=-4.8486]
      Epoch 400: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9043, nll=712.6957, entp=-4.8481]
      Epoch 401: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9031, nll=712.6794, entp=-4.8476]
      Epoch 402: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9045, nll=712.7100, entp=-4.8474]
      Epoch 403: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9042, nll=712.6786, entp=-4.8470]
      Epoch 404: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9037, nll=712.6332, entp=-4.8476]
      Epoch 405: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9037, nll=712.6638, entp=-4.8458]
      Epoch 406: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9040, nll=712.6635, entp=-4.8474]
      Epoch 407: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9024, nll=712.6544, entp=-4.8475]
      Epoch 408: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9010, nll=712.6387, entp=-4.8469]
      Epoch 409: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9015, nll=712.6426, entp=-4.8469]
      Epoch 410: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9013, nll=712.6502, entp=-4.8474]
      Epoch 411: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9023, nll=712.6759, entp=-4.8478]
      Epoch 412: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9006, nll=712.6415, entp=-4.8482]
      Epoch 413: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9014, nll=712.6525, entp=-4.8485]
      Epoch 414: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9014, nll=712.6303, entp=-4.8489]
      Epoch 415: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8998, nll=712.6201, entp=-4.8492]
      Epoch 416: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9004, nll=712.6271, entp=-4.8492]
      Epoch 417: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9008, nll=712.6130, entp=-4.8495]
      Epoch 418: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9004, nll=712.6124, entp=-4.8494]
      Epoch 419: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8977, nll=712.6362, entp=-4.8497]
      Epoch 420: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8981, nll=712.5999, entp=-4.8513]
      Epoch 421: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8972, nll=712.5703, entp=-4.8510]
      Epoch 422: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8977, nll=712.5848, entp=-4.8516]
      Epoch 423: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8980, nll=712.5703, entp=-4.8518]
      Epoch 424: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8963, nll=712.5823, entp=-4.8540]
      Epoch 425: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8965, nll=712.5681, entp=-4.8521]
      Epoch 426: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8964, nll=712.5646, entp=-4.8542]
      Epoch 427: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8958, nll=712.5368, entp=-4.8546]
      Epoch 428: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8949, nll=712.5601, entp=-4.8547]
      Epoch 429: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8951, nll=712.5101, entp=-4.8557]
      Epoch 430: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8938, nll=712.5363, entp=-4.8565]
      Epoch 431: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8944, nll=712.5156, entp=-4.8563]
      Epoch 432: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8930, nll=712.4794, entp=-4.8584]
      Epoch 433: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8943, nll=712.4833, entp=-4.8589]
      Epoch 434: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8898, nll=712.4614, entp=-4.8602]
      Epoch 435: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8910, nll=712.4304, entp=-4.8618]
      Epoch 436: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8917, nll=712.4125, entp=-4.8639]
      Epoch 437: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8898, nll=712.3208, entp=-4.8677]
      Epoch 438: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8876, nll=712.2071, entp=-4.8733]
      Epoch 439: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8871, nll=711.9540, entp=-4.8839]
      Epoch 440: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8827, nll=711.2427, entp=-4.9026]
      Epoch 441: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8844, nll=709.4694, entp=-4.9375]
      Epoch 442: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8814, nll=707.2357, entp=-4.9733]
      Epoch 443: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8823, nll=705.8269, entp=-4.9945]
      Epoch 444: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8851, nll=705.3021, entp=-5.0026]
      Epoch 445: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8925, nll=705.4094, entp=-5.0071]
      Epoch 446: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8993, nll=705.7726, entp=-5.0083]
      Epoch 447: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9048, nll=705.9611, entp=-5.0091]
      Epoch 448: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9060, nll=705.5903, entp=-5.0097]
      Epoch 449: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9062, nll=705.1102, entp=-5.0095]
      Epoch 450: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9061, nll=704.6893, entp=-5.0103]
      Epoch 451: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9057, nll=704.4758, entp=-5.0098]
      Epoch 452: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9065, nll=704.2980, entp=-5.0083]
      Epoch 453: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9071, nll=704.1460, entp=-5.0078]
      Epoch 454: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9051, nll=704.0667, entp=-5.0063]
      Epoch 455: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9074, nll=703.9779, entp=-5.0045]
      Epoch 456: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9080, nll=703.9155, entp=-5.0018]
      Epoch 457: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9094, nll=703.8004, entp=-5.0009]
      Epoch 458: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9086, nll=703.8508, entp=-4.9998]
      Epoch 459: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9085, nll=703.7759, entp=-5.0009]
      Epoch 460: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9096, nll=703.7665, entp=-5.0021]
      Epoch 461: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9073, nll=703.7120, entp=-5.0075]
      Epoch 462: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9064, nll=703.6733, entp=-5.0166]
      Epoch 463: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9054, nll=703.6680, entp=-5.0282]
      Epoch 464: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9052, nll=703.6181, entp=-5.0472]
      Epoch 465: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.9018, nll=703.5168, entp=-5.0708]
      Epoch 466: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.9016, nll=703.3632, entp=-5.0909]
      Epoch 467: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8992, nll=703.3508, entp=-5.1014]
      Epoch 468: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8974, nll=703.3025, entp=-5.1021]
      Epoch 469: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8963, nll=703.2497, entp=-5.0998]
      Epoch 470: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8934, nll=703.1874, entp=-5.1030]
      Epoch 471: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8922, nll=703.0361, entp=-5.1088]
      Epoch 472: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8903, nll=702.9584, entp=-5.1150]
      Epoch 473: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8856, nll=702.8839, entp=-5.1214]
      Epoch 474: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8844, nll=702.8003, entp=-5.1255]
      Epoch 475: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8836, nll=702.7729, entp=-5.1269]
      Epoch 476: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8838, nll=702.7462, entp=-5.1291]
      Epoch 477: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8813, nll=702.6487, entp=-5.1311]
      Epoch 478: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8805, nll=702.5448, entp=-5.1330]
      Epoch 479: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8797, nll=702.5411, entp=-5.1327]
      Epoch 480: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8791, nll=702.4848, entp=-5.1319]
      Epoch 481: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8810, nll=702.4212, entp=-5.1308]
      Epoch 482: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8790, nll=702.4563, entp=-5.1296]
      Epoch 483: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8797, nll=702.3812, entp=-5.1294]
      Epoch 484: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8792, nll=702.3645, entp=-5.1299]
      Epoch 485: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8806, nll=702.3057, entp=-5.1316]
      Epoch 486: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8786, nll=702.2696, entp=-5.1321]
      Epoch 487: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8800, nll=702.2945, entp=-5.1317]
      Epoch 488: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8800, nll=702.2295, entp=-5.1325]
      Epoch 489: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8799, nll=702.2055, entp=-5.1325]
      Epoch 490: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8798, nll=702.1702, entp=-5.1323]
      Epoch 491: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8805, nll=702.1726, entp=-5.1320]
      Epoch 492: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8799, nll=702.1550, entp=-5.1317]
      Epoch 493: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8803, nll=702.1506, entp=-5.1318]
      Epoch 494: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8804, nll=702.1088, entp=-5.1316]
      Epoch 495: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8783, nll=702.0708, entp=-5.1321]
      Epoch 496: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8813, nll=702.0699, entp=-5.1321]
      Epoch 497: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8789, nll=702.0817, entp=-5.1323]
      Epoch 498: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8780, nll=702.0333, entp=-5.1325]
      Epoch 499: 100%|██████████| 1/1 [00:00<00:00,  1.12it/s, cl=4.8797, nll=702.0351, entp=-5.1329]
      Epoch 500: 100%|██████████| 1/1 [00:00<00:00,  1.11it/s, cl=4.8794, nll=702.0185, entp=-5.1329]

.. container:: cell-output

   .. code-block:: text

      Training finished!
      Model saved to DATA/Simulation1_HD/checkpoint.pt

.. container:: cell-output

   .. code-block:: text

      generating embeddings (slice 0)...: 100%|██████████| 1/1 [00:00<00:00,  1.40it/s]

6. Cell-type evaluation
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

.. image:: _images/1c8e9ae2b43f813ead800a0a5fcbc446a2536622d4171a30cb2dd9a80c5123c3.png
   :alt: Saved result from Simulated tumor microenvironment
   :width: 100%

.. image:: _images/1895530203e0345004ed049790dcea9116d2b3b00728e003da56f2ab138d238d.png
   :alt: Saved result from Simulated tumor microenvironment
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

      evaluated cells: 8030/8030 (excluding NA and ambiguous)
      cells predicted as a type absent from the ground truth: 0 (0.00%), types: []
      Accuracy     : 0.8232
      F1 (macro)   : 0.6835
      F1 (weighted): 0.7871
      
                          precision    recall  f1-score   support
      
                     CAF       0.96      1.00      0.98      1159
              CD4 T cell       0.88      1.00      0.94       424
              CD8 T cell       1.00      0.93      0.96       746
             Endothelial       0.99      0.99      0.99       145
              Epithelial       0.97      1.00      0.98       591
              Fibroblast       0.97      0.93      0.95      1078
                  Goblet       1.00      1.00      1.00       465
              Macrophage       0.97      1.00      0.99       479
                Mature B       1.00      0.97      0.99       355
                  Plasma       0.93      1.00      0.96       232
      SM Stress Response       0.00      0.00      0.00       141
           Smooth Muscle       0.00      0.00      0.00       618
                 Tumor I       0.73      1.00      0.84       506
                Tumor II       0.00      0.00      0.00       397
               Tumor III       0.96      1.00      0.98       284
                Tumor IV       0.28      0.99      0.44       133
                 Tumor V       0.00      0.00      0.00       126
                     vSM       0.18      1.00      0.30       151
      
                accuracy                           0.82      8030
               macro avg       0.66      0.77      0.68      8030
            weighted avg       0.77      0.82      0.79      8030

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

      niche: 5 clusters
      spots with a niche label: 320110 / 360000

.. image:: _images/1f1f4949e3ac37688c0db512ad9523b6b5b4e3f44a1616c1d0bbe4471917537a.png
   :alt: Saved result from Simulated tumor microenvironment
   :width: 100%

.. container:: cell-output

   .. code-block:: text

      evaluated cells: 8030 / 8030
      GT niches: 5,  pred niches: 5
      NMI = 0.8463
      ARI = 0.8809

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

.. image:: _images/5455d5c14d84fccb82c1121d7c035035272b88e3c8f1de0554daff809264517b.png
   :alt: Saved result from Simulated tumor microenvironment
   :width: 100%

