Xenium
================================================================================

:download:`Download original notebook <examples/Xenium_run_hiNiche.ipynb>`

Code and saved outputs are reproduced from this notebook. No cells are executed during documentation builds.

.. rubric:: hiNiche on single-cell spatial data (single slice): human breast cancer Xenium

.. raw:: html

   <p><strong>Environment:</strong> run this notebook in the main <strong><code>hiNiche</code></strong> environment (<code>environment.yml</code>).</p>

.. raw:: html

   <p>This notebook applies hiNiche (<code>hiNiche_img</code>, cell → niche) to a single Xenium slice of human breast cancer. The cell / niche embeddings are written to the AnnData and clustered with GPU-accelerated Leiden (<code>rapids_singlecell</code>). This dataset has <strong>no ground-truth labels</strong>, so the clusters are only visualized (no NMI / ARI evaluation).</p>

1. Imports
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   import config
   import utils
   import model
   from train import run_train_img
   
   import anndata as ad
   import scanpy as sc
   import numpy as np
   import pandas as pd
   
   import matplotlib.pyplot as plt
   
   import os
   import glob
   import warnings
   import json
   from tqdm import tqdm
   
   # GPU-accelerated single-cell tools (Leiden clustering).
   # cupy (used by rapids_singlecell) compiles GPU kernels at runtime and needs the CUDA headers shipped inside the conda environment
   import sys
   os.environ.setdefault("CUDA_PATH", os.path.join(sys.prefix, "targets", "x86_64-linux"))
   import rapids_singlecell as rsc
   
   %load_ext autoreload
   %autoreload 2
   
   warnings.filterwarnings("ignore")

.. container:: cell-output

   .. code-block:: text

      /hpc/home/wz220/miniconda3/envs/hiNiche/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
        from .autonotebook import tqdm as notebook_tqdm

2. Configuration
--------------------------------------------------------------------------------

.. raw:: html

   <p>Data paths, the patch directory (<code>PATCH_LOC</code>) and model / training hyperparameters. There is no ground-truth column, so the clustering granularity is set by the Leiden resolution (<code>leiden_resolution</code>).</p>

.. code-block:: python
   :linenos:

   SAMPLE = "Breast_cancer_Xenium_single_slice"
   
   # Data paths: a single-slice AnnData (raw counts in adata.X)
   h5_path = f"DATA/{SAMPLE}/Xenium_Breast_Cancer.h5ad"
   
   # Patch save path
   PATCH_LOC = f"DATA/{SAMPLE}/mini_batch_cache"
   
   # Model save path (if the file exists, the model is loaded and training is skipped)
   MODEL_SAVE_DIR = f"DATA/{SAMPLE}/checkpoint.pt"
   
   # Model / training hyperparameters
   single_patch = True    # the whole slice is used as a single patch
   n_pcs       = 30
   graph_knn   = 12
   niche_knn   = 20
   hidden_dim  = 32
   lr          = 0.005
   num_epochs  = 100
   edge_drop   = 0.5
   member_drop = 0.5
   neg_samples = 5
   tau         = 1
   seed        = 42
   leiden_knn  = 15       # neighbors of the KNN graph used for Leiden
   leiden_resolution = 0.2   # no ground truth: higher -> more clusters
   device      = "cuda:0"

3. Load data
--------------------------------------------------------------------------------

.. raw:: html

   <p>Read the AnnData and show per-cell QC metrics (cell and nucleus area) in space.</p>

.. code-block:: python
   :linenos:

   adata = sc.read(h5_path)
   adata.var_names_make_unique()
   print(adata)
   
   sc.pl.embedding(adata, basis="spatial", color=["cell_area", "nucleus_area"], s=10)

.. container:: cell-output

   .. code-block:: text

      AnnData object with n_obs × n_vars = 167780 × 313
          obs: 'x_centroid', 'y_centroid', 'transcript_counts', 'control_probe_counts', 'control_codeword_counts', 'total_counts', 'cell_area', 'nucleus_area'
          var: 'ensemble_id', 'type'
          obsm: 'spatial'

.. image:: _images/f37e03744cba645b77e7a5c4348799c40ed6ed947d08d20212abaa75bd11d32e.png
   :alt: Saved result from Xenium
   :width: 100%

4. Build graphs, niches and mini-batch
--------------------------------------------------------------------------------

.. raw:: html

   <p>Normalize + log1p the raw counts, scale + PCA, build KNN niches (<code>niche_knn</code> cells per niche), the cell graph (expression PCA KNN) and the niche graph (spatial KNN), then cache the slice as a single block in <code>PATCH_LOC</code>.</p>

.. code-block:: python
   :linenos:

   adata.layers["counts"] = adata.X.copy()
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   
   X_backup = adata.X.copy()
   sc.pp.scale(adata)
   sc.tl.pca(adata, n_comps=n_pcs)
   adata.X = X_backup
   
   adata.obsm["X_input"] = adata.X
   adata.obsm["X_mle"] = adata.layers["counts"]
   spatial_coords_px = adata.obsm["spatial"]
   
   # Each niche = a center cell and its niche_knn nearest cells (including itself)
   S_c2n_all, _, _ = utils.build_cell_niche_mapping_knn(spatial_coords_px, k_niche = niche_knn)
   
   cell_edges = utils.build_spot_knn_graph(adata.obsm["X_pca"], graph_knn)    # cell graph: expression (PCA) KNN
   niche_edges = utils.build_spot_knn_graph(spatial_coords_px, graph_knn)     # niche graph: spatial KNN
   batch_mask, block_ids = utils.make_cell_block_masks(spatial_coords_px, S_c2n = S_c2n_all, single_patch = single_patch)
   
   # Remove old blocks, otherwise the loader would also read leftovers from a previous run
   for f in glob.glob(os.path.join(PATCH_LOC, "block_*.pt")):
       os.remove(f)
   utils.build_cell_subgraph_data(adata, ref_matrix = None,
                                  cell_edges = cell_edges,
                                  niche_edges = niche_edges,
                                  S_c2n_all = S_c2n_all,
                                  batch_mask = batch_mask,
                                  block_ids = block_ids,
                                  save_dir = PATCH_LOC)
   
   sample_number = {"cell": S_c2n_all.shape[0], "niche": S_c2n_all.shape[1]}

.. container:: cell-output

   .. code-block:: text

      [Log] Active niches: 167780, Cells per niche (k): 20
      [Log] 3352948 edges generated from 167780 nodes (k=12)
      [Log] 2263048 edges generated from 167780 nodes (k=12)

.. container:: cell-output

   .. code-block:: text

      [Log] Building & saving cell mini-batches: 100%|██████████| 1/1 [00:04<00:00,  4.07s/it]

.. container:: cell-output

   .. code-block:: text

      [Log] saved 1 cell blocks → DATA/Breast_cancer_Xenium_single_slice/mini_batch_cache

5. Training
--------------------------------------------------------------------------------

.. raw:: html

   <p>Single slice: pass a dict of unit numbers and a single patch directory. If <code>MODEL_SAVE_DIR</code> already exists, the model is loaded and training is skipped. Embeddings are written to <code>adata.obsm[&quot;X_cell_emb&quot;]</code> and <code>adata.obsm[&quot;X_niche_emb&quot;]</code>.</p>

.. code-block:: python
   :linenos:

   emb_all, _ = run_train_img(sample_number,
                              PATCH_LOC = PATCH_LOC,
                              MODEL_SAVE_DIR = MODEL_SAVE_DIR,
                              ref_matrix = None,
                              lr = lr,
                              in_dim = adata.n_vars,
                              hidden_dim = hidden_dim,
                              num_epochs = num_epochs,
                              edge_drop = edge_drop,
                              member_drop = member_drop,
                              neg_samples = neg_samples,
                              tau = tau,
                              seed = seed,
                              device = device)
   
   # With KNN niches every cell is a niche center, so niche embeddings are aligned with cells as well
   adata.obsm["X_cell_emb"]  = emb_all["cell"].detach().cpu().numpy()
   adata.obsm["X_niche_emb"] = emb_all["niche"].detach().cpu().numpy()

.. container:: cell-output

   .. code-block:: text

      Epoch 1: 100%|██████████| 1/1 [00:01<00:00,  1.24s/it, cl=3.7033]
      Epoch 2: 100%|██████████| 1/1 [00:00<00:00,  2.15it/s, cl=3.1942]
      Epoch 3: 100%|██████████| 1/1 [00:00<00:00,  2.46it/s, cl=3.0523]
      Epoch 4: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.9949]
      Epoch 5: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.9539]
      Epoch 6: 100%|██████████| 1/1 [00:00<00:00,  2.48it/s, cl=2.9313]
      Epoch 7: 100%|██████████| 1/1 [00:00<00:00,  2.49it/s, cl=2.9169]
      Epoch 8: 100%|██████████| 1/1 [00:00<00:00,  2.42it/s, cl=2.9048]
      Epoch 9: 100%|██████████| 1/1 [00:00<00:00,  2.26it/s, cl=2.8976]
      Epoch 10: 100%|██████████| 1/1 [00:00<00:00,  2.28it/s, cl=2.8868]
      Epoch 11: 100%|██████████| 1/1 [00:00<00:00,  2.29it/s, cl=2.8734]
      Epoch 12: 100%|██████████| 1/1 [00:00<00:00,  2.39it/s, cl=2.8629]
      Epoch 13: 100%|██████████| 1/1 [00:00<00:00,  2.31it/s, cl=2.8570]
      Epoch 14: 100%|██████████| 1/1 [00:00<00:00,  2.37it/s, cl=2.8517]
      Epoch 15: 100%|██████████| 1/1 [00:00<00:00,  2.44it/s, cl=2.8437]
      Epoch 16: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.8393]
      Epoch 17: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.8323]
      Epoch 18: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.8275]
      Epoch 19: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.8249]
      Epoch 20: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.8156]
      Epoch 21: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.8102]
      Epoch 22: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.8059]
      Epoch 23: 100%|██████████| 1/1 [00:00<00:00,  2.52it/s, cl=2.7983]
      Epoch 24: 100%|██████████| 1/1 [00:00<00:00,  2.50it/s, cl=2.7944]
      Epoch 25: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7903]
      Epoch 26: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7876]
      Epoch 27: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7834]
      Epoch 28: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7826]
      Epoch 29: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7806]
      Epoch 30: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7775]
      Epoch 31: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7760]
      Epoch 32: 100%|██████████| 1/1 [00:00<00:00,  2.53it/s, cl=2.7729]
      Epoch 33: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7703]
      Epoch 34: 100%|██████████| 1/1 [00:00<00:00,  2.53it/s, cl=2.7694]
      Epoch 35: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7692]
      Epoch 36: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7667]
      Epoch 37: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7662]
      Epoch 38: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7658]
      Epoch 39: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7644]
      Epoch 40: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7634]
      Epoch 41: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7641]
      Epoch 42: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7605]
      Epoch 43: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7599]
      Epoch 44: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7595]
      Epoch 45: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7597]
      Epoch 46: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7589]
      Epoch 47: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7586]
      Epoch 48: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7578]
      Epoch 49: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7567]
      Epoch 50: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7563]
      Epoch 51: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7564]
      Epoch 52: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7554]
      Epoch 53: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7561]
      Epoch 54: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7560]
      Epoch 55: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7562]
      Epoch 56: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7527]
      Epoch 57: 100%|██████████| 1/1 [00:00<00:00,  2.53it/s, cl=2.7545]
      Epoch 58: 100%|██████████| 1/1 [00:00<00:00,  2.47it/s, cl=2.7546]
      Epoch 59: 100%|██████████| 1/1 [00:00<00:00,  2.44it/s, cl=2.7534]
      Epoch 60: 100%|██████████| 1/1 [00:00<00:00,  2.41it/s, cl=2.7520]
      Epoch 61: 100%|██████████| 1/1 [00:00<00:00,  2.53it/s, cl=2.7532]
      Epoch 62: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7523]
      Epoch 63: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7524]
      Epoch 64: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7522]
      Epoch 65: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7521]
      Epoch 66: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7520]
      Epoch 67: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7509]
      Epoch 68: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7498]
      Epoch 69: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7515]
      Epoch 70: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7501]
      Epoch 71: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7504]
      Epoch 72: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7497]
      Epoch 73: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7506]
      Epoch 74: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7479]
      Epoch 75: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7515]
      Epoch 76: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7494]
      Epoch 77: 100%|██████████| 1/1 [00:00<00:00,  2.49it/s, cl=2.7481]
      Epoch 78: 100%|██████████| 1/1 [00:00<00:00,  2.47it/s, cl=2.7496]
      Epoch 79: 100%|██████████| 1/1 [00:00<00:00,  2.43it/s, cl=2.7482]
      Epoch 80: 100%|██████████| 1/1 [00:00<00:00,  2.36it/s, cl=2.7491]
      Epoch 81: 100%|██████████| 1/1 [00:00<00:00,  2.37it/s, cl=2.7483]
      Epoch 82: 100%|██████████| 1/1 [00:00<00:00,  2.33it/s, cl=2.7485]
      Epoch 83: 100%|██████████| 1/1 [00:00<00:00,  2.40it/s, cl=2.7476]
      Epoch 84: 100%|██████████| 1/1 [00:00<00:00,  2.54it/s, cl=2.7489]
      Epoch 85: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7478]
      Epoch 86: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7477]
      Epoch 87: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7468]
      Epoch 88: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7462]
      Epoch 89: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7469]
      Epoch 90: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7467]
      Epoch 91: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7473]
      Epoch 92: 100%|██████████| 1/1 [00:00<00:00,  2.55it/s, cl=2.7471]
      Epoch 93: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7470]
      Epoch 94: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.7480]
      Epoch 95: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.7464]
      Epoch 96: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.7451]
      Epoch 97: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.7464]
      Epoch 98: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7463]
      Epoch 99: 100%|██████████| 1/1 [00:00<00:00,  2.56it/s, cl=2.7465]
      Epoch 100: 100%|██████████| 1/1 [00:00<00:00,  2.57it/s, cl=2.7458]

.. container:: cell-output

   .. code-block:: text

      Training finished!
      Model saved to DATA/Breast_cancer_Xenium_single_slice/checkpoint.pt

.. container:: cell-output

   .. code-block:: text

      generating embeddings (slice 0)...: 100%|██████████| 1/1 [00:00<00:00,  2.44it/s]

6. Clustering and visualization
--------------------------------------------------------------------------------

.. raw:: html

   <p>GPU-accelerated Leiden clustering (<code>rapids_singlecell</code>) on the cell and niche embeddings, shown in space. Labels are stored in <code>adata.obs[&quot;leiden_cell&quot;]</code> / <code>adata.obs[&quot;leiden_niche&quot;]</code>.</p>

.. code-block:: python
   :linenos:

   spatial = adata.obsm["spatial"]
   x, y = spatial[:, 0], spatial[:, 1]
   
   # Leiden on the GPU: KNN graph on each embedding, then Leiden
   for level in ["cell", "niche"]:
       adata_tmp = ad.AnnData(np.asarray(adata.obsm[f"X_{level}_emb"]))
       rsc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=leiden_knn)
       rsc.tl.leiden(adata_tmp, resolution=leiden_resolution, random_state=seed)
       adata.obs[f"leiden_{level}"] = pd.Categorical(adata_tmp.obs["leiden"].values)
       print(f"{level}: {adata.obs[f'leiden_{level}'].nunique()} Leiden clusters")
   
   point_size = 0.5
   fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
   panels = [(adata.obs["leiden_cell"].cat.codes,  "Leiden cell"),
             (adata.obs["leiden_niche"].cat.codes, "Leiden niche")]
   for ax, (c, title) in zip(axes, panels):
       ax.scatter(x, y, c=c, s=point_size, linewidths=0, cmap="tab20")
       ax.set_xticks([]); ax.set_yticks([])
       ax.set_aspect("equal")
       ax.set_title(title)
   plt.tight_layout()
   plt.show()

.. container:: cell-output

   .. code-block:: text

      cell: 12 Leiden clusters
      niche: 10 Leiden clusters

.. image:: _images/b090d86dfb3a4da13bffa30c32d7d39427e3bdb852f6a7263f3a4a44bc2d860b.png
   :alt: Saved result from Xenium
   :width: 100%

