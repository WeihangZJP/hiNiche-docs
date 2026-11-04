Colorectal cancer
================================================================================

:download:`Download original notebook <examples/HDST_CRC_run_hiNiche_multi.ipynb>`

Code and saved outputs are reproduced from this notebook. No cells are executed during documentation builds.

These source notebooks expect preprocessed files produced by preprocess_multislice.py (including meta.pkl and per-slice AnnData files). That script is not present in the supplied project directory; obtain it and prepare these inputs before running this notebook.

.. rubric:: hiNiche: multi-slice mouse brain Visium HD analysis

.. raw:: html

   <p>Joint multi-scale (spot / cell / niche) cell-type deconvolution and niche discovery across several Visium HD slices.</p>

.. raw:: html

   <p><strong>Environment:</strong> run this notebook in the main <strong><code>hiNiche</code></strong> environment (<code>environment.yml</code>).</p>

.. raw:: html

   <p><strong>Workflow.</strong> Preprocessing is done slice by slice with <code>preprocess_multislice.py</code> (all preprocessing parameters are set at the top of that script), so the slices are never loaded together:</p>

.. code-block:: bash

   python preprocess_multislice.py --mode genes                  # shared genes + reference
   python preprocess_multislice.py --mode slice --slice-idx 0    # one job per slice
   python preprocess_multislice.py --mode slice --slice-idx 1
   python preprocess_multislice.py --mode slice --slice-idx 2
   python preprocess_multislice.py --mode collect                # meta.pkl for training

.. raw:: html

   <p>This notebook then trains all slices jointly and analyzes the results.</p>

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
   import pickle
   import warnings
   
   # --- Data handling ---
   import numpy as np
   import pandas as pd
   import anndata as ad
   
   # --- Deep learning ---
   import torch
   import torch.nn.functional as F
   
   # --- Clustering ---
   import rapids_singlecell as rsc     # GPU-accelerated neighbors / Leiden / UMAP
   from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
   
   # --- Plotting ---
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches
   from matplotlib.colors import LinearSegmentedColormap
   
   %load_ext autoreload
   %autoreload 2
   
   warnings.filterwarnings("ignore")

.. container:: cell-output

   .. code-block:: text

      The autoreload extension is already loaded. To reload it, use:
        %reload_ext autoreload

2. Parameter configuration
--------------------------------------------------------------------------------

.. raw:: html

   <p>Preprocessing and training hyperparameters are stored in <code>meta.pkl</code> by <code>preprocess_multislice.py</code>; only notebook-side settings are set here.</p>

.. code-block:: python
   :linenos:

   SAMPLE = "CRC_HD_multi_slices"
   
   # Output directory of preprocess_multislice.py (OUT_DIR): meta.pkl, adata_slice{i}.h5ad
   PREPROCESS_DIR = f"DATA/{SAMPLE}/preprocessed"
   
   #Model save path (if the file exists, the model is loaded and training is skipped)
   MODEL_SAVE_DIR = f"DATA/{SAMPLE}/checkpoint.pt"
   
   # Cell-type color map defined in config.py
   celltype_color = getattr(config, "colormap_coloncatlas_18")
   umap_color = getattr(config, "umap_40")
   # Slice / batch colors: blue -> pink (with 3 slices: 0.12 / 0.5 / 0.88 of the gradient)
   bluepink = LinearSegmentedColormap.from_list(
       "bluepink", ["#2b3fb0", "#6f8be0", "#b6a3d8", "#e58bbf", "#ff6fae"])
   
   # Model / training hyperparameters
   lr          = 0.005
   num_epochs  = 30
   edge_drop   = 0.5
   member_drop = 0.5
   neg_samples = 5
   tau         = 1
   alpha       = 1
   beta        = 1        # dataset-specific: mouse brain 2, CRC 1
   seed        = 42
   
   # Niche clustering / merging
   emb_k        = 50      # neighbors of the KNN graph on embeddings (Leiden / UMAP)
   resolution   = 0.8     # Leiden resolution; higher -> more clusters
   merge_thresh = 0.05    # cosine distance below which niche clusters with similar composition are merged
   
   device = "cuda:0"

3. Load preprocessed slices
--------------------------------------------------------------------------------

.. raw:: html

   <p><code>meta.pkl</code> holds the training inputs (unit numbers and patch directory of each slice, reference matrix, hyperparameters). For each slice only the spot coordinates and the spot → cell matrix are loaded from <code>adata_slice{i}.h5ad</code>; the expression matrix stays on disk (<code>backed=&quot;r&quot;</code>).</p>

.. code-block:: python
   :linenos:

   with open(os.path.join(PREPROCESS_DIR, "meta.pkl"), "rb") as f:
       meta = pickle.load(f)
   
   SLICES     = meta["slice_names"]          # training slice index i <-> SLICES[i]
   n_slice    = len(SLICES)
   slice_colors = [bluepink(v) for v in np.linspace(0.12, 0.88, n_slice)]   # one color per slice (all slice / batch plots)
   P          = meta["params"]               # preprocessing + training hyperparameters
   celltypes  = meta["celltypes"]
   ref_matrix = meta["ref_matrix"]
   active_niche_indices_list = meta["active_niche_indices_list"]
   
   print(f"{n_slice} slices: {SLICES} | input genes {meta['in_dim']} | cell types {meta['hidden_dim']}")
   for k, v in P.items():
       print(f"  {k:20s} = {v}")
   
   # Per-slice spot coordinates and spot -> cell matrices
   spatial_grid_list, spatial_pixel_list, S_s2c_list = [], [], []
   for i, s in enumerate(SLICES):
       a = ad.read_h5ad(os.path.join(PREPROCESS_DIR, f"adata_slice{i}.h5ad"), backed="r")
       spatial_grid_list.append(np.asarray(a.obsm["spatial_grid"]))
       spatial_pixel_list.append(np.asarray(a.obsm["spatial_pixel"]))
       S_s2c_list.append(utils.sp_to_torch_sparse(a.obsm["S_s2c"]))
       a.file.close()
   
       # Consistency check with the cached mini-batches
       n = meta["sample_number_list"][i]
       assert S_s2c_list[i].shape == (n["spot"], n["cell"]), f"{s}: adata_slice{i}.h5ad does not match meta.pkl"
       print(f"[Log] {s}: {n['spot']} spots, {n['cell']} cells, {n['niche']} niches")

.. container:: cell-output

   .. code-block:: text

      3 slices: ['slice1', 'slice2', 'slice3'] | input genes 6008 | cell types 18
        bg_threshold         = 50
        bg_rings             = 4
        max_spots_per_cell   = 100
        min_spots_per_cell   = 3
        Patch_size           = 512
        halo                 = 32
        graph_knn            = 12
        niche_knn            = 15
      [Log] slice1: 5704076 spots, 297969 cells, 297969 niches
      [Log] slice2: 7432511 spots, 373439 cells, 373439 niches
      [Log] slice3: 7087246 spots, 289027 cells, 289027 niches

4. Joint multi-slice training
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   emb_all, slice_id = run_train_seq(meta["sample_number_list"],
                                     PATCH_LOC = meta["PATCH_LOC"],
                                     MODEL_SAVE_DIR = MODEL_SAVE_DIR,
                                     ref_matrix = ref_matrix,
                                     lr = lr,
                                     in_dim = meta["in_dim"],
                                     hidden_dim = meta["hidden_dim"],
                                     num_epochs = num_epochs,
                                     edge_drop = edge_drop,
                                     member_drop = member_drop,
                                     neg_samples = neg_samples,
                                     tau = tau,
                                     alpha = alpha,
                                     beta = beta,
                                     seed = seed,
                                     device = device)
   # emb_all[level]: embeddings of all slices concatenated; slice_id[level]: slice index of each row

.. container:: cell-output

   .. code-block:: text

      Epoch 1: 100%|██████████| 128/128 [00:45<00:00,  2.83it/s, cl=5.2788, nll=344.7968, entp=-4.1412]
      Epoch 2: 100%|██████████| 128/128 [00:29<00:00,  4.35it/s, cl=5.1498, nll=336.4618, entp=-3.7094]
      Epoch 3: 100%|██████████| 128/128 [00:29<00:00,  4.36it/s, cl=5.1225, nll=335.5056, entp=-3.6363]
      Epoch 4: 100%|██████████| 128/128 [00:29<00:00,  4.41it/s, cl=5.0970, nll=335.2065, entp=-3.6463]
      Epoch 5: 100%|██████████| 128/128 [00:28<00:00,  4.44it/s, cl=5.0755, nll=335.0396, entp=-3.6535]
      Epoch 6: 100%|██████████| 128/128 [00:28<00:00,  4.45it/s, cl=5.0645, nll=334.9708, entp=-3.6291]
      Epoch 7: 100%|██████████| 128/128 [00:28<00:00,  4.48it/s, cl=5.0481, nll=334.8431, entp=-3.6699]
      Epoch 8: 100%|██████████| 128/128 [00:28<00:00,  4.50it/s, cl=5.0367, nll=334.7532, entp=-3.6970]
      Epoch 9: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=5.0295, nll=334.6798, entp=-3.7007]
      Epoch 10: 100%|██████████| 128/128 [00:28<00:00,  4.52it/s, cl=5.0187, nll=334.6885, entp=-3.7324]
      Epoch 11: 100%|██████████| 128/128 [00:28<00:00,  4.53it/s, cl=5.0129, nll=334.6224, entp=-3.7244]
      Epoch 12: 100%|██████████| 128/128 [00:28<00:00,  4.52it/s, cl=5.0077, nll=334.5856, entp=-3.7350]
      Epoch 13: 100%|██████████| 128/128 [00:28<00:00,  4.48it/s, cl=5.0039, nll=334.5953, entp=-3.7382]
      Epoch 14: 100%|██████████| 128/128 [00:28<00:00,  4.50it/s, cl=4.9988, nll=334.5287, entp=-3.7570]
      Epoch 15: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9948, nll=334.5307, entp=-3.7540]
      Epoch 16: 100%|██████████| 128/128 [00:28<00:00,  4.46it/s, cl=4.9908, nll=334.5113, entp=-3.7582]
      Epoch 17: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9866, nll=334.4691, entp=-3.7798]
      Epoch 18: 100%|██████████| 128/128 [00:28<00:00,  4.47it/s, cl=4.9824, nll=334.5052, entp=-3.7839]
      Epoch 19: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9787, nll=334.4460, entp=-3.7885]
      Epoch 20: 100%|██████████| 128/128 [00:28<00:00,  4.49it/s, cl=4.9777, nll=334.4224, entp=-3.7848]
      Epoch 21: 100%|██████████| 128/128 [00:28<00:00,  4.48it/s, cl=4.9758, nll=334.4114, entp=-3.7942]
      Epoch 22: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9716, nll=334.4129, entp=-3.8012]
      Epoch 23: 100%|██████████| 128/128 [00:28<00:00,  4.52it/s, cl=4.9710, nll=334.4378, entp=-3.7964]
      Epoch 24: 100%|██████████| 128/128 [00:28<00:00,  4.52it/s, cl=4.9711, nll=334.3937, entp=-3.7625]
      Epoch 25: 100%|██████████| 128/128 [00:28<00:00,  4.52it/s, cl=4.9658, nll=334.3666, entp=-3.7918]
      Epoch 26: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9624, nll=334.3644, entp=-3.8071]
      Epoch 27: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9642, nll=334.3834, entp=-3.7933]
      Epoch 28: 100%|██████████| 128/128 [00:28<00:00,  4.51it/s, cl=4.9585, nll=334.3417, entp=-3.8148]
      Epoch 29: 100%|██████████| 128/128 [00:28<00:00,  4.49it/s, cl=4.9572, nll=334.3560, entp=-3.8215]
      Epoch 30: 100%|██████████| 128/128 [00:28<00:00,  4.49it/s, cl=4.9562, nll=334.3345, entp=-3.8051]

.. container:: cell-output

   .. code-block:: text

      Training finished!
      Model saved to DATA/CRC_HD_multi_slices/checkpoint.pt

.. container:: cell-output

   .. code-block:: text

      generating embeddings (slice 0)...: 100%|██████████| 45/45 [00:05<00:00,  7.50it/s]
      generating embeddings (slice 1)...: 100%|██████████| 41/41 [00:10<00:00,  3.81it/s]
      generating embeddings (slice 2)...: 100%|██████████| 42/42 [00:08<00:00,  5.01it/s]

5. Re-assign unmapped spots
--------------------------------------------------------------------------------

.. code-block:: python
   :linenos:

   # Re-assign unmapped (background) spots to nearby cells using spot embeddings, slice by slice
   S_s2c_updated_list = []
   for b, s in enumerate(SLICES):
       spot_embs = emb_all["spot"][slice_id["spot"] == b].detach().cpu().numpy()
       new_assignments = utils.reassign_unmapped_spots(
           S_s2c_list[b],
           spatial_grid_list[b],                # grid coordinates of this slice, 1 unit = 1 spot
           spot_embs,
           algorithm="fixed_boundary",
           fixed_distance=2,
       )
       print(f"[Log] {s}: {len(new_assignments)} background spots re-assigned")
       S_s2c_updated_list.append(utils.update_S(S_s2c_list[b], new_assignments))

.. container:: cell-output

   .. code-block:: text

      Total unassigned spots: 3773477

.. container:: cell-output

   .. code-block:: text

      Assigning spots (boundary): 100%|██████████| 3773477/3773477 [01:58<00:00, 31781.30it/s]

.. container:: cell-output

   .. code-block:: text

      [Log] slice1: 2448207 background spots re-assigned
      Total unassigned spots: 4876154

.. container:: cell-output

   .. code-block:: text

      Assigning spots (boundary): 100%|██████████| 4876154/4876154 [02:25<00:00, 33537.20it/s]

.. container:: cell-output

   .. code-block:: text

      [Log] slice2: 2911315 background spots re-assigned
      Total unassigned spots: 5177174

.. container:: cell-output

   .. code-block:: text

      Assigning spots (boundary): 100%|██████████| 5177174/5177174 [02:16<00:00, 38011.16it/s]

.. container:: cell-output

   .. code-block:: text

      [Log] slice3: 2703439 background spots re-assigned

6. Visualization
--------------------------------------------------------------------------------

6.1 Spot- and cell-level cell-type maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def type_labels(prob_max, prob_idx):
       """Cell-type label per row: "filtered" (no cell), "ambiguous" (max prob < 0.1) or the argmax type."""
       labels = []
       for p_, i_ in zip(prob_max, prob_idx):
           if i_ == -1:
               labels.append("filtered")
           elif p_ < 0.1:
               labels.append("ambiguous")
           else:
               labels.append(celltypes[i_])
       return labels
   
   keys = ["spot", "cell"]
   fig, axes = plt.subplots(n_slice, 2, figsize=(6, 3 * n_slice), dpi=600, squeeze=False)
   labels_all_list = []
   for b, s in enumerate(SLICES):
       x_raw, y_raw = spatial_pixel_list[b][:, 0], spatial_pixel_list[b][:, 1]
       S = S_s2c_updated_list[b].coalesce()
   
       labels_all = {}
       for ci, check in enumerate(keys):
           prob = F.softmax(emb_all[check][slice_id[check] == b], dim=1)   # embeddings of this slice
           max_prob_raw, max_idx_raw = torch.max(prob, dim=1)
   
           if check == "cell":
               # Broadcast each cell's prediction to its spots (spot -> cell mapping)
               rows, cols = S.indices()[0], S.indices()[1]
               spot2target = torch.full((S.size(0),), -1, dtype=torch.long, device=S.device)
               spot2target[rows] = cols
               max_idx  = torch.where(spot2target >= 0, max_idx_raw[spot2target],  torch.tensor(-1, device=S.device))
               max_prob = torch.where(spot2target >= 0, max_prob_raw[spot2target], torch.tensor(0.0, device=S.device))
           else:
               max_prob, max_idx = max_prob_raw, max_idx_raw
   
           labels_all[check] = type_labels(max_prob.cpu().numpy(), max_idx.cpu().numpy())
           ax = axes[b][ci]
           ax.scatter(x_raw, y_raw, c=[celltype_color[n] for n in labels_all[check]], s=0.5, marker='.', linewidths=0)
           ax.set_title(f"{s} · {check}")
           ax.invert_yaxis(); ax.axis("off")
       labels_all_list.append(labels_all)
   
   plt.tight_layout()
   plt.show()
   
   # --- Legend (shared by all slices) ---
   present = sorted({l for la in labels_all_list for l in la["spot"]})
   handles = [mpatches.Patch(color=celltype_color.get(c, "#cccccc"), label=c) for c in present]
   figL, axL = plt.subplots(figsize=(11, 0.6)); axL.axis("off")
   axL.legend(handles=handles, ncol=7, frameon=False, loc="center", fontsize=9)
   plt.show()

.. image:: _images/101294a83378c98243f4ecd7a6fc335b899b53e63fa54ab93bb52cdb6154a10f.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

.. image:: _images/06ae5129744728e44f719cb756f9c9ab5aacf5c8e2367ec6a75d557732e609a1.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

6.2 Joint niche clustering across slices (UMAP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Leiden clustering of the niche embeddings of all slices together, so niche labels are shared across slices.</p>

.. code-block:: python
   :linenos:

   import torch
   torch.cuda.empty_cache()
   # --- Cluster niche embeddings of all slices (Leiden) + UMAP, sharing one neighbor graph ---
   adata_niche = ad.AnnData(emb_all["niche"].detach().cpu().numpy().astype(np.float32))
   rsc.pp.neighbors(adata_niche, use_rep="X", n_neighbors=emb_k)
   rsc.tl.leiden(adata_niche, resolution=resolution, random_state=seed)
   rsc.tl.umap(adata_niche, random_state=seed)
   niche_label = adata_niche.obs["leiden"].astype(int).values
   niche_umap  = adata_niche.obsm["X_umap"]
   clusters_id = {"niche": niche_label}
   sid_niche   = slice_id["niche"].numpy()
   print(f"niche: {len(np.unique(niche_label))} clusters")
   
   # --- Left: dominant cell type; middle: niche cluster; right: slice (integration check) ---
   fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
   axes[0].scatter(niche_umap[:, 0], niche_umap[:, 1],
                   c=[umap_color[f"Cluster_{v % 40}"] for v in niche_label], s=1, linewidths=0)
   axes[0].set_title(f"niche cluster (n={len(np.unique(niche_label))})")
   
   axes[1].scatter(niche_umap[:, 0], niche_umap[:, 1],
                   c=[slice_colors[b] for b in sid_niche], s=1, linewidths=0)
   axes[1].set_title("slice (integration)")
   axes[1].legend(handles=[mpatches.Patch(color=slice_colors[b], label=s) for b, s in enumerate(SLICES)],
                  fontsize=7, frameon=False, loc="upper right")
   
   for ax in axes:
       ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
   plt.tight_layout()
   plt.show()

.. container:: cell-output

   .. code-block:: text

      niche: 25 clusters

.. image:: _images/0271000429fadc614aa82fc2a1f83f1bd5287afb5dd1f271a286c5af1be6449c.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

.. code-block:: python
   :linenos:

   # --- Left: dominant cell type; middle: niche cluster; right: slice (integration check) ---
   fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
   axes[0].scatter(niche_umap[:, 0], niche_umap[:, 1],
                   c=[umap_color[f"Cluster_{v % 40}"] for v in clusters_id["niche_merged"]], s=1, linewidths=0)
   axes[0].set_title(f"niche cluster (n={len(np.unique(clusters_id["niche_merged"]))})")
   
   axes[1].scatter(niche_umap[:, 0], niche_umap[:, 1],
                   c=[slice_colors[b] for b in sid_niche], s=1, linewidths=0)
   axes[1].set_title("slice (integration)")
   axes[1].legend(handles=[mpatches.Patch(color=slice_colors[b], label=s) for b, s in enumerate(SLICES)],
                  fontsize=7, frameon=False, loc="upper right")
   
   for ax in axes:
       ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
   plt.tight_layout()
   plt.show()

.. image:: _images/bc313e8b83e2d4bd5c2e699481b618230cc097cdc774ad3240004541eb066a79.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

6.3 Merge niches by cell-type composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>The composition of a niche cluster is the cell-type distribution of the niche centers (niche j of slice b is centered at cell <code>active_niche_indices_list[b][j]</code>).</p>

.. code-block:: python
   :linenos:

   # ---- Cell type of each cell, then of each niche center (aligned with the rows of emb_all["niche"]) ----
   cell_type = np.asarray(celltypes)[F.softmax(emb_all["cell"], dim=1).argmax(1).cpu().numpy()]
   sid_cell = slice_id["cell"].numpy()
   niche_center_type = np.concatenate([cell_type[sid_cell == b][np.asarray(active_niche_indices_list[b])]
                                       for b in range(n_slice)])
   
   # ---- Composition table (before merge) ----
   df_counts = pd.crosstab(np.asarray(clusters_id["niche"]), niche_center_type)
   
   # ---- Hierarchical clustering + merge ----
   comp = df_counts.div(df_counts.sum(1), axis=0)              # row-normalized composition
   Z = linkage(comp.values, method="average", metric="cosine")
   merged = fcluster(Z, t=merge_thresh, criterion="distance")
   group_members = {}
   for old, g in zip(comp.index, merged):
       group_members.setdefault(g, []).append(int(old))
   old2new = {o: min(members) for members in group_members.values() for o in members}   # smallest id in each group becomes the new label
   
   print(f"{len(comp)} niche clusters -> {len(set(old2new.values()))} merged")
   for rep in sorted(set(old2new.values())):
       members = sorted(o for o, r in old2new.items() if r == rep)
       if len(members) > 1:
           print(f"  {members} -> {rep}")
   
   niche_merged = np.array([old2new[l] for l in clusters_id["niche"]])
   clusters_id["niche_merged"] = niche_merged
   
   # ---- Composition tables as proportions (each bar stacks to 1) ----
   dfm = pd.crosstab(niche_merged, niche_center_type)
   comp_before = comp
   comp_after  = dfm.div(dfm.sum(1), axis=0)
   
   # ================= Layout: top row before/after merge, bottom row full-width dendrogram =================
   fig = plt.figure(figsize=(12, 9))
   gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.45, wspace=0.3)
   ax_before = fig.add_subplot(gs[0, 0])
   ax_after  = fig.add_subplot(gs[0, 1])
   ax_dendro = fig.add_subplot(gs[1, :])
   
   def stacked(ax, df, title, xlabel):
       """Stacked bar chart of cell-type proportions per niche; x tick labels colored by niche."""
       df.plot(kind="bar", stacked=True, ax=ax,
               color=[celltype_color[c] for c in df.columns], width=0.9, legend=False)
       for tick, nid in zip(ax.get_xticklabels(), df.index):
           tick.set_color(umap_color[f"Cluster_{nid % 40}"]); tick.set_fontweight("bold")
       ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("cell type proportion")
       ax.set_ylim(0, 1)
   
   stacked(ax_before, comp_before, "Before merge", "niche")
   stacked(ax_after,  comp_after,  "After merge",  "merged niche (min id)")
   
   handles, labels = ax_before.get_legend_handles_labels()
   fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.92), loc="upper left",
              title="Cell Type", prop={"size": 8})
   
   dendrogram(Z, labels=comp.index.tolist(), ax=ax_dendro)
   ax_dendro.axhline(merge_thresh, ls="--", c="r")
   ax_dendro.set_ylabel("1 - cosine")
   ax_dendro.set_title("Niche similarity (average linkage, cosine)")
   
   fig.suptitle("Cell-type composition of niches: before vs after merge", fontsize=14)
   plt.show()

.. container:: cell-output

   .. code-block:: text

      25 niche clusters -> 19 merged
        [1, 7, 15] -> 1
        [9, 12] -> 9
        [18, 19, 20, 21] -> 18

.. image:: _images/dae284a5278e6ab7c439349eb4c2a1e3fb1b1352ad57701be8493a91b3eda8fa.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

6.4 Spatial maps per slice: cell types and niches before / after merge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
   :linenos:

   def col(labels):
       """Niche colors depend only on the label id, so colors stay consistent before / after merge."""
       return [umap_color[f"Cluster_{l % 40}"] for l in labels]
   
   fig, axes = plt.subplots(n_slice, 3, figsize=(12, 4 * n_slice), dpi=300, squeeze=False)
   for b, s in enumerate(SLICES):
       # Cell position = mean pixel coordinate of its spots (after re-assignment)
       S = S_s2c_updated_list[b].coalesce()
       s_rows, c_cols = S.indices()
       n_cell = S.size(1)
       spatial = torch.as_tensor(spatial_pixel_list[b], dtype=torch.float32)
       cell_pos = torch.zeros((n_cell, 2)).index_add_(0, c_cols, spatial[s_rows])
       cell_cnt = torch.zeros(n_cell).index_add_(0, c_cols, torch.ones_like(c_cols, dtype=torch.float32))
       cell_pos = (cell_pos / cell_cnt.clamp(min=1).unsqueeze(1)).numpy()
   
       # Niche labels of this slice mapped to their center cells (-1 = not a niche center)
       m = sid_niche == b
       active = np.asarray(active_niche_indices_list[b])
       cell_niche = np.full(n_cell, -1); cell_niche[active] = clusters_id["niche"][m]
       cell_niche_merged = np.full(n_cell, -1); cell_niche_merged[active] = clusters_id["niche_merged"][m]
       keep = (cell_cnt.numpy() > 0) & (cell_niche >= 0)
       ct_b = cell_type[sid_cell == b]
   
       axes[b][0].scatter(cell_pos[keep, 0], cell_pos[keep, 1], c=[celltype_color[c] for c in ct_b[keep]], s=0.5, linewidths=0)
       axes[b][0].set_title(f"{s} · cell type")
       axes[b][1].scatter(cell_pos[keep, 0], cell_pos[keep, 1], c=col(cell_niche[keep]), s=0.5, linewidths=0)
       axes[b][1].set_title(f"{s} · niche (before merge)")
       axes[b][2].scatter(cell_pos[keep, 0], cell_pos[keep, 1], c=col(cell_niche_merged[keep]), s=0.5, linewidths=0)
       axes[b][2].set_title(f"{s} · niche (after merge)")
       for ax in axes[b]:
           ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
   
   plt.tight_layout()
   plt.show()

.. image:: _images/bb7b347166a260feea2abd25a8ffebd5bdb5d17d65e3ad738b51618814a5b02d.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

.. code-block:: python
   :linenos:

   # --- Save niche cluster ids (before / after merge) and niche UMAP coordinates ---
   # One row per niche (row order = rows of emb_all["niche"], all slices concatenated)
   niche_out = pd.DataFrame({
       "slice":              np.asarray(SLICES)[sid_niche],
       "center_cell":        np.concatenate([np.asarray(active_niche_indices_list[b]) for b in range(n_slice)]),   # cell index within its slice
       "niche_before_merge": np.asarray(clusters_id["niche"]),
       "niche_after_merge":  np.asarray(clusters_id["niche_merged"]),
       "UMAP1":              niche_umap[:, 0],
       "UMAP2":              niche_umap[:, 1],
   })
   niche_out.index.name = "niche"
   niche_out_path = f"DATA/{SAMPLE}/niche_clusters.csv"
   niche_out.to_csv(niche_out_path)
   print(f"[Log] saved {len(niche_out)} niches -> {niche_out_path}")
   niche_out.head()

.. container:: cell-output

   .. code-block:: text

      [Log] saved 960435 niches -> DATA/CRC_HD_multi_slices/niche_clusters.csv

.. container:: cell-output

   .. code-block:: text

      slice  center_cell  niche_before_merge  niche_after_merge      UMAP1  \
      niche                                                                          
      0      slice1            0                  15                  1  11.242119   
      1      slice1            1                  15                  1  10.614147   
      2      slice1            2                   8                  8 -16.305706   
      3      slice1            3                  15                  1  11.782028   
      4      slice1            4                   8                  8  12.479117   
      
                 UMAP2  
      niche             
      0      15.606508  
      1      15.525043  
      2      -7.138694  
      3      15.091286  
      4      14.672501

6.5 Cell embedding UMAP (integration check)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>UMAP of the cell embeddings of all slices, colored by predicted cell type and by slice.</p>

.. code-block:: python
   :linenos:

   adata_cell = ad.AnnData(emb_all["cell"].detach().cpu().numpy().astype(np.float32))
   rsc.pp.neighbors(adata_cell, use_rep="X", n_neighbors=emb_k)
   rsc.tl.umap(adata_cell, random_state=seed)
   cell_umap = adata_cell.obsm["X_umap"]
   
   fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
   axes[0].scatter(cell_umap[:, 0], cell_umap[:, 1], c=[celltype_color.get(c, "#cccccc") for c in cell_type], s=1, linewidths=0)
   axes[0].set_title(f"cell type (n={len(np.unique(cell_type))})")
   axes[1].scatter(cell_umap[:, 0], cell_umap[:, 1], c=[slice_colors[b] for b in sid_cell], s=1, linewidths=0)
   axes[1].set_title("slice (integration)")
   axes[1].legend(handles=[mpatches.Patch(color=slice_colors[b], label=s) for b, s in enumerate(SLICES)],
                  fontsize=7, frameon=False, loc="upper right")
   for ax in axes:
       ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
   plt.tight_layout()
   plt.show()

.. image:: _images/81c26aa127a8814b11b2c0e390cf1531410d3e4fefaaf828e48f3996cab58f76.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

.. code-block:: python
   :linenos:

   targets   = [3, 6, 10, 11]
   NICHE_KEY = "niche_merged"       # "niche" (before merge) or "niche_merged" (after merge)
   grey, red = "#e0e0e0", "#e6194b"
   
   # ---- Per slice: cell positions and the niche label of each cell (via the niche's center cell) ----
   cell_xy_list, cell_niche_list = [], []
   for b in range(n_slice):
       S = S_s2c_updated_list[b].coalesce()
       s_rows, c_cols = S.indices()
       n_cell = S.size(1)
       spatial = torch.as_tensor(spatial_pixel_list[b], dtype=torch.float32)
       cell_pos = torch.zeros((n_cell, 2)).index_add_(0, c_cols, spatial[s_rows])
       cell_cnt = torch.zeros(n_cell).index_add_(0, c_cols, torch.ones_like(c_cols, dtype=torch.float32))
       cell_pos = (cell_pos / cell_cnt.clamp(min=1).unsqueeze(1)).numpy()
   
       lab = np.full(n_cell, -1)                                   # -1 = not a niche center
       lab[np.asarray(active_niche_indices_list[b])] = np.asarray(clusters_id[NICHE_KEY])[sid_niche == b]
   
       keep = cell_cnt.numpy() > 0                                 # cells with at least one spot
       cell_xy_list.append(cell_pos[keep])
       cell_niche_list.append(lab[keep])
   
   # ---- Highlight each target niche (red) on all cells (grey), one row per slice ----
   fig, axes = plt.subplots(n_slice, len(targets), figsize=(4 * len(targets), 4 * n_slice), dpi=300, squeeze=False)
   for b in range(n_slice):
       xy, lab = cell_xy_list[b], cell_niche_list[b]
       for j, t in enumerate(targets):
           ax = axes[b][j]
           ax.scatter(xy[:, 0], xy[:, 1], c=grey, s=0.5, linewidths=0, rasterized=True)
           hit = lab == t
           ax.scatter(xy[hit, 0], xy[hit, 1], c=red, s=0.5, linewidths=0, rasterized=True, zorder=3)
           ax.set_title(f"{SLICES[b]} · niche {t}  ({hit.sum()})")
           ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
   
   plt.tight_layout()
   plt.savefig("niche_highlight.png")
   plt.show()

.. image:: _images/44c26cefe812201996859093debe5e168f8716c9d23e404bba88dcb6ce4bc584.png
   :alt: Saved result from Colorectal cancer
   :width: 100%

