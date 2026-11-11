# Group Overlap and Data Leakage Audit Report

**Project:** FractureFuseNet (Project 12)
**Dataset:** Human Bone Fractures Multi-modal Image Dataset (HBFMID)
**Seed:** 42 | **Folds:** 5

## 1. Executive Leakage Audit Summary

To prevent data leakage caused by source image augmentations and near-identical radiographs, every image is grouped by `group_id`, which unites filename prefixes (`base_id`) and perceptual hash clusters (`dup_group_id`, dihedral Hamming distance $\le 10$).

### Verified Zero-Leakage Split Results:

| Split 1 | Split 2 | Filename Overlap | `base_id` Overlap | `dup_group_id` Overlap | `group_id` Overlap | Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | `valid` | 0 | 0 | 0 | 0 | **LEAK_FREE (PASSED)** |
| `train` | `test` | 0 | 0 | 0 | 0 | **LEAK_FREE (PASSED)** |
| `valid` | `test` | 0 | 0 | 0 | 0 | **LEAK_FREE (PASSED)** |
| `active_all` | `excluded` | 0 | 11 | 1 | 11 | **SEPARATED (PASSED)** |

## 2. Comparison with Original Roboflow Partition Leakage

In the original Roboflow partition (`1347` train, `128` valid, `64` test):
- **100 out of 245 `base_id`s (40.82%)** and **84 out of 657 perceptual duplicate groups (12.79%)** leaked across splits.
- `train` vs `valid`: 70 base_ids / 51 duplicate groups leaked.
- `train` vs `test`: 47 base_ids / 34 duplicate groups leaked.
- `valid` vs `test`: 19 base_ids / 5 duplicate groups leaked.
- Original test set had **0 `Linear`** and **0 `Segmental`** images.

## 3. Mathematical Impossibility Proof for Segmental (Pigeonhole Principle)

- In the clean single-label dataset ($N=1,429$), `Segmental` has only **8 images belonging to 2 `base_id` groups**: `70_jpg` (5 images) and `225_jpg` (3 images).
- A fundamental theorem of combinatorics (Dirichlet's Box / Pigeonhole Principle) states that partitioning $n=2$ atomic non-splittable objects into $k=3$ non-empty subsets is impossible ($2 < 3$).
- Under strict zero-leakage, at least one split MUST receive 0 groups of `Segmental`.
- Allocating `70_jpg` to Train and `225_jpg` to Valid ensures the model receives sufficient training examples while enabling validation monitoring. Fixed test receives 0 images, but **100% of the 8 images are evaluated out-of-fold in 5-fold cross-validation (`cv_fold`)**.
