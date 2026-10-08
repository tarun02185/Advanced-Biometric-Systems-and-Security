# Assignment 2 — Face Verification on AgeDB

Face verification pipeline built on the AgeDB dataset, covering dataset preparation, face detection/alignment, embedding extraction, and biometric performance evaluation under noise.

## Contents

- `Assignment_2_AgeDB_Dataset_Preparation.ipynb` — Downloads AgeDB from Hugging Face and builds a 200-identity subset with at least 2 images per identity. Images at age 5+ are preferred (not a hard cutoff); for each identity, the pair with the largest available age gap is selected.
- `biometric-a-2-2.ipynb` — Remaining face verification pipeline (run on Kaggle), starting from the prepared 200-identity subset:
  1. Locate and extract the prepared ZIP
  2. Validate subjects, images, ages, age gaps and gender
  3. MTCNN face detection and alignment
  4. FaceNet embeddings
  5. Genuine and impostor verification pair generation
  6. Original-image evaluation
  7. Gaussian-noise evaluation
  8. Salt-and-pepper-noise evaluation
  9. Optional synthetic/GAN evaluation
  10. FMR, FNMR, ROC, AUC and EER computation
  11. Comparison plots
  12. Save all numerical results
- `AgeDB_Assignment2.zip` — Prepared 200-identity subset produced by the dataset preparation notebook, used as input to the verification pipeline notebook.

## Pipeline Overview

1. **Dataset preparation**: sample 200 identities from AgeDB, each with a genuine pair selected for maximum age gap.
2. **Preprocessing**: MTCNN detection and alignment on all selected images.
3. **Embeddings**: FaceNet used to generate feature embeddings per image.
4. **Pair generation**: genuine (same identity) and impostor (different identity) pairs built from embeddings.
5. **Evaluation**: verification performance measured on original images and under Gaussian and salt-and-pepper noise (plus optional synthetic/GAN images), reporting FMR, FNMR, ROC/AUC, and EER.
