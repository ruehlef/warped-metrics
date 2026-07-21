# Warped Metrics on the Dwork Quintic

This repository contains the notebook-based pipeline used to study warped compactifications on the Dwork Quintic, including geometry checks, harmonic-form learning, and warp-factor reconstruction.

## Repository Structure

- `Euler_number.ipynb`  
  Computes the Euler number as a consistency check of the point generators and sampling strategies.

- `WP_metric.ipynb`  
  Computes the Weil-Petersson (WP) metric as a cross-check of both point generation and harmonic-form-related ingredients. We also illustrate the two-step training procedure as a simple way to reduce the error of the CY metric.

- `solve_pdes.ipynb`  
  Runs the full PDE pipeline:
  1. Learn the Calabi-Yau metric (Monge-Ampere)
  2. Learn harmonic `(2,1)` forms
  3. Learn the warp factor

- `warped_metrics.py`  
  Shared library module with the classes and helper functions used throughout the notebooks.

## Scientific Goal

The main objective is to provide an end-to-end numerical workflow for:

1. validating geometric data generation,
2. solving intermediate geometric learning problems, and
3. obtaining the warped solution via coupled PDE stages.

The three notebooks are designed to be used together, but can also be run independently for targeted checks.

## IPS Sampling Note

This project includes an IPS sampling implementation developed specifically for the Dwork Quintic.  
It will be integrated into `cymetric` soon, where it will work for general CICYs.

## Citation

If you use this code, please cite the associated paper (to appear on arXiv).

Temporary placeholder (to be updated once the arXiv identifier is available):

```bibtex
@article{YOURKEY2026,
  title   = {TITLE PLACEHOLDER},
  author  = {AUTHOR PLACEHOLDER},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Status

This repository is under active development. Interfaces and notebook details may evolve as components are upstreamed to `cymetric`.
