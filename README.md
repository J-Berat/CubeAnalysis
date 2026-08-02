# CubeAnalysis

Analyse automatique de cubes 3D FITS ou HDF5 avec Julia.

CubeAnalysis produit notamment des projections, histogrammes, diagrammes de
phase, spectres de puissance, fonctions de structure et diagnostics HRO.

## Installation

```bash
cd "/Users/jb270005/Desktop/CubeAnalysis"
git pull origin main
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Fichiers attendus

Chaque simulation RAMSES doit contenir :

```text
density.fits
temperature.fits
Bx.fits
By.fits
Bz.fits
Vx.fits
Vy.fits
Vz.fits
```

Le script RAMSES vérifie que les cubes font `256 x 256 x 256` et utilise une
boîte de `50 x 50 x 50 pc`.

## Analyser toutes les simulations

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl \
  "/Users/jb270005/Desktop/simu_RAMSES" \
  "/Users/jb270005/Desktop/simu_RAMSES/results/CubeAnalysis_final"
```

Pour tester seulement la première simulation, ajouter `1` :

```bash
julia --startup-file=no --project=. scripts/run_simu_ramses.jl \
  "/Users/jb270005/Desktop/simu_RAMSES" \
  "/Users/jb270005/Desktop/simu_RAMSES/results/CubeAnalysis_test" \
  1
```

Les simulations déjà terminées sont ignorées. Comme `overwrite=false`, utiliser
un nouveau dossier de sortie pour régénérer les figures après une modification.

## Analyser un seul cube

Modifier `input_dir` et `output_dir` dans `config/cube_analysis.toml`, puis :

```bash
julia --project=. bin/analyze_cube.jl config/cube_analysis.toml
```

## Sorties principales

- tranches et projections, dont la densité de colonne ;
- histogrammes groupés de `Bx, By, Bz` et `Vx, Vy, Vz` ;
- diagrammes de phase avec équilibre thermique et isothermes ;
- spectres scalaires, magnétiques, cinétiques, de vitesse et de vorticité ;
- séparation solénoïdale/compressive ;
- fréquence de Nyquist et pentes de Kolmogorov/Burgers ;
- HRO et paramètre d’alignement ;
- fonctions de structure, autocorrélations et diagnostics physiques.

Les figures sont sans titre et sans grille. Les textes, ticks, légendes et
colorbars utilisent LaTeX. Par défaut, seuls les graphiques sont sauvegardés.

Pour écrire aussi les CSV et TOML :

```toml
[run]
save_data = true
```

## Mémoire

Réglages recommandés pour les gros cubes :

```toml
[run]
field_by_field = true
memory_budget_gb = 12
sample_stride = 16
atomic_directory = true
```

`field_by_field=true` évite de conserver tous les champs en mémoire. Une valeur
`memory_budget_gb = 0` désactive la limite.

## HRO et Ibáñez

Deux figures différentes peuvent être produites :

- `alignment_*.png` : HRO générique configurable ;
- `xi_*.png` : convention du script `make_ibanez_alignment.jl`.

La figure `xi_*.png` utilise 40 bins en cosinus, avec
`|cos(phi)| >= 0.75` pour le parallèle et `|cos(phi)| <= 0.25` pour le
perpendiculaire. C’est cette figure qu’il faut comparer aux anciens résultats
Ibáñez.

## Tests

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```
