=================
Sulphane geom.opt
=================

with Torch ANI

https://share.google/aimode/8kOk2JnPsy69gVXBi


(venv) milias@DESKTOP-7OTLCGO:~/work/projects/Molecular-and-Materials-Modeling-2026-May/tests/torchani-geomopt/sulphane/.python code02.py
/home/milias/work/software/venv/lib/python3.12/site-packages/torchani/csrc/__init__.py:56: UserWarning: The extensions: ['cuaev', 'mnp', 'cell_list'] are not installed and will not be available. To install the extensions first install the CUDA Toolkit, and afterwards  run `ani build-extensions` To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1 For example, if using bash, you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc
  warnings.warn(
/home/milias/work/projects/Molecular-and-Materials-Modeling-2026-May/tests/torchani-geomopt/sulphane/code02.py:12: FutureWarning: Please use atoms.calc = calc
  atoms.set_calculator(calculator)
------------------------------
Parameter       | ANI-2x     | Exp.       | Diff.
------------------------------
S-H1 (A)        | 1.3403     | 1.3356     | 0.0047
S-H2 (A)        | 1.3403     | 1.3356     | 0.0047
H-S-H (deg)     | 93.22      | 92.11      | 1.11
------------------------------

(venv) milias@DESKTOP-7OTLCGO:~/work/projects/Molecular-and-Materials-Modeling-2026-May/tests/torchani-geomopt/sulphane/.python code03_vibfreq.py
/home/milias/work/software/venv/lib/python3.12/site-packages/torchani/csrc/__init__.py:56: UserWarning: The extensions: ['cuaev', 'mnp', 'cell_list'] are not installed and will not be available. To install the extensions first install the CUDA Toolkit, and afterwards  run `ani build-extensions` To suppress warn set the env var TORCHANI_NO_WARN_EXTENSIONS=1 For example, if using bash, you may add `export TORCHANI_NO_WARN_EXTENSIONS=1` to your .bashrc
  warnings.warn(
---------------------------------------------------------------------------
Mode                 | ANI-2x       | Exp.       | Relat. Diff. (%)
---------------------------------------------------------------------------
Bending (v2)         | 1282.9       | 1183.0     | 8.44%
Symm. Stretch (v1)   | 2808.9       | 2615.0     | 7.42%
Asymm. Stretch (v3)  | 2844.6       | 2626.0     | 8.33%
---------------------------------------------------------------------------
