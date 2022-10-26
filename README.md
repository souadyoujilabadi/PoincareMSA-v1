PoincaréMSA is a tool for protein family vizualisation. It takes as input a multiple sequence alignemnt (MSA) and builds its projection on a Poincaré disk using the method developed by Klimovskaia et al. in [1].

# About
PoincareMSA builds an interactive projection of an input protein multiple sequence alignemnt (MSA) using a method based on Poincaré maps described by Klimovskaia et al [1]. It reproduces both local proximities of protein sequences and hierarchy contained in give data. Thus, sequences located closer to the center of projection correspond to the proteins sharing the most general functional properites and/or appearing at the earlier stages of evolution.

# Colab version
`PoincareMSA_colab.ipynb` provides an interactive Google Colab notebook for projection of protein MSA starting from a file in `.mfasta` format. The user can also provide an annotation in `.csv` format which will be used for coloring.

# Version for local installation

To get a local copy of the software run:

```
git clone git@github.com:klanita/PoincareMSA.git
cd PoincareMSA
```

The program is implemented in `python3.7` using `pytorch` librarie for Poincaré disk construction and `plotly` for interactive visualisation of the resulting projections.

If you are working in Linux, you can use e a conda environment to access all the cenessary libraries :

```
conda env create -f env_poincare.yml
conda activate env_poincare
```
otherwise here is a list of necessary dependencies to install:

```
pytorch 1.7.1
sklearn 0.24.1
numpy 1.19.2
pandas 1.2.3
scipy 1.6.0
seaborn 0.11.1
plotly 5.8.0
```

## Python notebooks

The best way to try PoincaréMSA is by launching python notebooks with provided examples. To launch a particular example one needs to put the corresponding jupyter notebook to the repository root.

For example, to run PoincaréMSA on kinase dataset on should execute:

```
cp examples/kinases/PoincareMSA_kinases.ipynb ./
jupyter-notebook PoincareMSA_kinases.ipynb
```
The notebook can be then easily modified to work with any user-provided dataset.


Otherwise, the used can also launche the projection generation step by step as described below.

## Command line step-by-step version

Data preparation

The user is invited to provide its MSA in the classical `.mfasta` format. Each sequence of the alignment is translated to a profile using position-specific scoring matrix (PSSM) according to the pseudo-count algortihm of Henikoff & Henikoff. The related scripts are located in `scripts/prepare_data/` directory and driver scripts are provided for every example as `create_projection.sh`.

The resulting PSSM profiles representing each protein of MSA are stored in the directory `fastas0.9`, where 0.9 indicate the threshold percentage of gaps per position used to filter initial alignment. To build a Poincaré disk from this data, one needs to run a command from `scripts/build_poincare_map/` directory:

```
python main.py --input_path path_to_files/fastas0.9/ --output_path output_dir --knn 5 --gamma 2 --batchsize 4 --epochs 1000
```
which will create an output `.csv` file with protein coordinates in the final projection and `.png` images reflecting the learning process. The `.csv` file can be further used to build interactive visualisation.


# Examples of use
We provide several examples of PoincareMSA usage for different protien families in the `examples` folder. Each example comes with a bash script alloqing to reproduce the results starting from MSA and labels contained in `data`.

# References
The projection construction is adapted from the original code: https://github.com/facebookresearch/PoincareMaps

[1] Klimovskaia, A., Lopez-Paz, D., Bottou, L. et al. Poincaré maps for analyzing complex hierarchies in single-cell data. Nat Commun 11, 2966 (2020). https://doi.org/10.1038/s41467-020-16822-4

