# Installations
```bash
conda create -n mdlm python=3.10
conda activate mdlm
pip install -e .
pip install flash_attn==2.7.4.post1
```


### Additional dependencies
```bash
conda install -c conda-forge openmm pdbfixer
conda install -c bioconda anarci
```


### Download training data
From the [OAS search tool](https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/), select `Species == human` and use the provided shell-script command to pull the data

Prepare the training data by running `preprocess.py`.


### Download index for sequence similarity search
The index is build for sequence similarity search using [KAsearch](https://github.com/oxpig/kasearch)
```bash
wget -qnc "https://zenodo.org/record/7562025/files/OAS-aligned-small.tar" -O small_OAS.tar
tar -xf small_OAS.tar
```
Note: The full OAS index (2.4B sequences) takes 68GB and around 30mins to search per entry. Stick with the small index (86M sequences) for now


# Structure Prediction
### Structure prediction with Boltz-2
[Boltz-2](https://github.com/jwohlwend/boltz/tree/main) is a SOTA open-sourced protein structure prediction model. Boltz currently uses an old version of flash-attention. It is easiest to make a separate environment for running boltz
```bash
conda create -n boltz python=3.10
conda activate boltz
pip install boltz[cuda]

# 1st run will install cache files. Point it towards the working directory instead of home
boltz predict test_protein.yaml --use_msa_server --cache ./boltz --out_dir ./output/boltz/
```

For subsequent runs, use this script
```

```


### [OLD] Structure prediction with ImmuneBuilder
* ImmuneBuilder: Run [visualization.ipynb](visualizer.ipynb) to visualize ImmuneBuilder predicted structure. 


# Other metrics
### Druggability
5 druggability metrics can be calculated with the [Therapeutic Antibody Profiler (TAP)](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap) using their web portal. Can't find an API 😢


