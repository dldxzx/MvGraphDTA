# MvGraphDTA

MvGraphDTA: Multi-view-based graph deep model for drug-target affinity prediction by introducing the graphs and line graphs

## Requirements

[numpy](https://numpy.org/)==1.23.5

[pandas](https://pandas.pydata.org/)==1.5.2

[biopython](https://biopython.org/)==1.79

[scipy](https://scipy.org/)==1.9.3

[torch](https://pytorch.org/)==2.0.1

[torch_geometric]([PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/index.html))==2.3.1

## Example usage

### 1. Use our pre-trained model
In this section，we provide the core set data of pdbbindv2016 and Li's data(filtered casf2013 and casf2016), you can directly execute the following command to run our pre-trained model and get the results on the core set.

```bash
# Run the following command.
python test_pretrain.py
```

### 2. Run on your datasets

In this section, you must provide .sdf file of the drug as well as .pdb file of the target.

 ```bash
# You can get the graph and line graph of drug and target by running the following command.
python data_process.py

# When all the data is ready, you can train your own model by running the following command.
python training.py

 ```
