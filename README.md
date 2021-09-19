# Watusi - An Initial Investigation of Neural Decompilation for WebAssembly

### Generating a Dataset

A dataset of Wasm and C pairs can be generated using the `generate_parallel.sh` script:

```
./generate_parallel.sh <nb_parallel_tasks> <grammar_json> <nb_pairs> <output_file>
```

`grammar_json` can be any of the files provided in the `benchmarks/` folder or a new grammar following that format.

### Training the Model

To create a new trained instance of the model based on this dataset, run the `train.sh` script after specifying correct input values in `model/config.yaml`.

### Inferring translations

To test the decompilation using the new model instance, use the `translate.sh` script by specifying the input sequences and optionally, the target decompilations.