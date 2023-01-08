from datasets import load_dataset_builder

name = "fashion_mnist"

builder = load_dataset_builder(name)

print(builder.info.description)
