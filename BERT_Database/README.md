# BERT_Databse

Description: This is an example of KNN implementation that replicates the same efforts in [this repo](https://github.com/dmmagdal/BERT_Database).


### Notes

 - Unfortunately, there are not any modules like Spotify's ANNOY or Facebook/Meta's faiss that do the clustering, KNN implementation, and indexing for me. I've decided to implement my own index by holding all BERT embeddings in a tensorflowjs 2D tensor.
 - The `python/` folder contains a script (and everything needed for a docker environment) to download the wikitext dataset from huggingface hub and save the dataset to JSON files.
 - For the BERT model, the database expects a pretrained BERTModel exported from Huggingface/pytorch to ONNX (see [here](https://github.com/dmmagdal/MobileML/tree/main/BERT/Export-HF) for more information). Even though you can use a BERT model exported from a Huggingface pipeline, I made this with the other implementation


### TODO for v1:

[x] Save/load index & data
[x] Unit tests for database (verify basic functions work)
[ ] Full run involving loading texts from wikitext to the database


### References

 - [Wikitext dataset](https://huggingface.co/datasets/wikitext) on Huggingface datasets
 - [Dataset class](https://huggingface.co/docs/datasets/v2.7.1/en/package_reference/main_classes#datasets.Dataset) on Huggingface datasets
 - [Tokenizers documentation](https://huggingface.co/docs/transformers.js/api/tokenizers) from transformers-js on Huggingface
 - [TensorflowJS documentation](https://js.tensorflow.org/api/latest/)