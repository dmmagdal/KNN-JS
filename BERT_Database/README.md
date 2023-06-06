# BERT_Databse

Description: This is an example of KNN implementation that replicates the same efforts in [this repo](https://github.com/dmmagdal/BERT_Database).


### Notes

 - Unfortunately, there are not any modules like Spotify's ANNOY or Facebook/Meta's faiss that do the clustering, KNN implementation, and indexing for me. I've decided to implement my own index by holding all BERT embeddings in a tensorflow 2D tensor.
 - The `python/` folder contains a script (and everything needed for a docker environment) to download the wikitext dataset from huggingface hub and save the dataset to JSON files.