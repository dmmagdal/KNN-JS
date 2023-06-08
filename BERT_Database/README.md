# BERT_Databse

Description: This is an example of KNN implementation that replicates the same efforts in [this repo](https://github.com/dmmagdal/BERT_Database).


### Notes

 - Unfortunately, there are not any modules like Spotify's ANNOY or Facebook/Meta's faiss that do the clustering, KNN implementation, and indexing for me. I've decided to implement my own index by holding all BERT embeddings in a tensorflowjs 2D tensor.
 - The `python/` folder contains a script (and everything needed for a docker environment) to download the wikitext dataset from huggingface hub and save the dataset to JSON files.
 - For the BERT model, the database expects a pretrained BERTModel exported from Huggingface/pytorch to ONNX (see [here](https://github.com/dmmagdal/MobileML/tree/main/BERT/Export-HF) for more information). Even though you can use a BERT model exported from a Huggingface pipeline, I made this with the other implementation
 - When testing loading the WikiText to the BERT database, there is a caveat regarding file sizes that can be read into NodeJS. NodeJS has a set maximum for the length/size of a string that can be created, which is ~1GB for 64 bit systems and ~512MB for 32 bit systems. For reference, the problem file (`wikitext-103-v1_train.json`) is 538MB in size.
     - This issue of limited string sizes extends to writing large amounts of data to a string. When using JSON.stringify() on the data object, there is a maximum string size that NodeJS JSON.stringify() is able to create. It does not take much to reach that limit. As such saving/loading data is done in chunks. The index and data are sliced into chunks of equal size and stored to their respective JSON files, each marked with their own number.
 - Steps to run
     1. Download wikitext dataset with the script in `python/`. Build the dockerfile and run it or simply type `python download_wikitext.py` in the `python/` folder.
     2. Download and export the pretrained BERT model. Refer to the export function [here](https://github.com/dmmagdal/MobileML/tree/main/BERT/Export-HF) in the MobileML repository. Placed the export `plain_bert.onnx` file in this folder.
     3. Run the main program by issuing the following command: `npm install; npm run build; node dist/index.js`.
 - A full run of the program takes around 45 minutes to complete.


### TODO for v1:

[x] Save/load index & data
[x] Unit tests for database (verify basic functions work)
[x] Full run involving loading texts from wikitext to the database


### TODO for next time:

[ ] More efficient algorithms
     [ ] Reduce memory overhead for all functions
         * Reduce number of variables used/initialized
         * Reduce number of copies that are made for the sake of convenience
     [ ] Better computational efficiency/runtime for all functions
         * Better way to get k shortest distances
 [ ] Leverage GPU for some operations (optional given other applications or projects that use this may be using the device GPU for something more important)
     * Compute BERT embeddings in ONNX
     * tensorflowjs operations


### References

 - [Wikitext dataset](https://huggingface.co/datasets/wikitext) on Huggingface datasets
 - [Dataset class](https://huggingface.co/docs/datasets/v2.7.1/en/package_reference/main_classes#datasets.Dataset) on Huggingface datasets
 - [Tokenizers documentation](https://huggingface.co/docs/transformers.js/api/tokenizers) from transformers-js on Huggingface
 - [TensorflowJS documentation](https://js.tensorflow.org/api/latest/)
 - tensorflowJS [npm package](https://www.npmjs.com/package/@tensorflow/tfjs)
 - transformersJS [npm package](https://www.npmjs.com/package/@xenova/transformers)
 - onnxruntime-node [npm package](https://www.npmjs.com/package/onnxruntime-node?activeTab=readme)
 - cli-progress [npm package](https://www.npmjs.com/package/cli-progress)