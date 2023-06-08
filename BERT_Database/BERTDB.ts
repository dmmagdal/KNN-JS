// BERT Database class.

import * as fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs';
import ort from 'onnxruntime-node';
import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';


interface functionMap extends Record<string, Function> {
  [key: string]: Function;
}

interface valueIndicesMap {
  values: number[];
  indices: number[];
}


export class BERTDatabase {
  private vectorIndex: tf.Tensor2D;
  private data: string[][];
  private bertPath: string;
  private distanceFunc: string;
  private validDistances: functionMap = {
    'L2': this.euclideanDistance,
    'Cosine': this.cosineDistance,
  };
  private limit: number;
  private indexChunkSize: number = 5_000;
  private dataChunkSize: number = 5_000;
  private session: ort.InferenceSession;
  private tokenizer: PreTrainedTokenizer;
  public initalizedModel: boolean;

  constructor(bertPath: string, distanceFunc: string = 'L2', limit: number = 100) {
    // Assert bertPath exists.
    if (!fs.existsSync(bertPath)) throw new Error(
      'ERROR: BERT model path ' + bertPath + ' not found.'
    );
    
    // Initialize empty (2D) tensor for vector index. Pass in empty
    // array because no valies are initialized yet.
    this.vectorIndex = tf.tensor2d([], [0, 768], 'float32');

    // Initialize empty list to contain the (key, entry) strings.
    this.data = [];

    // Assertion on the distance function. Only euclidean L2 and cosine
    // are supported at the moment.
    const validKeys = Object.keys(this.validDistances);
    if (validKeys.indexOf(distanceFunc) === -1) throw new Error(
      'Invalid distance function ' + distanceFunc + '. Valid ' + 
      'fuctions include: ' + validKeys.toString()
    );

    // Set other class variables.
    this.bertPath = bertPath;         // path to bert model
    this.distanceFunc = distanceFunc; // distance function
    this.limit = limit;               // max number of embeddings for index
    this.initalizedModel = false;     // whether the BERT model was initialized/loaded
  }

  public async initializeIndex(): Promise<void> {
    // Load (pretrained) BERT tokenizer.
    const cacheDir: string = './bert_tokenizer_local';
    const local_: string = path.join(
      cacheDir, 'Xenova', 'tokenizer.json'
    );
    const localConfig: string = path.join(
      cacheDir, 'Xenova', 'tokenizer_config.json'
    );
    const loadLocal: boolean = [
      fs.existsSync(cacheDir), fs.existsSync(local_), 
      fs.existsSync(localConfig),
    ].every(v => v === true);
    this.tokenizer = await AutoTokenizer.from_pretrained(
      'Xenova/bert-base-uncased', 
      {
        cache_dir: cacheDir,
        local_files_only: loadLocal,
      }
    );

    // Initalize ORT inference session and load the ONNX BERT model to
    // that session object. Note that constructors can NEVER be async,
    // so this function must be called after the class is initialized.
    this.session = await ort.InferenceSession.create(
      this.bertPath, 
      {
        // executionProviders: ['wasm'], // can also specify 'webgl'
        // executionProviders: ['webgl'], // can also specify 'wasm'
        graphOptimizationLevel: 'all',
      }
    );

    this.initalizedModel = true;
  }

  private async embed(text: string): Promise<tf.Tensor2D> {
    // Tokenize the text.
    const inputs = this.tokenizer(text);

    // Convert the tokenized inputs to ORT tensors.
    const tensorInputs = {
      'input_ids': new ort.Tensor(
        'int64',
        inputs.input_ids.data,
        inputs.input_ids.dims,
      ),
      'attention_mask': new ort.Tensor(
        'int64',
        inputs.attention_mask.data,
        inputs.attention_mask.dims,
      ),
      'token_type_ids': new ort.Tensor(
        'int64',
        inputs.token_type_ids.data,
        inputs.token_type_ids.dims,
      ),
    };

    // Return the pooled output from the BERT model.
    const bertOutputs = await this.session.run(
      {
        'input_ids': tensorInputs.input_ids,
        'attention_mask': tensorInputs.attention_mask,
        'token_type_ids': tensorInputs.token_type_ids,
      }
    );
    const pooledOutput = bertOutputs['1740'];

    // Copy over the pooled output from the ort.Tensor to an array.
    // This intermediate step is necessary because tensorflowjs wont
    // play nice with the ort.Tensor.data despite it being a
    // Float32Array.
    const dataArray = new Array(pooledOutput.data.length);
    for (let i = 0; i < pooledOutput.data.length; i++) {
      dataArray[i] = pooledOutput.data[i];
    }
    
    // Return embedding now saved in a tensorflowjs 2D tensor.
    return tf.tensor2d(dataArray, [1, 768], 'float32');
  }

  private findIndex(query: tf.Tensor2D): number {
    // Perform element-wise equality comparison between the index and
    // the query vector.
    const equalComparison = tf.equal(this.vectorIndex, query);

    // Check if query vector exists in the index along the 0 axis.
    const alongAxis0 = tf.cast(
      tf.all(equalComparison, 1), 'bool'
    ).dataSync();

    // Find the index where query exists in the index.
    const index = alongAxis0.findIndex((value) => value === 1);

    // Return the index or -1 if B does not exist in A
    return index !== -1 ? index : -1;
  }

  private getKLowest(distances: tf.Tensor1D, k: number): valueIndicesMap {
    // Create a copy of the distance data.
    const distanceCopy: number[] = distances.arraySync();

    // Iterate through array of distances, picking the k smallest
    // values and appending them to number arrays.
    const indices: number[] = [];
    const values: number[] = [];
    for (let i = 0; i < k; i++) {
      // Get lowest value from the distances array and its index. Use
      // spread syntax (...arrayVariable) to pass in array.
      const value: number = Math.min(...distanceCopy); 
      const index: number = distanceCopy.indexOf(value);
      values.push(value);
      indices.push(index);

      // Remove value from copy.
      distanceCopy.splice(index, 1);
    }

    const returnValues: valueIndicesMap = {
      values: values, indices: indices
    };
    return returnValues;
  }

  private euclideanDistance(index: tf.Tensor2D, query: tf.Tensor2D): tf.Tensor1D {
    // Compute the euclidean distance from the query vector across each
    // element of the index.
    return tf.sum(tf.squaredDifference(index, query), 1);
  }

  private cosineDistance(query: tf.Tensor2D): tf.Tensor1D {
    // Compute the dot product between the two tensors.
    const dotProduct = tf.matMul(this.vectorIndex, query.transpose());

    // Compute the norms of the tensors.
    const normA = tf.norm(this.vectorIndex, 1);
    const normB = tf.norm(query, 1);

    // Compute the cosine similarity.
    const cosineSimilarity = tf.div(dotProduct, tf.mul(normA, normB).clipByValue(1e-8, Infinity));

    // Compute the cosine distance by subtracting the similarity from 1.
    const distance = tf.sub(1, cosineSimilarity);

    return distance.squeeze();
  }

  public modelPath(): string {
    return this.bertPath;
  }

  public length(): number {
    return this.vectorIndex.shape[0];
  }

  public max_length(): number {
    return this.limit;
  }

  public get(keys: string[], verbose=false): string[][] {
    // If the index is empty, simply return now.
    if (this.length() === 0) {
      // Assertion to make sure that the index and data lengths are not
      // out of sync.
      if (this.length() !== this.data.length) throw new Error(
        'Index length and data length should match: ' +
        this.length().toString() + ' (index) vs ' + 
        this.data.length.toString() + ' (data).'
      );

      if (verbose) {
        console.debug('Index is empty. Results returned are blank.');
      }

      const retrievedPairs: string[][] = [];
      keys.forEach(() => {
        retrievedPairs.push(['', '']);
      });
      return retrievedPairs;
    }

    // Initialize a copy of the data, keeping only the key strings.
    let entries: string[] = [];
    this.data.forEach((pair: string[]) => {
      entries.push(pair[0]);
    });

    const retrievedPairs: string[][] = [];
    keys.forEach((key: string) => {
      // Acquire the index of the key from the copy of the data.
      const index = entries.indexOf(key);
      if (index !== -1) {
        // Push the (key, value) string from the data onto the return
        // array.
        retrievedPairs.push(this.data[index]);
      }
      else {
        // Output that the target key could not be found in the data if
        // verbose is true.
        if (verbose) {
          console.debug('Entry for', key, 'was not found.');
        }

        // Append a list of two empty strings (which should not be a
        // valid entry into the database). This still keeps things in
        // order with respect to the original indexes of the query.
        // retrievedPairs.push(undefined);
        retrievedPairs.push(['', '']);
      }
    });

    // Return the list of (entry, continuation) pairs.
    return retrievedPairs;
  }

  public async add(inputPairs: string[][], verbose=false): Promise<void> {
    // BERT model must be initialized and loaded before proceeding.
    if (!this.initalizedModel) throw new Error(
      'Index model not initialized. Call initializeIndex() to load ' +
      'the model required to embed the text for the database.'
    );
    
    // Initialize a copy of the data, keeping only the key strings.
    let entries: string[] = [];
    this.data.forEach((pair: string[]) => {
      entries.push(pair[0]);
    });
    
    // Add each input pair to the database.
    for (let i = 0; i < inputPairs.length; i++) {
      const pair = inputPairs[i];

      // Assert that all pair string arrays are strictly of length 2.
      if (pair.length !== 2) throw new Error(
        'Input pairs to index contained an invalid entry: Must have ' +
        'each entry with only two strings!'
      );

      // Extract the key string from the (key, continuation) pair
      // string array.
      const key = pair[0];

      const entryIndex = entries.indexOf(key);
      if (entryIndex !== -1) {
        // Update the continuation in the (entry, continuation) from
        // the data. No need to update embeddings in the index because
        // the entry is still the same.
        this.data[entryIndex] = pair;

        // Output that the key was updated in the data if verbose is
        // true.
        if (verbose) {
          console.debug('Updated', key, 'entry to be', pair);
        }
      }
      else {
        // Verify that the index is not full.
        if (this.length() === this.max_length()) {
          // Output that the index is already full if verbose is true.
          if (verbose) {
            console.debug('Index is full. Cannot add', pair, 'to index.');
          }
          
          // This return should be functionally the same as 'continue'
          // in this forEach loop.
          return;
        }
        // Add the new (entry, continuation) pair to the data.
        this.data.push(pair);

        // Embed the key with BERT.
        const embedding = await this.embed(key);

        // Concatentate/append the output tensor to the index.
        if (this.length() !== 0) {
          this.vectorIndex = tf.concat(
            [this.vectorIndex, embedding], 0
          );
        }
        else {
          this.vectorIndex = embedding;
        }

        // One last assertion to make sure that the index and data
        // lengths are not out of sync.
        if (this.length() !== this.data.length) throw new Error(
          'Index length and data length should match: ' +
          this.length().toString() + ' (index) vs ' + 
          this.data.length.toString() + ' (data).'
        );
      }
    };
  }

  public async remove(keys: string[], verbose=false): Promise<void> {
    // If the index is empty, simply return now.
    if (this.length() === 0) {
      // Assertion to make sure that the index and data lengths are not
      // out of sync.
      if (this.length() !== this.data.length) throw new Error(
        'Index length and data length should match: ' +
        this.length().toString() + ' (index) vs ' + 
        this.data.length.toString() + ' (data).'
      );

      return;
    }

    // Initialize a copy of the data, keeping only the key strings.
    let entries: string[] = [];
    this.data.forEach((pair: string[]) => {
      entries.push(pair[0]);
    });

    // Remove entries from data storage where key (entry) is found in
    // the data.
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      const entryIndex = entries.indexOf(key);
      if (entryIndex !== -1) {
        // Get the BERT embedding for the key string.
        const embedding = await this.embed(key);

        // Retrieve the index of the embeding from the index.
        const embeddingIndex = this.findIndex(embedding);

        // Assertion to verify that the index of the key in the data
        // matches the index of the embedding in the index.
        if (embeddingIndex !== entryIndex) throw new Error(
          'ERROR: Index and stored data may be out of sync. Entry ' +
          'of ' + key + ' was not found in the same place: ' +
          embeddingIndex.toString() + ' (index) vs ' + 
          entryIndex.toString() + ' (data).'
        );

        // Remove the embedding from the index.
        this.vectorIndex = tf.concat(
          [
            // this.vectorIndex.slice([0, 0], [embeddingIndex, -1]),
            // this.vectorIndex.slice([embeddingIndex + 1, 0], [-1, -1])
            this.vectorIndex.slice([0, 0], [embeddingIndex, -1]),
            this.vectorIndex.slice(
              [embeddingIndex + 1, 0], 
              [this.length() - embeddingIndex - 1, -1]
            ),
          ], 
          0 // axis
        );

        // Remove the (key, continuation) pair from the data.
        this.data.splice(entryIndex, 1);
        entries.splice(entryIndex, 1); // update entries too!

        // One last assertion to make sure that the index and data
        // lengths are not out of sync.
        if (this.length() !== this.data.length) throw new Error(
          'Index length and data length should match: ' +
          this.length().toString() + ' (index) vs ' + 
          this.data.length.toString() + ' (data).'
        );
      }
      else if (verbose) {
        // Output that the target key could not be found in the data if
        // verbose is true.
        console.debug('Entry', key, 'was not found in database.');
      }
    };
  }

  public clear(verbose=false): void {
    if (verbose) {
      console.debug('Clearing index and data.');
    }

    this.vectorIndex = tf.tensor2d([], [0, 768], 'float32');
    this.data = [];

    // One last assertion to make sure that the index and data lengths
    // are not out of sync.
    if (this.length() !== this.data.length) throw new Error(
      'Index length and data length should match: ' +
      this.length().toString() + ' (index) vs ' + 
      this.data.length.toString() + ' (data).'
    );
  }

  public async get_knn(queryTexts: string[]): Promise<string[][][]> {
    // Assertion to verify the length of the index is greater than 0.
    if (this.length() == 0) throw new Error(
      'Index is unpopulated. Please populate index before calling get_knn().'
    );

    // K value. This is fixed based on RETRO model. Use 1 if the index
    // is too small.
    const k = this.length() > 1 ? 2 : 1;

    const returnedTexts: string[][][] = [];
    for (let i = 0; i < queryTexts.length; i++) {
      const text = queryTexts[i];

      // Embed the text.
      const embedding = await this.embed(text);

      // Compute the distance.
      const distances: tf.Tensor1D = this.validDistances[this.distanceFunc](
        this.vectorIndex, embedding
      );

      // Get indices and values of top k neighbors.
      const valuesIndices: valueIndicesMap = this.getKLowest(distances, k);

      // Return the text (entry, continuation) pair values from the 
      // data given the nearest neighbors.
      const localTextPairs: string[][] = [];
      valuesIndices.indices.forEach((idx: number) => {
        localTextPairs.push(this.data[idx]);
      });
      returnedTexts.push(localTextPairs);
    };

    return returnedTexts;
  }

  public save(pathDir: string): void {
    // Save paths.
    // const indexPath = path.join(pathDir, 'index.json');
    // const dataPath = path.join(pathDir, 'data.json');
    if (!fs.existsSync(pathDir) || fs.statSync(pathDir).isFile) {
      fs.mkdirSync(pathDir, {recursive: true});
    }

    // Convert the index to a 2D array and save it. Note that
    // JSON.stringify() has a limit on the size of the string it can
    // create, so the data will have to be saved to multiple files in
    // chunks.
    const index: number[][] = this.vectorIndex.arraySync();
    let indexChunkCounter: number = 1;
    for (let i = 0; i < this.length(); i += this.indexChunkSize) {
      // Extract slice and save it to file.
      const indexSlice: number[][] = index.slice(i, i + this.indexChunkSize);

      const indexString = 'index_' + indexChunkCounter + '.json';
      const indexPath = path.join(pathDir, indexString);
      fs.writeFileSync(indexPath, JSON.stringify(indexSlice));
      indexChunkCounter++;
    }

    // Save the data. Will also need to chunk the data.
    let dataChunkCounter: number = 1;
    for (let j = 0; j < this.length(); j += this.dataChunkSize) {
      // Extract slice and save it to file.
      const dataSlice = this.data.slice(j, j + this.indexChunkSize);

      const dataString = 'data_' + dataChunkCounter + '.json';
      const dataPath = path.join(pathDir, dataString);
      fs.writeFileSync(dataPath, JSON.stringify(dataSlice));
      dataChunkCounter++;
    }
  }

  public load(pathDir: string): void {
    // Warning about loading databases.
    console.debug(
      'WARNING: Loading a database may result in different behavior ' +
      'the loaded model, distance function, or max limit differ, ' +
      'even if the data and index are exactly the same.'
    );

    // Save paths. While these may not be all the possible save paths,
    // at the bare minimum, at least one JSON for the index and data
    // should exist.
    const indexPath = path.join(pathDir, 'index_1.json');
    const dataPath = path.join(pathDir, 'data_1.json');

    // Assert that the initial index and json save paths exist.
    if (!fs.existsSync(indexPath)) throw new Error(
      'ERROR: Index load path ' + indexPath + ' does not exist.'
    );
    if (!fs.existsSync(dataPath)) throw new Error(
      'ERROR: Data load path ' + dataPath + ' does not exist.'
    );

    // If there is already an existing index/data, this function will
    // overwrite it. Warn that is the case.
    if (this.length() > 0 && this.initalizedModel) {
      console.debug(
        'Loading from save path will override existing index and data.'
      );
    }

    // Index all the index .json files in the pathDir folder.
    const indexFolder: string[] = fs.readdirSync(pathDir);
    const indexFiles: string[] = indexFolder.filter(
      (file: string) => file.includes('index_')
    );

    // Sort the index .json files in numerical order.
    const sortedIndexFiles: string[] = indexFiles.sort((a: string, b: string) => {
      const regex = /index_(\d+)\.json/;
      const [, numA] = a.match(regex)!;
      const [, numB] = b.match(regex)!;
      return parseInt(numA, 10) - parseInt(numB, 10);
    });

    // Iterate through each index .json file in order. Load the local
    // slices to number[][] array.
    const index: number[][] = [];
    for (let i = 0; i < sortedIndexFiles.length; i++) {
      const indexPath = path.join(pathDir, sortedIndexFiles[i]);
      index.push(...JSON.parse(fs.readFileSync(indexPath).toString()));
    }

    // Convert the number[][] array to a tensorflowjs tf.tensor2d OR
    // initialize a new (empty) tensorflowjs tf.tensor2d if the length
    // of the number[][] read in is 0.
    if (index.length > 0) {
      this.vectorIndex = tf.tensor2d(
        index, [index.length, 768]
      );
    }
    else {
      this.vectorIndex = tf.tensor2d([], [0, 768], 'float32');
    }

    // Index all the data .json files in the pathDir folder.
    const dataFolder: string[] = fs.readdirSync(pathDir);
    const dataFiles: string[] = dataFolder.filter(
      (file: string) => file.includes('data_')
    );

    // Sort the data .json files in numerical order.
    const sortedDataFiles: string[] = dataFiles.sort((a, b) => {
      const regex = /data_(\d+)\.json/;
      const [, numA] = a.match(regex)!;
      const [, numB] = b.match(regex)!;
      return parseInt(numA, 10) - parseInt(numB, 10);
    });

    // Iterate through each data .json file in order. Load the local
    // slices to string[][] array. Set the data to the read in
    // string[][] array.
    const data: string[][] = [];
    for (let i = 0; i < sortedDataFiles.length; i++) {
      const dataPath = path.join(pathDir, sortedDataFiles[i]);
      data.push(...JSON.parse(fs.readFileSync(dataPath).toString()));
    }
    this.data = data;

    // Assert that the index and data lengths are equal.
    if (this.length() !== this.data.length) throw new Error(
      'Index length and data length should match: ' +
      this.length().toString() + ' (index) vs ' + 
      this.data.length.toString() + ' (data).'
    );
  }
}