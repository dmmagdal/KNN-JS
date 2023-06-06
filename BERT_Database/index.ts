// BERT Database class.

import * as tf from '@tensorflow/tfjs';
import ort from 'onnxruntime-node';
import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';
import path from 'path';
import process from 'process';


interface functionMap {
  [key: string]: Function;
}


class BERTDatabase {
  private vectorIndex: tf.Tensor2D;
  private data: string[][];
  private bertPath: string;
  private distanceFunc: string;
  // private validDistances: object = {
  //   'L2': this.euclideanDistance,
  //   'Cosine': this.cosineDistance
  // };
  private validDistances: functionMap = {
    'L2': this.euclideanDistance,
    'Cosine': this.cosineDistance
  };
  private limit: number;
  private session: ort.InferenceSession;
  private tokenizer: PreTrainedTokenizer;
  public initalizedModel: boolean;

  constructor(bertPath: string, distanceFunc: string = 'L2', limit: number = 100) {
    // Initialize empty (2D) tensor for vector index.
    this.vectorIndex = tf.tensor2d([0, 0], [0, 768]);

    // Initialize empty list to contain the (key, entry) strings.
    this.data = [];

    // Assertion on the distance function. Only euclidean L2 and cosine
    // are supported at the moment.
    if (this.validDistances.keys().indexOf(distanceFunc) === -1) throw new Error(
      'Invalid distance function ' + distanceFunc + '. Valid ' + 
      'fuctions include: ' + this.validDistances.keys().toString()
    );

    // Set other class variables.
    this.bertPath = bertPath;         // path to bert model
    this.distanceFunc = distanceFunc; // distance function
    this.limit = limit;               // max number of embeddings for index
    this.initalizedModel = false;     // whether the BERT model was initialized/loaded
  }

  public async initializeIndex(): Promise<void> {
    // Load (pretrained) BERT tokenizer.
    this.tokenizer = await AutoTokenizer.from_pretrained(
      'Xenova/bert-base-uncased',
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

  private euclideanDistance(query: tf.Tensor2D): tf.Tensor1D {
    // Compute the euclidean distance from the query vector across each
    // element of the index.
    return tf.squaredDifference(this.vectorIndex, query);
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
    // Initialize a copy of the data, keeping only the key strings.
    let entries: string[] = [];
    this.data.forEach((pair: string[]) => {
      entries.push(pair[0]);
    });

    let retrievedPairs: string[][] = []
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
          console.debug('Entry for', key, 'was not found');
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
    inputPairs.forEach(async (pair: string []) => {
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
        if (this.length === this.max_length) {
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
        this.vectorIndex = tf.concat(
          [this.vectorIndex, embedding], 0
        );

        // One last assertion to make sure that the index and data
        // lengths are not out of sync.
        if (this.length() !== this.data.length) throw new Error(
          'Index length and data length should match: ' +
          this.length().toString() + ' (index) vs ' + 
          this.data.length.toString() + ' (data).'
        );
      }
    });
  }

  public async remove(keys: string[], verbose=false): Promise<void> {
    // Initialize a copy of the data, keeping only the key strings.
    let entries: string[] = [];
    this.data.forEach((pair: string[]) => {
      entries.push(pair[0]);
    });

    // Remove entries from data storage where key (entry) is found in
		// the data.
    keys.forEach(async (key: string) => {
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
            )
          ], 0
        );

        // Remove the (key, continuation) pair from the data.
        this.data.splice(entryIndex, 1);

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
    });
  }

  public async get_knn(queryTexts: string[]): Promise<string[][][]> {
    if (queryTexts.length === 0) throw new Error(
      'get_knn() requires the number of query texts to be at least one'
    );

    const k = this.length() > 1 ? 2 : 1; // This is fixed based on RETRO model.

    const returnedTexts: string[][][] = [];
    queryTexts.forEach(async (text: string) => {
      // Embed the text.
      const embedding = await this.embed(text);

      // Compute the distance.
      const distances = this.validDistances[this.distanceFunc](embedding);
    
      // Get indices and values of top k neighbors.
      const { values, indices } = tf.topk(distances, k, false);

      // Return the text (entry, continuation) pair values from the 
      // data given the nearest neighbors.
      const localTextPairs: string[][] = [];
      indices.forEach((idx: number) => {
        localTextPairs.push(this.data[idx]);
      });
      returnedTexts.push(localTextPairs);
    });

    return returnedTexts;
  }

  public load(): void {

  }

  public save(): void {

  }
}


// Inputs.
const inputs = '\
 There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';

// Model paths.
const model_ = path.join(process.cwd(), '..', 'plain_bert.onnx'); // Exported BERT
// const model_pipe = path.join(process.cwd(), '..', 'bert.onnx'); // Exported BERT pipeline

// Model variables.
const func = 'L2'; // may go back and have it set by default
const limit = 100; // same here regarding having a default

// Initialize inference session with ort.
// const db = new BERTDatabase(model_);
const db = new BERTDatabase(model_, func, limit);
await db.initializeIndex();
