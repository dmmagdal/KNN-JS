// BERT Database class.

import * as fs from 'fs';
import path from 'path';
import process from 'process';
import * as tf from '@tensorflow/tfjs';
import ort from 'onnxruntime-node';
import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';


interface functionMap extends Record<string, Function> {
  [key: string]: Function;
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
    this.vectorIndex = tf.tensor2d([], [0, 768]);

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
    // await inputPairs.forEach(async (pair: string[]) => {
    // // for (const pair: string[] in inputPairs) {
    for (let i=0; i < inputPairs.length; i++) {
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

  public clear(verbose=false): void {
    if (verbose) {
      console.debug('Clearing index and data.');
    }

    this.vectorIndex = tf.tensor2d([], [0, 768], 'float32');
    this.data = [];
  }

  public async get_knn(queryTexts: string[]): Promise<string[][][]> {
    // Assertion to verify the length of the string array is > 1.
    // if (queryTexts.length === 0) throw new Error(
    //   'get_knn() requires the number of query texts to be at least one'
    // );
    // Assertion to verify the length of the index is greater than 0.
    if (this.length() == 0) throw new Error(
      'Index is unpopulated. Please'
    );

    // K value. This is fixed based on RETRO model. Use 1 if the index
    // is too small.
    const k = this.length() > 1 ? 2 : 1; // 

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

  public load(pathDir: string): void {
    // Save paths.
    const indexPath = path.join(pathDir, 'index.json');
    const dataPath = path.join(pathDir, 'data.json');

    // Convert the index to a 2D array and save it.
    const index: number[][] = this.vectorIndex.arraySync();
    fs.writeFileSync(indexPath, JSON.stringify(dataPath));

    // Save the data.
    fs.writeFileSync(dataPath, JSON.stringify(this.data));
  }

  public save(pathDir: string): void {
    // Save paths.
    const indexPath = path.join(pathDir, 'index.json');
    const dataPath = path.join(pathDir, 'data.json');

    // Assert that the paths exist.
    if (fs.existsSync(indexPath)) throw new Error(
      'ERROR: Index load path ' + indexPath + ' does not exist.'
    );
    if (fs.existsSync(dataPath)) throw new Error(
      'ERROR: Data load path ' + dataPath + ' does not exist.'
    );

    // If there is already an existing index/data, this function will
    // overwrite it. Warn that is the case.
    if (this.length() > 0 && this.initalizedModel) {
      console.debug(
        'Loading from save path will override existing index and data.'
      );
    }

    // Convert the index to a 2D array and save it.
    const indexJSON = JSON.parse(fs.readFileSync(indexPath).toString());
    this.vectorIndex = tf.tensor2d(indexJSON.data, indexJSON.shape);

    // Load the data.
    this.data = JSON.parse(fs.readFileSync(dataPath).toString()).data;
  }
}


// --------------------------------------------------------------------
// Initialize database
// --------------------------------------------------------------------

// Model paths.
const model_ = path.join(process.cwd(), 'plain_bert.onnx'); // Exported BERT
// const model_pipe = path.join(process.cwd(), 'bert.onnx'); // Exported BERT pipeline

// Model variables.
const func = 'L2'; // may go back and have it set by default
const limit = 100; // same here regarding having a default

// Initialize inference session with ort.
// const db = new BERTDatabase(model_);
const db = new BERTDatabase(model_, func, limit);
await db.initializeIndex();

// Inputs.
const inputs = '\
 There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';
const entries: string[] = [
  "Hello there.", "I am the Senate.", 
  "I don't like sand.", "Lightsabers make a fine",
  "Lambda class shuttle", "This is the way.",
  "You are on the council,", "master yoda",
  "Help me obi wan kenobi", "It's not the jedi way.",
];
const values: string[] = [
  "General Kenobi!", "It's treason then.", 
  "It's coarse and rough.", "addition to my collection.",
  "is on approach from scarif", "This is the way.",
  "but we do not grant you the rank of master", "you survived",
  "you're my only hope.", "Dewit.",
];
const invalidEntries: string[] = [
  "Short for a storm trooper.", "You were the chosen one!",
  "I have no such weaknesses."
];


// --------------------------------------------------------------------
// Test add function.
// --------------------------------------------------------------------

console.log('Testing ADD function:');
console.log('='.repeat(72));

// Input data in a batch.
// neighbors = entries[:6]
// continuations = values[:6]
// pairs = list(zip(neighbors, continuations))
// db.add(pairs)
let batchZipInput: string[][] = [];
for (let i = 0; i < 6; i++) {
  batchZipInput.push([entries[i], values[i]]);
}
await db.add(batchZipInput);
console.log('Added 6 entries to database');
console.log('\tExpected length is 6');
console.log('\tFound length of', db.length());

// Input data one value at a time.
// neighbors = [entries[7]]
// continuations = [values[7]]
// pairs = list(zip(neighbors, continuations))
// db.add(pairs)
let singleZipInput: string[][] = [[entries[7], values[7]]];
await db.add(singleZipInput);
console.log('Added 1 entry to database');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Input data using a key that already exists.
// neighbors = entries[4:6]
// continuations = values[4:6]
// pairs = list(zip(neighbors, continuations))
// db.add(pairs)
batchZipInput = [];
for (let i = 4; i < 6; i++) {
  batchZipInput.push([entries[i], values[i]]);
}
await db.add(batchZipInput);
console.log('Added 2 existing entries to database');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

console.log('='.repeat(72));


// --------------------------------------------------------------------
// Test get function.
// --------------------------------------------------------------------

console.log('Testing GET function:');
console.log('='.repeat(72));

// Attempt to retrieve data with a valid key (batch retrieval).
// valid_response = db.get([entries[0], entries[3]])
// print(valid_response)
let queryInput: string[] = [entries[0], entries[3]];
let validResponse = db.get(queryInput);
console.log('Querying batch of 2 strings from the database');
for (let i=0; i < queryInput.length; i++) {
  console.log('\tQuery:', queryInput[i]);
  console.log('\tValues:', validResponse[i]);
}

// Attempt to retrieve data with a valid key (single retrieval).
// valid_response = db.get([entries[4]])
// print(valid_response)
validResponse = db.get([entries[4]]);
console.log('Querying a single string from the database');
console.log('\tQuery:', entries[4]);
console.log('\tValues:', validResponse[0]);

// Attempt to retrieve data with an invalid key (batch retrieval).
// invalid_response = db.get(invalid_entries)
// print(invalid_response)
let invalidResponse = db.get(invalidEntries);
console.log('Querying batch of invalid strings from the database');
for (let i=0; i < invalidEntries.length; i++) {
  console.log('\tQuery:', invalidEntries[i]);
  console.log('\tValues:', invalidResponse[i]);
}

// Attempt to retrieve data with an invalid key (single retrieval).
// invalid_response = db.get([invalid_entries[1]])
// print(invalid_response)
invalidResponse = db.get([invalidEntries[1]]);
console.log('Querying a single invalid string from the database');
console.log('\tQuery:', invalidEntries[1]);
console.log('\tValues:', invalidResponse[0]);

// Attempt to retrieve data with both invald and valid key (batch
// retrieval only).
// mixed_response = db.get([invalid_entries[0], entries[3]])
// print(mixed_response)
let mixedResponse = db.get([invalidEntries[0], entries[3]]);
console.log('Querying batch of valid and invalid strings from the database');
console.log('\tQuery:', invalidEntries[0]);
console.log('\tValues:', mixedResponse[0]);
console.log('\tQuery:', entries[3]);
console.log('\tValues:', mixedResponse[1]);

console.log('='.repeat(72));


// --------------------------------------------------------------------
// Test remove function.
// --------------------------------------------------------------------

console.log('Testing REMOVE function:');
console.log('='.repeat(72));

// Attempt to remove an entry with a valid key (batch removal).
// valid_response = db.remove(entries[3:5])

// Attempt to remove an entry with a valid key (single removal).
// valid_response = db.remove([entries[2]])

// Attempt to remove an entry with an invalid key (batch removal).
// invalid_response = db.remove(invalid_entries[:2])

// Attempt to remove an entry with an invalid key (single removal).
// invalid_response = db.remove([invalid_entries[0]])

// Attempt to remove an entry with both a valid and an invalid key
// (batch removal only).
// mixed_response = db.remove([invalid_entries[-1], entries[-1]])


// --------------------------------------------------------------------
// Test KNN function.
// --------------------------------------------------------------------

console.log('Testing GET_KNN function:');
console.log('='.repeat(72));

// Retrieve the K Nearest Neighbors from the database given the input
// text. KNN doesnt require the input text to be a part of the
// database.

// Start by clearing out the databse entirely (resets the index and
// data), then populating it with all (key, value) pairs. Be sure to 
// reset the contents of the index before populating it.


console.log('='.repeat(72));


// --------------------------------------------------------------------
// Test SAVE/LOAD function.
// --------------------------------------------------------------------

console.log('Testing SAVE & LOAD function:');
console.log('='.repeat(72));


console.log('='.repeat(72));
