// Main program.

import * as fs from 'fs';
import path from 'path';
import process from 'process';
import { AutoTokenizer, PreTrainedTokenizer } from '@xenova/transformers';
import { SingleBar, Presets } from 'cli-progress';
import { BERTDatabase } from './BERTDB.js';
import { processText } from './datasetHelpers.js';


// ====================================================================
// UNIT TESTS
// ====================================================================

console.log('UNIT TESTS:');

// --------------------------------------------------------------------
// Initialize database
// --------------------------------------------------------------------

// Model paths.
const model_ = path.join(process.cwd(), 'plain_bert.onnx'); // Exported BERT
// const model_pipe = path.join(process.cwd(), 'bert.onnx'); // Exported BERT pipeline

// Model variables.
const func = 'L2';
const limit = 100;

// Initialize inference session with ort.
// const db = new BERTDatabase(model_);
let db = new BERTDatabase(model_, func, limit);
await db.initializeIndex();

// Inputs.
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
let batchZipInput: string[][] = [];
for (let i = 0; i < 6; i++) {
  batchZipInput.push([entries[i], values[i]]);
}
await db.add(batchZipInput);
console.log('Added 6 entries to database');
console.log('\tExpected length is 6');
console.log('\tFound length of', db.length());

// Input data one value at a time.
await db.add([[entries[7], values[7]]]);
console.log('Added 1 entry to database');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Input data using a key that already exists.
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
let queryInput: string[] = [entries[0], entries[3]];
let validResponse = db.get(queryInput);
console.log('Querying batch of 2 strings from the database');
for (let i=0; i < queryInput.length; i++) {
  console.log('\tQuery:', queryInput[i]);
  console.log('\tValues:', validResponse[i]);
}

// Attempt to retrieve data with a valid key (single retrieval).
validResponse = db.get([entries[4]]);
console.log('Querying a single string from the database');
console.log('\tQuery:', entries[4]);
console.log('\tValues:', validResponse[0]);

// Attempt to retrieve data with an invalid key (batch retrieval).
let invalidResponse = db.get(invalidEntries);
console.log('Querying batch of invalid strings from the database');
for (let i=0; i < invalidEntries.length; i++) {
  console.log('\tQuery:', invalidEntries[i]);
  console.log('\tValues:', invalidResponse[i]);
}

// Attempt to retrieve data with an invalid key (single retrieval).
invalidResponse = db.get([invalidEntries[1]]);
console.log('Querying a single invalid string from the database');
console.log('\tQuery:', invalidEntries[1]);
console.log('\tValues:', invalidResponse[0]);

// Attempt to retrieve data with both invald and valid key (batch
// retrieval only).
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

// Add all values to the index.
batchZipInput = [];
for (let i = 0; i < entries.length; i++) {
  batchZipInput.push([entries[i], values[i]]);
}
await db.add(batchZipInput); // Sets dataset length to 10.

// Attempt to remove an entry with a valid key (batch removal).
await db.remove(entries.slice(3, 5));
console.log('Removed 2 entries from database');
console.log('\tExpected length is 8');
console.log('\tFound length of', db.length());


// Attempt to remove an entry with a valid key (single removal).
await db.remove([entries[2]]);
console.log('Removed 1 entry from database');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with an invalid key (batch removal).
await db.remove(invalidEntries.slice(0, 2));
console.log('Removed 2 invalid entries from database (no change to data)');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with an invalid key (single removal).
await db.remove([invalidEntries[0]]);
console.log('Removed 1 invalid entry from database (no change to data)');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with both a valid and an invalid key
// (batch removal only).
await db.remove([invalidEntries.at(-1), entries.at(-1)]);
console.log('Removed 2 entries (invalid and valid) from database (some change to data)');
console.log('\tExpected length is 6');
console.log('\tFound length of', db.length());


// --------------------------------------------------------------------
// Test KNN function.
// --------------------------------------------------------------------

console.log('Testing GET_KNN function:');
console.log('='.repeat(72));

// Retrieve the K Nearest Neighbors from the database given the input
// text. KNN doesnt require the input text to be a part of the
// database.

// Start by clearing out the database entirely (resets the index and
// data), then populating it with all (key, value) pairs.
await db.clear();
await db.add(batchZipInput);

// Get KNN entries from batch.
let query1: string[] = [
  "I don't like sand.", 
  "I am a jedi, like my father before me."
];
const knnResults1 = await db.get_knn(query1);
console.log('Querying KNN from batch.')
for (let i = 0; i < query1.length; i++) {
  console.log('\tQuery:', query1[i]);
  console.log('\tResults:', knnResults1[i]);
}

// Get KNN entries from a single sample.
let query2: string[] = ["The senate will decide your fate."];
const knnResults2 = await db.get_knn(query1);
console.log('Querying KNN from single source.')
console.log('\tQuery:', query2[0]);
console.log('\tResults:', knnResults2[0]);

console.log('='.repeat(72));


// --------------------------------------------------------------------
// Test SAVE/LOAD function.
// --------------------------------------------------------------------

console.log('Testing SAVE & LOAD function:');
console.log('='.repeat(72));

// Test database save function.
db.save("./BERT_DB");
const fileStatus: boolean[] = [
  fs.existsSync('./BERT_DB/data.json'),
  fs.existsSync('./BERT_DB/index.json'),
];
const filesExist: boolean = fileStatus.every((value) => value === true);
console.log('Current database successfully saved:', filesExist);

// Test database load function.
const db_copy = new BERTDatabase(model_);
await db_copy.initializeIndex();
db_copy.load("./BERT_DB");
console.log('Database successfully loaded.')

// Given that I privatized the data variables, there is no way for me
// to do a good comparison between the loaded database and the existing
// one. The two databases may even have different behaviors depending
// on the model loaded, distance function, and max limit passed in
// (even if the data and index are the same).
console.log(
  'Loaded database and current database lengths match',
  db.length() === db_copy.length()
);

console.log('='.repeat(72));


// ====================================================================
// LOAD WIKITEXT
// ====================================================================

console.log('TESTING LOADING WIKITEXT:');

// Reinitialize database with practical limit.
db = new BERTDatabase(model_, 'L2', 250_000);
db.initializeIndex();

// Initialize GPT-2 tokenizer.
const cacheDir: string = './gpt2_tokenizer_local';
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
const gpt2Tokenizer: PreTrainedTokenizer = await AutoTokenizer.from_pretrained(
  'Xenova/gpt2', 
  {
    cache_dir: cacheDir,
    local_files_only: loadLocal,
  }
);

// Index all the dataset .json files in the python/ folder.
const datasetFolder: string[] = fs.readdirSync('./python');
const datasetFiles: string[] = datasetFolder.filter(
  (file: string) => file.endsWith('.json')
);

// Iterate through each dataset.
console.log('Processing wikitext dataset files...');
const chunkedDatasets: string[][] = [];
datasetFiles.forEach((file: string) => {
  // Skip 'wikitext-103_train.json' file because its size will cause
  // errors for NodeJS.
  if (file.includes('wikitext-103-v1_train')) {
    return;
  }
  console.log('\tProcessing', file, '...');

  // Load the data from the file.
  const fullPath = path.join('./python', file);
  const data: string[] = JSON.parse(
    fs.readFileSync(fullPath, {encoding: 'utf-8'}).toString()
  )['text'];

  // Process the dataset file text so that an array is returned .
  const textChunks: string[][] = processText(data, gpt2Tokenizer);
  chunkedDatasets.push(...textChunks); // Use spread to concatentate the arrays
});
console.log('Processed all wikitext dataset files.');

// Initialize a cli progressbar. Since this next step is going to take
// a while, it would be good to do so.
const progressBar = new SingleBar({
  format: 'Progress | {bar} | {percentage}% | ETA: {eta}s | {value}/{total}',
  barCompleteChar: '\u2588',
  barIncompleteChar: '\u2591',
  hideCursor: true,
}, Presets.shades_classic);

// Start the progress bar.
progressBar.start(chunkedDatasets.length, 0);

// Load all the wikitext dataset data to the BERT Database.
console.log('Loading Wikitext to BERT Database...');
const startTime = new Date();
for (let i = 0; i < chunkedDatasets.length; i++) {
  if (db.length() === db.max_length()) {
    // Update the progress bar.
    progressBar.update(i + 1);
    continue;
  }

  const section = chunkedDatasets[i];
  if (section.length === 1) {
    await db.add([[section[0], '']]);
  }
  else {
    const data: string[][] = [];
    for (let j = 0; j < section.length - 1; j++) {
      data.push([section[j], section[j + 1]]);
    }
    await db.add(data);
  }

  // Update the progress bar.
  progressBar.update(i + 1);
};
const endTime = new Date();

// Stop the progress bar.
progressBar.stop();

// Output status.
console.log('Successfully loaded Wikitext to BERT Database.');
console.log('Number of database entries:', db.length());
console.log('Database size limit (entries):', db.max_length());

// Compute the time it took to load the data to the database.
const elapsedMilliseconds = endTime.getTime() - startTime.getTime();
const elapsedSeconds = Math.floor(elapsedMilliseconds / 1000);
const elapsedMinutes = Math.floor(elapsedSeconds / 60);
const elapsedHours = Math.floor(elapsedMinutes / 60);
const formattedTime = `${elapsedHours} hours, ${elapsedMinutes % 60} minutes, ${elapsedSeconds % 60}seconds`;
console.log('Time to load Wikitext to database:', formattedTime);

// Save the database.
console.log('Saving Wikitext BERT Database...');
db.save('./wikitext-BERT_DB');
const oldLength: number = db.length();
console.log('Wikitext BERT Database saved.');

// Test loading the database.
console.log('Loading Wikitext BERT Database...');
db.load('./wikitext-BERT_DB');
console.log('Wikitext BERT Database loaded.');

// Verify all data loaded (cannot verify much else about the integrity
// of the data loaded).
const sameLength: boolean = oldLength === db.length();
console.log('Successfully reloaded Wikitext to BERT Databse:', sameLength);

// Exit the program.
process.exit(0);