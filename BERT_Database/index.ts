// Main program.


import * as fs from 'fs';
import path from 'path';
import process from 'process';
import { BERTDatabase } from './BERTDB.js';


// ====================================================================
// UNIT TESTS
// ====================================================================

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
const db = new BERTDatabase(model_, func, limit);
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
// valid_response = db.remove(entries[3:5])
await db.remove(entries.slice(3, 5));
console.log('Removed 2 entries from database');
console.log('\tExpected length is 8');
console.log('\tFound length of', db.length());


// Attempt to remove an entry with a valid key (single removal).
// valid_response = db.remove([entries[2]])
await db.remove([entries[2]]);
console.log('Removed 1 entry from database');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with an invalid key (batch removal).
// invalid_response = db.remove(invalid_entries[:2])
await db.remove(invalidEntries.slice(0, 2));
console.log('Removed 2 invalid entries from database (no change to data)');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with an invalid key (single removal).
// invalid_response = db.remove([invalid_entries[0]])
await db.remove([invalidEntries[0]]);
console.log('Removed 1 invalid entry from database (no change to data)');
console.log('\tExpected length is 7');
console.log('\tFound length of', db.length());

// Attempt to remove an entry with both a valid and an invalid key
// (batch removal only).
// mixed_response = db.remove([invalid_entries[-1], entries[-1]])
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

// Start by clearing out the databse entirely (resets the index and
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


console.log('='.repeat(72));



// ====================================================================
// LOAD WIKITEXT
// ====================================================================
