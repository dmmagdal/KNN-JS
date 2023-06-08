// datasetHelpers.ts

import { PreTrainedTokenizer } from '@xenova/transformers';


export function processText(data: string[], tokenizer: PreTrainedTokenizer): string[][] {
  // List to contain all tokenized.
  const textChunks: string[][] = [];

  // Iterate through each line item in the list of text.
  data.forEach((text: string) => {
    // Use regex to find lines that are titles (formatted as
    // " = [title] = ", " = = [title] = = ", or
    // " = = = [title] = = = ").
    const title1Regex: RegExp = / = [A-Za-z ]+ =/;
    const title2Regex: RegExp = / = = [A-Za-z ]+ = =/;
    const title3Regex: RegExp = / = = = [A-Za-z ]+ = = =/;
    const testArray: boolean[] = [
      title1Regex.test(text), title2Regex.test(text),
      title3Regex.test(text)
    ];

    // Skip all title lines and lines that are empty.
    if (testArray.every((value) => value === true) || text === '') {
      return;
    }

    // Tokenize the text. From transformers-js GPT2 tokenizer, the
    // tokens are {input_ids, attention_mask} where the tensor shape is
    // (batch_size, seq_len). batch_size = 1 because we are passing
    // string text in 1 string at a time.
    const tokens = tokenizer(text);
    const inputIds: BigInt64Array = tokens.input_ids.data;

    // Check the length of the tokenized output. Each set of tokens
		// will be broken in to chunks of 64. These chunks will become the
		// neighbor and continuation in their un-tokenized form for the
		// BERT database. 
    const localTokenChunks: BigInt64Array[] = divideChunksBigInt64Arr(
      inputIds, 64
    );
    const localTextChunks: string[] = [];
    localTokenChunks.forEach((tokenChunk: BigInt64Array) => {
      localTextChunks.push(tokenizer.decode(
        Array.from(tokenChunk, (value) => Number(value))
      )); // tokenizer takes in number[]
    });

    // Append local text chunks to list of all text chunks.
    textChunks.push(localTextChunks);
  });

  // Returned the list of processed text chunks. 
  return textChunks;
}

function divideChunksBigInt64Arr(inputs: BigInt64Array, n: number): BigInt64Array[] {
  const chunks: BigInt64Array[] = [];

  for (let i = 0; i < inputs.length; i += n) {
    const chunk: BigInt64Array = inputs.slice(i, i + n);
    chunks.push(chunk);
  }

  return chunks;
}