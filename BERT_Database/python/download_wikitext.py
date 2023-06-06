# download_wikitext.py
# Simple script that downloads the wikitext datasets from huggingface
# transformers.
# Python 3.7
# Windows/MacOS/Linux


import json
import datasets
from datasets import load_dataset


def main():
	# Load wikitext dataset. Both have test, train, and validation
	# splits. Note that the test and validation splits between wiki-103
	# and wiki-2 are the same size but wiki-103 has 60x larger train
	# split.
	revisions = ['wikitext-103-v1', 'wikitext-2-v1']
	splits = ['train', 'test', 'validation']

	for rev in revisions:
		for split in splits:
			# Load dataset split.
			dataset = load_dataset('wikitext', rev, split=split)

			# Convert dataset to dictionary.
			dataset_dict = dataset.to_dict()

			# Save to json. 
			filename = './' + rev + '_' + split + '.json'
			with open(filename, 'w+') as f:
				json.dump(dataset_dict, f, indent=4)

			# Can also do this, but the above is 'cleaner' to rebuild.
			# dataset_dict.to_json(filename)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()