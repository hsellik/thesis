import os
from pathlib import Path

ROOT = "C:/Users/Kasutaja/DATASET/duplicated/java-small-iclr/vocab/"
RESERVED_TOKENS = []
BPE_LIMIT = 25000
MIN_SUBTOKEN_COUNT = 1000

def main():
	bpe_tokens = build_vocab(token_generator())
	with open('bpe-vocab.txt', 'w', encoding='utf-8') as f:
		for token, _ in sorted(bpe_tokens.items(), key=lambda e: e[1], reverse=True):
			f.write(token)
			f.write('\n')
	
def token_generator():
	# for file in Path(ROOT).rglob('*.java'):
	for file in os.listdir(ROOT):
		print(file)
		with open(os.path.join(ROOT, file), "r", encoding='utf-8') as f:
			for line in f:
				for t in line.split(" "):
					yield t

def merge(dict, key, count):
	if key not in dict:
		dict[key] = count
	else:
		dict[key] += count

def merge_sets(dict, key, element):
	if key not in dict:
		dict[key] = set([element])
	else:
		dict[key].add(element)

def build_vocab(token_generator):
	counts = {}
	for token in token_generator:
		if token in RESERVED_TOKENS: continue
		if token not in counts:
			counts[token] = 0
		counts[token] += 1
	counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)
	counts = [(list(t), c) for t, c in counts if len(t) > 0]
	for ix in range(len(counts)):
		counts[ix][0][-1] += "#"
	bpe_pairs = {t:1e12 + (100-ix) for ix, t in enumerate(RESERVED_TOKENS)}
	char_counts = {}
	count_table = {}
	loc_table = {}
	for tix, (token, count) in enumerate(counts):
		for ix, c in enumerate(token):
			merge(char_counts, c, count)
			if ix > 0:
				pair = token[ix-1] + c
				merge(count_table, pair, count)
				merge_sets(loc_table, pair, tix)
	
	for char, count in char_counts.items():
		if count >= MIN_SUBTOKEN_COUNT:
			bpe_pairs[char] = count
			print("Step:", len(bpe_pairs))
	
	for step in range(BPE_LIMIT - len(bpe_pairs)):
		print("Step:", len(bpe_pairs) + 1)
		tc = 0
		top_pair = top_count = None
		for t, c in count_table.items():
			if t not in bpe_pairs and c > tc:
				top_pair, top_count = t, c
				tc = top_count
		if top_pair is None: break # Typically means vocabulary is too small for this BPE cut-off
		if top_count < MIN_SUBTOKEN_COUNT: break
		bpe_pairs[top_pair] = top_count
		for tix in loc_table[top_pair]:
			token, token_count = counts[tix]
			ix = 1
			while ix < len(token):
				if token[ix-1]+token[ix] == top_pair:
					if ix > 1: # Update counts of preceding token, if any
						count_table[token[ix-2]+token[ix-1]] -= token_count
						merge(count_table, token[ix-2]+top_pair, token_count)
						merge_sets(loc_table, token[ix-2]+top_pair, tix)
					if ix < len(token) - 1:
						count_table[token[ix]+token[ix+1]] -= token_count
						merge(count_table, top_pair+token[ix+1], token_count)
						merge_sets(loc_table, top_pair+token[ix+1], tix)
					# Finally, collapse the token and delete the remnant (so don't update ix)
					token[ix-1] = top_pair
					del token[ix]
				else:
					ix += 1
	return bpe_pairs


if __name__ == '__main__':
	main()
