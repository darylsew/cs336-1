import os
import regex as re
from typing import BinaryIO
import copy
import pdb
import pickle

DEBUG = False

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage


#with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:

# Daryl's reflection on errors:
# - Overcomplicating things - should take a step back if something is too hard
# - Not tracking where we are spending most of our time - should formally profile
#   before spending any time optimizing things 

def pretokenize_complex(fname: str, special_tokens: list[str]) -> dict[str, int]:
    # Remove special tokens ahead of pretokenization
    # temporarily
    #fname = "debug.txt"

    data = open(fname).read()
    # cool, this one i literally knew but i was on the plane so couldnt lookup syntax
    # wrong: split_data = re.split("|".join(special_tokens), data)
    split_data = re.split("|".join(re.escape(t) for t in special_tokens), data)
    print("Sanitizing...")
    name_split = str(fname).split("/")
    path_part = name_split[:-1]
    name_part = name_split[-1]
    sanitized_fname = "/".join(path_part + [f"removed_special_tokens_{name_part}"])
    open(sanitized_fname, "w").write("".join(split_data))
    print("Sanitized, now processing chunk boundaries")

    with open(sanitized_fname, "rb") as f:
        num_processes = 4
        print("Processing chunk boundaries....")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        counts = {}
        # using tqdm might be nice...

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunk_idx = 0
        # for now, let's not bother parallelizing... forget the syntax (lolz)
        # 
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            #print(f"Pretokenizing chunk: {chunk}")
            #print("================================")
            matches = re.finditer(PAT, chunk)
            for pretoken_match in matches:
                actual = pretoken_match.group()
                if actual in counts:
                    counts[actual] += 1
                else:
                    counts[actual] = 1
            #print("|".join(results))
            print(f"completed processing chunk {chunk_idx}")
            chunk_idx += 1
        return counts

def apply_merge(merge: tuple[bytes, bytes], vocabulary: dict[int, bytes],  corpus: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    # Step 1 - Add new merged tokens to the vocabulary
    new_token_id = 0 if not vocabulary else max(vocabulary) + 1

    vocabulary[new_token_id] = merge[0] + merge[1]
    #if merge[0] == bytes('n'.encode('utf-8')):
    #    import pdb; pdb.set_trace()
    new_corpus = {}
    keys_to_delete = set()
    for key in corpus:
        #print(f"Matching process on key: {key}, merge tokens {merge}")
        matches = False
        for i in range(len(key)-1):
            # Not sure about how break affects inner loop
            if matches:
                continue
            b1 = key[i]
            b2 = key[i+1]
            if b1 == merge[0] and b2 == merge[1]:
                #print("Matches!")
                matches = True
                ls = list(key)
                ls[i] = merge[0] + merge[1]
                del ls[i+1]
                new_key = tuple(ls)
                new_corpus[new_key] = corpus[key]
                keys_to_delete.add(key)

    for key in corpus:
        if key not in keys_to_delete:
            new_corpus[key] = corpus[key]
    
    #import pdb; pdb.set_trace()
    return new_corpus


def print_new_vocabulary(vocabulary: dict[int, bytes]):
    s = ""
    for i in range(256, max(vocabulary)+1):
        s += f"i: {i}, vocab: {vocabulary[i]},"
    print(s)



def find_merge(
    corpus: dict[tuple[bytes], int],
) -> tuple[tuple[bytes, bytes], dict[tuple[bytes], int]]:
    """
      Input: corpus (dict of word tuples → counts)
    Output: best pair to merge (or None if nothing left)

    Algorithm:
    1. For each word in corpus:
        For each adjacent pair in word:
            Add word's count to that pair's total
    2. Find pair with highest count (break ties lexicographically with max)
    3. Return that pair
    """
    byte_pair_freqs: dict[tuple[bytes], int] = {}
    for word in corpus:
        for i in range(len(word)-1):
            b0 = word[i]
            b1 = word[i+1]
            b_pair = tuple([b0, b1])
            if b_pair in byte_pair_freqs:
                byte_pair_freqs[b_pair] += corpus[word]
            else:
                byte_pair_freqs[b_pair] = corpus[word]
    if not byte_pair_freqs:
        print("No byte pair freqs computed")
        return None
    top_val = max(byte_pair_freqs.values())
    ties = [key for key in byte_pair_freqs if byte_pair_freqs[key] == top_val]
    ties.sort()
    return ties[-1]

    """


    # Corpus is a dict from individual character bytes to int
    # ie (l, o, w): 5
    # we can use for computing byte pair frequencies
    existing_vocab = set(vocabulary.values())
    byte_pair_freqs: dict[tuple[bytes], int] = {}

    # caching optimization:
    # only the pair counts that were changed due to merge are relevant
    # no need to consider all other merges

    if corpus_diff:
        corpus_target = corpus_diff
    else:
        # Optimization
        corpus_target = corpus

    for i in vocabulary:
        for j in vocabulary:
            byte_i = vocabulary[i]
            byte_j = vocabulary[j]
            byte_tuple = tuple([byte_i, byte_j])
            if byte_i + byte_j in existing_vocab:
                continue
            
            for key, value in corpus_target.items():
                matches = False
                for k in range(len(key)-1):
                    if key[k] == byte_i and key[k+1] == byte_j:
                        matches = True
                if matches:
                    if byte_tuple in byte_pair_freqs:
                        byte_pair_freqs[byte_tuple] += value
                    else:
                        byte_pair_freqs[byte_tuple] = value

    #print(f"Byte pair freqs: {byte_pair_freqs}")
    if not byte_pair_freqs:
        return None, {}
    try:
        top_val = max(byte_pair_freqs.values())
    except Exception as e:
        print(e)
        pdb.set_trace()
    ties = [key for key in byte_pair_freqs if byte_pair_freqs[key] == top_val and key not in vocabulary]
    ties.sort()
    # ties broken based on lexicographically greater, so take last element
    if not ties:
        return None, {}
    else:
        merge_tuple = ties[-1]
        return merge_tuple, byte_pair_freqs
    """



"""
Problem (train_bpe): BPE Tokenizer Training (15 points)
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.
Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
To test your BPE training function against our provided tests, you will first need to implement the
test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py.
Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
you can implement the key parts of your training method using some systems language, for instance
C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
require copying vs reading directly from Python memory, and make sure to leave build instructions, or
make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
even faster
"""

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs, # ??
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    cache_fname = str(input_path) + ".pkl2"
    if os.path.exists(cache_fname):
        print("Processing exists in cache! Using cache...")
        counts = pickle.loads(open(cache_fname, "rb").read())
    else:
        print(f"Doesn't exist in cache, computing pretokenization for {input_path}...")
        counts = pretokenize_complex(input_path, special_tokens)
        open(cache_fname, "wb").write(pickle.dumps(counts))
    #import pdb; pdb.set_trace()
    # Freq table is raw ascii counts
    freq_table: dict[tuple[bytes], int] = {}
    for key, value in counts.items():
        # key is a string, like "abc"
        # we want a tuple of bytes
        key_byte_ls = []
        for c in key:
            key_byte_ls.append(bytes(c.encode("utf-8")))
        key_byte_tuple = tuple(key_byte_ls)
        freq_table[key_byte_tuple] = value

    merges = []
    corpus = freq_table
    # every merge, we add a token to the vocabulary
    # so we want to do vocab_size - len(vocabulary) merges
    num_merges = vocab_size - len(vocabulary) - len(special_tokens)
    for i in range(num_merges):
        #print("=============")
        merge = find_merge(corpus)
        if merge is None or merge in vocabulary:
            print("No more valid merges!")
            break

        if DEBUG:
            print(f"Found merge! {merge}")
        merges.append(merge)
        if DEBUG:
            print("Applying merge - vocab before:")
            print_new_vocabulary(vocabulary)
        corpus = apply_merge(merges[-1], vocabulary, corpus)
        if DEBUG:
            print("Applying merge - vocab after:")
            print_new_vocabulary(vocabulary)
            print(f"Corpus after: {corpus}")
            print(f"Processing merge {i+1}/{num_merges}")
            print("=============")

    real_vocab_size = max(vocabulary) + 1 # off by one because 0 is a key
    for i in range(real_vocab_size, real_vocab_size + len(special_tokens)):
        idx = i - real_vocab_size
        vocabulary[i] = special_tokens[idx].encode("utf-8")
    #import pdb; pdb.set_trace()
    assert len(vocabulary) == real_vocab_size + len(special_tokens)
    assert len(vocabulary) == vocab_size
    assert len(merges) == num_merges
    return (vocabulary, merges)