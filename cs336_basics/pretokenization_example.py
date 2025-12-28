import os
import regex as re
from typing import BinaryIO
import copy
import pdb


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


open("debug.txt", "w").write("""low low low low low lower lower widest widest widest newest newest newest newest newest newest""")
#with open("../data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:


def pretokenize_complex(fname: str) -> dict[str, int]:
    with open(fname, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        counts = {}

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            print(f"Pretokenizing chunk: {chunk}")
            print("================================")
            matches = re.finditer(PAT, chunk)
            for pretoken_match in matches:
                actual = pretoken_match.group()
                if actual in counts:
                    counts[actual] += 1
                else:
                    counts[actual] = 1
            #print("|".join(results))
        return counts

def pretokenize_simple(fname: str) -> dict[str, int]:
    s = open(fname).read().split(" ")
    counts = {}
    for c in s:
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = 1
    return counts


def bpe_example():
    """
    Here is a stylized example from Sennrich et al. [2016]. Consider a corpus consisting of the following text
    low low low low low
    lower lower widest widest widest
    newest newest newest newest newest newest
    and the vocabulary has a special token <|endoftext|>.
    Vocabulary We initialize our vocabulary with our special token <|endoftext|> and the 256 byte
    values.
    Pre-tokenization For simplicity and to focus on the merge procedure, we assume in this example
    that pretokenization simply splits on whitespace. When we pretokenize and count, we end up with the
    frequency table.
    {low: 5, lower: 2, widest: 3, newest: 6}
    It is convenient to represent this as a dict[tuple[bytes], int], e.g. {(l,o,w): 5 …}. Note that even
    a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte,
    just as there is no char type in Python to represent a single character.
    Merges We first look at every successive pair of bytes and sum the frequency of the words where they
    appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es')
    and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the
    pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.
    In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would
    merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the
    sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
    'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].
    If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocabulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].
    With this vocabulary and set of merges, the word newest would tokenize as [ne, west].
    """
    # >>> bytes([ord('a')]).decode("utf-8")
    # 'a'
    vocabulary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # XXX: figure out how to deal with eot tokens correctly here -
    # i think they just get stripped out / replaced with special strings before processing
    # vocabulary.append("|endoftext|")
    counts = pretokenize_simple("debug.txt")
    #print(counts)
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
    for i in range(6):
        #print("=============")
        merge = find_merge(vocabulary, corpus)
        print(f"Found merge! {merge}")
        merges.append(merge)
        #print("Applying merge - vocab before:")
        print_new_vocabulary(vocabulary)
        corpus = apply_merge(merges[-1], vocabulary, corpus)
        print("Applying merge - vocab after:")
        print_new_vocabulary(vocabulary)
        print(f"Corpus after: {corpus}")
        print("=============")

    # XXX How do we actually merge pretokens?
    # Should we recompute over the source corpus with our new vocab?
    # Or can we get away with using that initial pretokenized map in our sorting?
    # If we don't consider cross-pretoken-boundaries for tokenizing, then using
    # the initial pretokenized map should be sufficient - we're doing subword tokenization


def apply_merge(merge: tuple[bytes, bytes], vocabulary: dict[int, bytes],  corpus: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    # Step 1 - Add new merged tokens to the vocabulary
    new_token_id = 0 if not vocabulary else max(vocabulary) + 1

    vocabulary[new_token_id] = merge[0] + merge[1]
    if merge[0] == bytes('n'.encode('utf-8')):
        import pdb; pdb.set_trace()


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



def find_merge(vocabulary: dict[int, bytes], corpus: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
    # Corpus is a dict from individual character bytes to int
    # ie (l, o, w): 5
    # we can use for computing byte pair frequencies
    byte_pair_freqs: dict[tuple[bytes], int] = {}
    for i in vocabulary:
        for j in vocabulary:
            byte_i = vocabulary[i]
            byte_j = vocabulary[j]
            byte_tuple = tuple([byte_i, byte_j])
            for key, value in corpus.items():
                matches = False
                for k in range(len(key)-1):
                    if key[k] == byte_i and key[k+1] == byte_j:
                        matches = True
                if matches:
                    if byte_tuple in byte_pair_freqs:
                        byte_pair_freqs[byte_tuple] += value
                    else:
                        byte_pair_freqs[byte_tuple] = value
    print(byte_pair_freqs)
    try:
        top_val = max(byte_pair_freqs.values())
    except Exception as e:
        print(e)
        pdb.set_trace()
    ties = [key for key in byte_pair_freqs if byte_pair_freqs[key] == top_val and key not in vocabulary]
    ties.sort()
    # ties broken based on lexicographically greater, so take last element
    if not ties:
        return None
    else:
        merge_tuple = ties[-1]
        return merge_tuple
    
bpe_example()