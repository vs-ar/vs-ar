def compress_lz77(data: str):
    """
    Compress a string using a naive LZ77-like approach.
    Returns a list of tokens, where each token is either:
      - A single character (literal), or
      - A tuple (offset, length) indicating a match.
    """
    i = 0
    tokens = []
    n = len(data)

    while i < n:
        best_offset = 0
        best_length = 0

        # Try to find the longest match of a substring starting at 'i'
        # in the *already processed* part of the data [0..i).
        for j in range(i):
            length = 0
            while (j + length < i) and (i + length < n) and data[j + length] == data[i + length]:
                length += 1
            # Keep track of the longest match found so far
            if length > best_length:
                best_length = length
                best_offset = i - j

        # Decide whether to emit a match or a literal
        # (Here we use a simple threshold: if the best match is < 2 chars, just use a literal.)
        if best_length < 2:
            # Emit a literal character
            tokens.append(data[i])
            i += 1
        else:
            # Emit a match (offset, length)
            tokens.append((best_offset, best_length))
            i += best_length

    return tokens


def decompress_lz77(tokens) -> str:
    """
    Decompress a list of LZ77-like tokens back into the original string.
    Tokens are either single-character literals or (offset, length) tuples.
    """
    result = []

    for token in tokens:
        if isinstance(token, str):
            # Literal character
            result.append(token)
        else:
            # A (offset, length) match
            offset, length = token
            start_index = len(result) - offset
            for _ in range(length):
                result.append(result[start_index])
                start_index += 1

    return "".join(result)


if __name__ == "__main__":
    # Example usage
    data = "ABBABBABABA"
    compressed = compress_lz77(data)
    decompressed = decompress_lz77(compressed)

    print("Original:   ", data)
    print("Compressed: ", compressed)
    print("Decompressed:", decompressed)
    print("Success?    ", data == decompressed)
