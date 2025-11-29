import numpy as np
import varint
def decode_single_list(file_path, offset):
    """
    Decode a single sublist from a varint+delta file at given byte offset.
    """
    with open(file_path, "rb") as f:
        f.seek(offset)

        # First, read the length (varint)
        decoded_bytes = bytearray()
        while True:
            b = f.read(1)
            if not b:
                raise EOFError("Unexpected end of file while reading length")
            decoded_bytes += b
            if b[0] < 128:
                break

        length = varint.decode_bytes(decoded_bytes)

        if length == 0:
            return np.array([], dtype=np.uint32)

        # Now read varint-encoded deltas
        deltas = []
        while len(deltas) < length:
            decoded_bytes = bytearray()
            while True:
                b = f.read(1)
                if not b:
                    raise EOFError("Unexpected end of file while reading delta")
                decoded_bytes += b
                if b[0] < 128:
                    break
            deltas.append(varint.decode_bytes(decoded_bytes))

        # Convert deltas to absolute IDs
        return np.cumsum(deltas, dtype=np.uint32)

def encode_and_save_indices_with_offsets( path, list_of_indices):
    """
    Encode list of lists like before, but return list of byte offsets for each sublist.
    """
    offsets = []
    ptr = 0  # track current byte offset

    with open(path, "wb") as f:
        for indices in list_of_indices:
            offsets.append(ptr)

            if len(indices) == 0:
                # Write length=0
                encoded_len = varint.encode(0)
                f.write(encoded_len)
                ptr += len(encoded_len)
                continue

            indices = np.array(indices, dtype=np.uint32)

            # Delta encode
            deltas = np.empty_like(indices)
            deltas[0] = indices[0]
            deltas[1:] = indices[1:] - indices[:-1]

            # Write length prefix
            encoded_len = varint.encode(len(deltas))
            f.write(encoded_len)
            ptr += len(encoded_len)

            # Write varint-encoded deltas
            for d in deltas:
                encoded_d = varint.encode(int(d))
                f.write(encoded_d)
                ptr += len(encoded_d)

    return offsets