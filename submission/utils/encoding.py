def write_var_int(file, value: int) -> None:
    """Write a non-negative integer using variable-byte encoding."""
    while value >= 128:
        file.write(bytes([(value & 127) | 128]))
        value >>= 7

    file.write(bytes([value]))

def encode_var_int(buffer: bytearray, value: int) -> None:
    """Append a non-negative integer using variable-byte encoding (fast inlined path)."""
    if value < 128:
        buffer.append(value)
    elif value < 16384:
        buffer.append((value & 127) | 128)
        buffer.append(value >> 7)
    else:
        while value >= 128:
            buffer.append((value & 127) | 128)
            value >>= 7
        buffer.append(value)

def read_var_int(file) -> int:
    """Read one variable-byte encoded non-negative integer."""
    value = 0
    shift = 0

    while True:
        byte = file.read(1)

        if not byte:
            raise EOFError("Unexpected end of index file")

        byte = byte[0]
        value |= (byte & 127) << shift

        if not (byte & 128):
            return value

        shift += 7

