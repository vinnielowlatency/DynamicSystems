def z85_decode(encoded_str):
    # Check if the input is exactly 5 characters
    if len(encoded_str) != 5:
        raise ValueError("Z85 decode requires exactly 5 characters")
    
    # Z85 decoder lookup table (index is ASCII value - 32)
    decoder_table = [
        0x00, 0x44, 0x00, 0x54, 0x53, 0x52, 0x48, 0x00, 0x4B, 0x4C, 0x46, 0x41, 
        0x00, 0x3F, 0x3E, 0x45, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
        0x08, 0x09, 0x40, 0x00, 0x49, 0x42, 0x4A, 0x47, 0x51, 0x24, 0x25, 0x26, 
        0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 
        0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x4D, 
        0x00, 0x4E, 0x43, 0x00, 0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 
        0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x4F, 0x00, 0x50, 0x00, 0x00
    ]
    
    # Convert each character to its Z85 value using the lookup table
    decoded_values = []
    for char in encoded_str:
        ascii_value = ord(char)
        if ascii_value < 32 or ascii_value > 127:
            raise ValueError(f"Invalid Z85 character: {char}")
        
        adjusted_index = ascii_value - 32
        z85_value = decoder_table[adjusted_index]
        
        if z85_value == 0 and char != ' ':  # Space is valid and maps to 0
            raise ValueError(f"Invalid Z85 character: {char}")
            
        decoded_values.append(z85_value)
    
    # Calculate the 32-bit integer from the 5 Z85 values
    value = 0
    for i in range(5):
        value = value * 85 + decoded_values[i]
    
    # Extract the 4 bytes from the 32-bit integer
    bytes_output = []
    for i in range(3, -1, -1):  # Extract bytes in big-endian order
        byte = (value >> (i * 8)) & 0xFF
        bytes_output.append(byte)
    
    return bytes_output

def main():
    # Get input from user
    encoded_str = input("Enter 5 Z85 encoded characters: ")
    
    try:
        decoded_bytes = z85_decode(encoded_str)
        
        # Print the decoded bytes in various formats
        print("\nDecoded Bytes:")
        print(f"Decimal: {decoded_bytes}")
        
        # Print in hexadecimal format
        hex_values = [f"0x{byte:02X}" for byte in decoded_bytes]
        print(f"Hexadecimal: {hex_values}")
        
        # Print as ASCII characters (when printable)
        ascii_chars = []
        for byte in decoded_bytes:
            if 32 <= byte <= 126:  # Printable ASCII range
                ascii_chars.append(chr(byte))
            else:
                ascii_chars.append(f"\\x{byte:02x}")  # Non-printable
        print(f"ASCII: {ascii_chars}")
        
        # Print as binary
        binary_values = [f"{byte:08b}" for byte in decoded_bytes]
        print(f"Binary: {binary_values}")
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()