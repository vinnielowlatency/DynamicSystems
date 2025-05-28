def z85_decode(char1, char2, char3, char4, char5):
   # Decoder lookup table (index is ASCII value - 32)
   decoder_lookup_array = [
       0x00, 0x44, 0x00, 0x54, 0x53, 0x52, 0x48, 0x00, 0x4B, 0x4C, 0x46, 0x41, 
       0x00, 0x3F, 0x3E, 0x45, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
       0x08, 0x09, 0x40, 0x00, 0x49, 0x42, 0x4A, 0x47, 0x51, 0x24, 0x25, 0x26, 
       0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 
       0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x4D, 
       0x00, 0x4E, 0x43, 0x00, 0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 
       0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 
       0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x4F, 0x00, 0x50, 0x00, 0x00
   ]
   
   # Convert ASCII characters to their corresponding values using lookup table
   # Subtract 32 from ASCII value to get the index in decoder array
   index1 = ord(char1) - 32
   index2 = ord(char2) - 32
   index3 = ord(char3) - 32
   index4 = ord(char4) - 32
   index5 = ord(char5) - 32
   
   # Look up the decoded values from the decoder table
   value1 = decoder_lookup_array[index1]
   value2 = decoder_lookup_array[index2]
   value3 = decoder_lookup_array[index3]
   value4 = decoder_lookup_array[index4]
   value5 = decoder_lookup_array[index5]
   
   # Combine the values to form a 32-bit integer
   # Each value is multiplied by appropriate power of 85
   result = value1 * (85**4) + value2 * (85**3) + value3 * (85**2) + value4 * 85 + value5
   
   # Extract the four bytes from the 32-bit value (big-endian)
   byte1 = (result >> 24) & 0xFF  # MSB
   byte2 = (result >> 16) & 0xFF
   byte3 = (result >> 8) & 0xFF
   byte4 = result & 0xFF  # LSB
   
   return byte1, byte2, byte3, byte4


def main():
   # Take input from user
   encoded = input("Enter 5 Z85 characters: ")
   if len(encoded) != 5:
       print("Error: Z85 decode requires exactly 5 characters")
       return
   
   try:
       # Decode the input
       byte1, byte2, byte3, byte4 = z85_decode(encoded[0], encoded[1], encoded[2], encoded[3], encoded[4])
        
       # Display as ASCII if printable
       ascii_chars = []
       for byte in [byte1, byte2, byte3, byte4]:
           if 32 <= byte <= 126:  # Printable ASCII range
               ascii_chars.append(chr(byte))
           else:
               ascii_chars.append(f"\\x{byte:02x}")
       
       print(f"Decoded bytes (ASCII): {''.join(ascii_chars)}")
       
   except IndexError:
       print("Error: Invalid index in decoder table")
   except Exception as e:
       print(f"Error: {e}")

if __name__ == "__main__":
   main()