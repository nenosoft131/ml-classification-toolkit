def encode_to_ascii(string):
    encoded_string = ""
    for char in string:
        encoded_string += str(ord(char)) + " "
    return encoded_string.strip()

def decode_from_ascii(encoded_string):
    decoded_string = ""
    ascii_list = encoded_string.split()
    for ascii_code in ascii_list:
        decoded_string += chr(int(ascii_code))
    return decoded_string