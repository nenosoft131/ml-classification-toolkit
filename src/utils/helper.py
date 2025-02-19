def encode_to_ascii(string):
    encoded_string = ""
    for char in string:
        encoded_string += str(ord(char)) + " "
    return encoded_string.strip()
