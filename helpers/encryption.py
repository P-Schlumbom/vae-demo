

def string_to_hex(string):
    return "".join("{:02x}".format(ord(c)) for c in string)


def hex_to_string(hex_code):
    return "".join([chr(int(hex_code[i:i + 2], 16)) for i in range(0, len(hex_code), 2)])


# full range of ascii characters
uni_start = 32
uni_end = 127
universe = [c for c in (chr(i) for i in range(uni_start, uni_end))]

uni_len = len(universe)


def vigenere(text: str, key: str, encrypt=True):
    result = ''

    for i in range(len(text)):
        letter_n = ord(text[i]) - uni_start
        key_n = ord(key[i % len(key)]) - uni_start

        if encrypt:
            value = (letter_n + key_n) % uni_len
        else:
            value = (letter_n - key_n) % uni_len
            print(letter_n, key_n, value)

        result += chr(value + uni_start)

    return result


def vigenere_encrypt(text: str, key: str):
    return vigenere(text=text, key=key, encrypt=True)


def vigenere_decrypt(text: str, key: str):
    return vigenere(text=text, key=key, encrypt=False)