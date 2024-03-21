# Training an encoder and decoder on the byte pair encoding algorithm with heuristics for chess

import pickle

def train():
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    changes = list(enumerate(chars))
    for i in range(5,13):
        for j in range(20,28):
            changes.append((len(changes),(j,i)))
    print(changes)
    with open('changes.pickle', 'wb') as f:
        pickle.dump(changes, f)


def encode(s):
    with open('changes.pickle', 'rb') as f:
        changes = pickle.load(f)
    encoded = list(s)
    lookup = { changes[i][1]:i for i in range(len(changes)) }
    i = 0
    while i < len(encoded):
        if isinstance(encoded[i], int):
            if i < len(encoded)-1:
                if lookup.get((encoded[i], encoded[i+1])):
                    encoded[i] = lookup.get((encoded[i], encoded[i+1]))
                    encoded.pop(i+1)
                else: i += 1
            else: break
        else: encoded[i] = lookup[encoded[i]]
    return encoded

def decode(encoded):
    with open('changes.pickle', 'rb') as f:
        changes = pickle.load(f)
    i = len(encoded)-1
    while i>=0:
        if len(changes[encoded[i]][1])==1:
            encoded[i] = changes[encoded[i]][1]
            i -= 1
        else:
            temp = changes[encoded[i]][1]
            encoded[i] = temp[0]
            i+=1
            encoded.insert(i, temp[1])
    return "".join(encoded)

# train()

# print(decode(encode("""e4 e5 Nf3""")))
