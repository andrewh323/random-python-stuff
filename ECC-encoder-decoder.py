import random

#Functions to compute inverse modulo
def egcd(a,b):
    prev_r = abs(a)
    curr_r = abs(b)
    x = 0
    y = 1
    prev_x = 1
    prev_y = 0
    while curr_r:
        prev_r, (divide, curr_r) = curr_r, divmod(prev_r, curr_r)
        x = prev_x - x*divide
        prev_x = x
        y = prev_y - divide*y
        prev_y = y
    if (a < 0):
        prev_x = -1 * prev_x
    if (b < 0):
        prev_y = -1 * prev_y
    return prev_r, prev_x, prev_y

def inverse_mod(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('mod inverse does not exist')
    else:
        return x % m

#Functions to convert from strings to integers
def stringToInt(msg):
    asciiMsg = ''
    for i in msg:
        asciiMsg += str(ord(i) + 100)
    return asciiMsg

def intToString(msg):
    i = 0
    wordSplit = ''
    intMsg = ''
    while i < len(str(msg)):
        wordSplit = int(msg[i:i+3]) - 100
        intMsg += chr(wordSplit)
        i = i + 3
    return intMsg


#Definition of a secp256k1
p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f

#Curve y^2 = x^3 + ax + b
a = 0x0
b = 0x7

#generator
g = (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798, 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)

#order
n = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141

#cofactor
h = 0x1

#Elliptical Curve Group Operations
def doubling(x1, y1):
    lm = ((3*(x1**2) + a) * inverse_mod(2*y1, p)) % p
    x3 = (lm**2 - 2*x1) % p
    y3 = ((x1 - x3)*lm - y1) % p
    return x3, y3

def addition(x1, y1, x2, y2):
    lm = ((y2 - y1) * inverse_mod(x2 - x1, p)) % p
    x3 = (lm**2 - x1 - x2) % p
    y3 = ((x1 - x3)*lm - y1) % p
    return x3, y3

def multiply(mult, gen):
    init = 0
    (xSpare, ySpare) = gen
    (x1,y1) = gen
    for i in str(bin(mult)[2:]):
        if (i == '1') and (init == 0):
            init = 1
        elif(i == '1') and (init == 1):
            (x3, y3) = doubling(x1,y1)
            (x3, y3) = addition(xSpare, ySpare, x3, y3)
        else:
            (x3, y3) = doubling(x1, y1)
    return x3, y3

#Generate random value, keys
k = random.getrandbits(256)
privKey = random.getrandbits(256)
pubKey = multiply(privKey, g)

#Encrypt and decrpyt messages with public and private keys
def encrypt(pubKey, msg):
    C1 = multiply(k, g)
    C2 = multiply(k, pubKey)[0] + int(msg)
    return C1, C2

def decrypt(C1, C2, privKey):
    return C2 - multiply(privKey, C1)[0]

#Given messages
M1 = "I am an undergraduate student at queen's university"
M2 = "Andrew Heaton"

print("Random Value k: ", k)
print("Public Key: ", pubKey)
print("Private Key: ", privKey)
print("Generator: ", g)

#Encrypt and decrypt first message M1
(C1, C2) = encrypt(pubKey, stringToInt(M1))
print("Generated Ciphertext for M1", C1, C2)
decryptM1 = decrypt(C1, C2, privKey)
plaintextM1 = intToString(str(decryptM1))
print("Decrypted Plaintext for M1: ", plaintextM1)
#Create key files
f = open("privateKey.txt", "a")
f.write(str(privKey))
f.close()
f = open("publicKey.txt", "a")
f.write(str(pubKey))
f.close()

#Create encoded and decoded message files for M1
f = open("Ciphertext1.txt", "a")
f.write(str(C1))
f.write(str(C2))
f.close()

f = open("DecryptedPlaintext1.txt", "a")
f.write(plaintextM1)
f.close()

#Encrypt and decrypt first message M2
(C1, C2) = encrypt(pubKey, stringToInt(M2))
print("Generated Ciphertext for M2", C1, C2)
decryptM2 = decrypt(C1, C2, privKey)
plaintextM2 = intToString(str(decryptM2))
print("Decrypted Plaintext for M2: ", plaintextM2)

#Create encoded and decoded message files for M2
f = open("Ciphertext2.txt", "a")
f.write(str(C1))
f.write(str(C2))
f.close()

f = open("DecryptedPlaintext2.txt", "a")
f.write(plaintextM2)
f.close()
