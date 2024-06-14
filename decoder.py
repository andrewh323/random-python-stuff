def decode(message_file):
    
    #initialize
    res = ""
    myList = []
    triangleDict = dict()
    decoded = []
    #read file and parse into array of strings
    f = open('message.txt', 'r')
    textStr = []
    for line in f:
        if line.strip():
            textStr.append(line.rstrip())
        
    f.close()
    print(textStr)

    #build dictionary
    for i in range(len(textStr)):
        key = ''.join(c for c in textStr[i] if c.isdigit())
        key = int(key)
        value = ''.join(j for j in textStr[i] if not j.isdigit()).strip()
        triangleDict[key] = value

    #sort strings in ascending numerical order
    sortedDict = dict(sorted(triangleDict.items()))

    #build list of pyramid ends
    for i in range(len(textStr)):
        triangleSide = (i+1)*(i+2)/2
        myList.append(int(triangleSide))

    for key in sortedDict:
        if key in myList:
            decoded.append(sortedDict[key])

    #format the string
    for i in range(len(decoded)):
        if i == 0:
            res += decoded[i]
        else:
            res += " " + decoded[i]
    return(res)
