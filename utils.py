def reversed(list):
    temp = []
    for i in range(len(list) - 1, -1, -1):
        temp.append(list[i])
    return temp
