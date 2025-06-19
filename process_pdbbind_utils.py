def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res



def GetPDBList(Path):
    with open(Path, 'rb') as f:
        print(f)
        lines = f.read().decode().strip().split('\n')
    res = []
    for line in lines:
        if "//" in line:
            temp = line.split()
            res.append(temp[0])
    return res


def GetPDBList_core(Path):
    with open(Path, 'r') as f:
        lines = f.read().split('\n')
    res = []
    for line in lines:
        if "code" not in line and len(line)!=0:
            temp = line.split(',')
            temp = temp[0].split()
            res.append(temp[0])
    return res

def GetPDBDict_core(Path):
    with open(Path, 'r') as f:
        lines = f.read().split('\n')
    res = {}
    for line in lines:
        if "code" not in line and len(line)!=0:
            temp = line.split(',')
            temp = temp[0].split()
            name, score = temp[0], float(temp[3])
            print(f"CORE INDEX LIST ROW: {temp}")
            res[name] = score
    return res
