import numpy as np

def string_to_element_array(gwstr, colsep=" ", rowsep="\n", elementsep="."):
    arr = []
    gwstr = gwstr.strip()
    for row in gwstr.split(rowsep):
        arr.append([])
        if colsep == "":
            cells = list(row)
        else:
            cells = row.split(colsep)
        for cell in cells:
            arr[-1].append([e for e in cell.split(elementsep) if e != ""])
    return arr

def element_array_to_string(ta, colsep=" ", rowsep="\n", elementsep="."):
    # format the elements in a location
    ta = [[elementsep.join(sorted(c)) if len(c) > 0 else elementsep for c in r] for r in ta]
    
    ta = np.array(ta, dtype=object)
    for col in range(ta.shape[1]):
        maxstrlen = max([len(s) for s in ta[:, col]])
        ta[:, col] = [s.center(maxstrlen) for s in ta[:, col]]

    tastring = []
    for r in ta:
        row = colsep.join(r)
        tastring.append(row)   

    tastring = rowsep.join(tastring)
    return tastring