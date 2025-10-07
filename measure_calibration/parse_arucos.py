import re

def parse_arucos(opti_string):
    pattern = r"aruco_(\d+)\((\S+)\): { Q: \[([-\d., ]+)\]"
    matches = re.findall(pattern, opti_string)
    arucos = {}
    for match in matches:
        aruco_id = int(match[0])
        parent = match[1]
        position = [float(coord) for coord in match[2].split(',')]
        arucos[aruco_id] = (parent, position)
    return arucos