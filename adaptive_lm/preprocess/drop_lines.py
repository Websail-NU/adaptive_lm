import random
import sys

input_path = sys.argv[1]
drop_rate = float(sys.argv[2])/100
control_path = sys.argv[3]
output_path = sys.argv[4]

control_words = set()
with open(control_path) as ifp:
    for line in ifp:
        control_words.add(line.strip())

with open(input_path) as ifp, open(output_path, 'w') as ofp:
    for line in ifp:
        tokens = line.strip().split()
        if any(map(lambda each: each in control_words, tokens)):
            if random.random() < drop_rate:
                continue
        ofp.write(line)
