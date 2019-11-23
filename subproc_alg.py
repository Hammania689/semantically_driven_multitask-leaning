import subprocess 
import sys
import re

output = subprocess.Popen(["similarity.pl --type 'WordNet::Similarity::lesk' paper#n#1 man#n#1"], shell=True, stdout=subprocess.PIPE)

output = output.stdout.read()
parsed = re.split('\s+', str(output))
# print(output)
print(str(output))
print(str(output).replace('#','.'))
print(output[-5:-1])
print(parsed[-1][:-3])
