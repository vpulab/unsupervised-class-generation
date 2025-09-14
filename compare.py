dataset_path = 'dataset_trucks/'

# Read the content of the first file into a set
with open(f'{dataset_path}/valid_indices_2.txt', 'r') as f1:
    files1 = set(line.strip() for line in f1)

# Read the content of the second file into another set
with open(f'{dataset_path}/valid_indices_aux.txt', 'r') as f2:
    files2 = set(line.strip() for line in f2)

# Find the intersection between the two sets
common_files = files1.intersection(files2)

# Output the results
print(f'Number of common files: {len(common_files)}')
print('Common files:')
count = 0
for file in common_files:
    count += 1

print(f'Files in file 1: {len(files1)}')
print(f'Files in file 2: {len(files2)}')
print(f'Count: {count}')
