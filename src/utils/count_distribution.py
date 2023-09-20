
distribution = {
    'Name': 0,
    'Date': 0,
    'Time': 0,
    'Initials': 0,
    'Profession': 0,
    'Internal_Location': 0,
    'Hospital': 0,
    'Address': 0,
    'Age': 0,
    'Phone': 0,
    'Email': 0,
    'URL': 0,
    'SSN': 0,
    'ID': 0,
    'Other': 0
}

import os

categorie_count = {}
from os.path import join, dirname 
TRAIN_PATH = join(dirname(__file__), '../../data/corpus/ehr/train')
TEST_PATH = join(dirname(__file__), '../../data/corpus/ehr/test')

for filename in os.listdir(TRAIN_PATH):
    if filename.endswith(".ann"):

        with open(join(TRAIN_PATH, filename), 'r') as file:
            for line in file:
                field = line.strip().split('\t')[1].split()[0]

                if field in distribution:
                    distribution[field] += 1

for filename in os.listdir(TEST_PATH):
    if filename.endswith(".ann"):

        with open(join(TEST_PATH, filename), 'r') as file:
            for line in file:
                field = line.strip().split('\t')[1].split()[0]

                if field in distribution:
                    distribution[field] += 1

print(distribution)     