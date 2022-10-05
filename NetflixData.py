from datetime import datetime
import os
import pandas as pd

start = datetime.now()
base = '/Users/marialentini/Library/CloudStorage' \
       '/OneDrive-PearsonPLC/User Profile/Recommender System/Netflix Data'

if not os.path.isfile('data.csv'):
    # read all txt file and store them in one big file
    data = open('data.csv', mode='w')

    row = list()
    files = [base + '/combined_data_1.txt', base + '/combined_data_2.txt',
             base + 'combined_data_3.txt', base + '/combined_data_4.txt']
    for file in files:
        print('reading ratings from {}...'.format(file))
        with open(file) as f:
            for line in f:
                del row[:]
                line = line.strip()
                if line.endswith(':'):
                    # all are rating
                    movid_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0, movid_id)
                    data.write(','.join(row))
                    data.write('\n')
        print('Done.\n')
    data.close()
print('time taken:', datetime.now() - start)
