import random
import os

DIRNAME = os.path.dirname(__file__)

FIRSTNAMES_PATH = os.path.join(DIRNAME, 'firstnames')
FEMALE_FIRSTNAMES_FILE = os.path.join(FIRSTNAMES_PATH, 'female_firstnames.txt')
MALE_FIRSTNAMES_FILE = os.path.join(FIRSTNAMES_PATH, 'male_firstnames.txt')
FIRSTNAMES_FILE = os.path.join(FIRSTNAMES_PATH, 'firstnames.txt')

LASTNAMES_PATH = os.path.join(DIRNAME, 'lastnames')
MALE_LASTNAMES_FILE = os.path.join(LASTNAMES_PATH, 'male_lastnames.txt')
LASTNAMES_FILE = os.path.join(LASTNAMES_PATH, 'lastnames.txt')

SEED = 1234

def save_list(list, file_path):
    with open(file_path, 'w') as output:
        output.write(str(list[0]))

        for i in range(1, len(list)):
            output.write('\n')
            output.write(str(list[i]))

def build_firstnames():
    female_names = open(FEMALE_FIRSTNAMES_FILE, 'r').read().splitlines()
    male_names = open(MALE_FIRSTNAMES_FILE, 'r').read().splitlines()

    female_names = [*map(lambda x: x.lower(), female_names)]
    male_names = [*map(lambda x: x.lower(), male_names)]

    names = []
    names.extend(female_names)
    names.extend(male_names)
    names.sort()

    save_list(names, FIRSTNAMES_FILE)

def build_lastnames():
    names = open(MALE_LASTNAMES_FILE, 'r').read().splitlines()
    names = [*map(lambda x: x.lower(), names)]
    names.sort()

    save_list(names, LASTNAMES_FILE)

def build_train_test(file_path, output_path):
    data=open(file_path, 'r').read().splitlines()
    
    random.seed(SEED)
    random.shuffle(data)

    border = int((len(data) + 1) * 0.9)

    train_data = data[:border]
    test_data = data[border:]

    train_path = os.path.join(output_path, 'train.txt')
    test_path = os.path.join(output_path, 'test.txt')

    save_list(train_data, train_path)
    save_list(test_data, test_path)


def main():
    build_firstnames()
    build_lastnames()

    build_train_test(FIRSTNAMES_FILE, FIRSTNAMES_PATH)
    build_train_test(LASTNAMES_FILE, LASTNAMES_PATH)

main()
