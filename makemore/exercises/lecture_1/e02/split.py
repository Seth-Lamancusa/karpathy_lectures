import random


def read_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def split_data(data, train_frac=0.8, dev_frac=0.1):
    random.shuffle(data)
    train_size = int(len(data) * train_frac)
    dev_size = int(len(data) * dev_frac)

    train_data = data[:train_size]
    dev_data = data[train_size : train_size + dev_size]
    test_data = data[train_size + dev_size :]

    return train_data, dev_data, test_data


def write_file(data, file_path):
    with open(file_path, "w") as file:
        for line in data:
            file.write(line + "\n")


# Replace 'your_file.txt' with your file path
file_path = "makemore/names.txt"
data = read_file(file_path)

train_data, dev_data, test_data = split_data(data)

# Writing the splits to files
write_file(train_data, "makemore/exercises/lecture_1/e02/train.txt")
write_file(dev_data, "makemore/exercises/lecture_1/e02/dev.txt")
write_file(test_data, "makemore/exercises/lecture_1/e02/test.txt")
