import sys
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.strip().strip("[]").split()
            data.append((int(numbers[0]), int(numbers[1])))
    return data

def create_histogram(data):
    values, frequencies = zip(*data)
    plt.bar(values, frequencies, alpha=0.7)
    plt.xlabel('Symbol Code')
    plt.ylabel('Number of Occurrences')
    plt.title('Histogram of Symbol Codes')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        data = read_data_from_file(file_path)
        create_histogram(data)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()