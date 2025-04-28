import csv
import matplotlib.pyplot as plt


def read_csv(filename):
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data = {headers[0]: [], headers[1]: []}
        for row in reader:
            data[headers[0]].append(float(row[0]))
            data[headers[1]].append(float(row[1]))
    return headers, data


def print_statistics(data, headers):
    for header in headers:
        values = data[header]
        print(f"Столбец '{header}':")
        print(f"  Количество: {len(values)}")
        print(f"  Минимум: {min(values)}")
        print(f"  Максимум: {max(values)}")
        print(f"  Среднее: {sum(values) / len(values):.4f}\n")


def calculate_regression(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    a = numerator / denominator
    b = mean_y - a * mean_x
    return a, b


def plot_data(x, y, xlabel, ylabel):
    plt.scatter(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Исходные точки')


def plot_regression_line(x, y, a, b):
    plt.plot(x, [a * xi + b for xi in x], color='red', label='Линия регрессии')
    plt.legend()


def plot_error_squares(x, y, a, b):
    for xi, yi in zip(x, y):
        y_pred = a * xi + b
        plt.plot([xi, xi], [yi, y_pred], color='green', linestyle='--')
        plt.fill_between([xi - 0.1, xi + 0.1], yi, y_pred, color='green', alpha=0.3)


def main():
    filename = input("Введите имя CSV файла (например, data.csv): ")
    headers, data = read_csv(filename)

    print("Доступные столбцы:", headers)
    x_col = input(f"Выберите столбец для X из {headers}: ")
    y_col = input(f"Выберите столбец для Y из {headers}: ")

    x = data[x_col]
    y = data[y_col]

    print_statistics(data, headers)

    # Фигуры для графиков
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Первый график: исходные точки
    plt.sca(axs[0])
    plot_data(x, y, x_col, y_col)

    # Второй график: исходные точки + прямая
    plt.sca(axs[1])
    plot_data(x, y, x_col, y_col)
    a, b = calculate_regression(x, y)
    plot_regression_line(x, y, a, b)

    # Третий график: исходные точки + прямая + квадраты ошибок
    plt.sca(axs[2])
    plot_data(x, y, x_col, y_col)
    plot_regression_line(x, y, a, b)
    plot_error_squares(x, y, a, b)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
