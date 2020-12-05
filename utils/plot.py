from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl


def plot_no_model_results(time_test, x_test, forecast, name):
    if name == "naive_config":
        plot_naive(time_test, x_test, forecast)
    elif name == "moving_average_config":
        plot_real_and_forecasts(time_test, x_test, forecast)


def plot_naive(time_test, x_test, forecast):
    plt.figure(figsize=(10, 6))
    plot_series(time_test, x_test, start=0, end=150, label="Series")
    plot_series(time_test, forecast, start=1, end=151, label="Forecast")
    plt.show()


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()


def plot_better(values, start=0, end=None, x_label='', y_label='', legend=''):
    plt.plot(values[start:end])
    tick_size = int((max(values[start:end]) - min(values[start:end])) / 8)
    ticks = [(min(values[start:end]) + tick_size * n) for n in range(9)]
    plt.yticks(ticks)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend([legend])
    plt.grid(True)
    plt.show()


def plot_real_and_forecasts(time_test, raw_test, forecasts, format="-", start=70, end=None, label1="Series",
                            label2="Forecast"):
    fig, ax = plt.subplots()
    ax.plot(time_test[start:end], raw_test[start:end], format, label=label1)
    ax.plot(time_test[start:end], forecasts[start:end], format, label=label2)

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode="anchor")
    plt.xticks(fontsize=10)
    # formatter = ticker.FormatStrFormatter('$%1.2f')
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label1 and label2:
        plt.legend(fontsize=14)
    plt.grid(True)
    # plt.grid(True, color='w', linestyle='-', linewidth=0.8)
    # plt.gca().patch.set_facecolor('0.85')
    plt.show()
