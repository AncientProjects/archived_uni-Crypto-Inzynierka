import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


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
    sequence_len = forecasts.shape[1]
    forecast_index = int(start/sequence_len)
    fig, ax = plt.subplots()
    ax.plot(time_test[start:end], raw_test[start:end], format, label=label1)
    if sequence_len == 1:
        ax.plot(time_test[start:end], forecasts[start:end], format, label=label2)
    else:
        for i in range(forecast_index, len(forecasts)):
            off_s = i * sequence_len
            off_e = off_s + sequence_len
            xaxis = [x for x in time_test[off_s:off_e]]
            yaxis = forecasts[i]
            ax.plot(xaxis, yaxis, format, label=label2 if i == forecast_index else '', color='red')

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
