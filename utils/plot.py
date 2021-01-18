import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from pandas import DataFrame


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()


def plot_series_with_ticks(values, start=0, end=None, x_label='', y_label='', legend=''):
    plt.plot(values[start:end])
    tick_size = int((max(values[start:end]) - min(values[start:end])) / 8)
    ticks = [(min(values[start:end]) + tick_size * n) for n in range(9)]
    plt.yticks(ticks)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend([legend])
    plt.grid(True)
    plt.show()


def plot_learning_curves(history, start=0, end=None, x_label='', y_label='', legend=''):
    values, values2 = history.history["loss"], history.history["val_loss"]
    plt.plot(values[start:end])
    plt.plot(values2[start:end])
    # tick_size = int((max(values[start:end]) - min(values[start:end])) / 8)
    # ticks = [(min(values[start:end]) + tick_size * n) for n in range(9)]
    # plt.yticks(ticks)
    plt.xlabel("epochs")
    plt.ylabel("error score[rmse]")
    plt.legend(["loss", "val_loss"])
    plt.title("Loss")
    plt.grid(True)
    plt.show()


def plot_real_and_forecasts(time_test, raw_test, forecasts, format="-", start=0, end=None, label1="Series",
                            label2="Forecast"):
    sequence_len = forecasts.shape[1]
    fig, ax = plt.subplots()
    ax.plot(time_test[start:end], raw_test[start:end], format, label=label1)
    if sequence_len == 1:
        ax.plot(time_test[start:end], forecasts[start:end], format, label=label2)
    else:
        for i in range(start, len(forecasts), sequence_len):
            off_s = i
            off_e = off_s + sequence_len + 1
            if off_e > len(forecasts):
                break
            xaxis = time_test[off_s:off_e]
            yaxis = [raw_test[off_s]] + [forecast for forecast in forecasts[i + sequence_len]]
            ax.plot(xaxis, yaxis, format, label=label2 if i == start else '', color='red')

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


def plot_score_boxplot(error_scores):
    for i in range(0, len(error_scores)):
        print('%d) Test RMSE: %.3f' % (i + 1, error_scores[i]))
    results = DataFrame()
    results['rmse'] = error_scores
    print(results.describe())
    results.boxplot()
    plt.show()


def plot_lr_by_loss(history):
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-4, 0, max(history.history["loss"])])
    plt.show()


def plot_lr_by_loss2(history):
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.semilogx(history.history["lr"], history.history["val_loss"])
    plt.axis([1e-8, 1e-4, 0, max(history.history["loss"])])
    plt.show()
