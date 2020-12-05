import argparse
from pandas import read_csv, DataFrame, concat, Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def load_two_datasets(training_data_path, validation_data_path):
    series = read_csv(training_data_path)
    series_val = read_csv(validation_data_path)
    return series, series_val


# def scale2(train, test):
#     scaler, train_scaled = scale_train(train)
#     test_scaled = scale_test(test)
#
#     return scaler, train_scaled, test_scaled


def scale_train(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    return scaler, scaled_train


def scale_test(test, scaler):
    return scaler.transform(test)


def prepare_data(train_raw, test_raw, window_size, sequence_len):
    scaler, train_prepared = prepate_train_data(train_raw, window_size, sequence_len)
    test_prepared = prepare_test_data(test_raw, window_size, sequence_len, scaler)
    return scaler, train_prepared, test_prepared


def prepate_train_data(raw_values, window_size, sequence_len):
    diff_values = difference2(raw_values)

    supervised = series_to_supervised(data=diff_values, n_in=window_size, n_out=sequence_len)
    supervised_values = supervised.values
    # num_of_tensors = train_size - window_size - sequence_len

    # train, test = supervised_values[:num_of_tensors], supervised_values[num_of_tensors:]
    scaler, train_scaled = scale_train(supervised_values)

    return scaler, train_scaled


def prepare_test_data(raw_values, window_size, sequence_len, scaler):
    diff_values = difference2(raw_values)
    supervised = series_to_supervised(data=diff_values, n_in=window_size, n_out=sequence_len)
    supervised_values = supervised.values
    # test_scaled = scale_test(supervised_values, scaler)
    test_scaled = scaler.transform(supervised_values)
    return test_scaled


def split_and_reshape(arr, window_size):
    x, y = split_data(arr, window_size)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    return x, y


def prepare_supervised_data_only(raw_values, window_size, sequence_len, num_of_tensors):
    raw_values = raw_values.reshape(len(raw_values), 1)

    supervised = series_to_supervised(data=raw_values, n_in=window_size, n_out=sequence_len)
    supervised_values = supervised.values

    supervised_x, supervised_y = supervised_values[:num_of_tensors], supervised_values[num_of_tensors:]
    supervised_x = supervised_x.reshape(supervised_x.shape[0], supervised_x.shape[1], 1)

    return supervised_x, supervised_y


def invert_data_transformations(test_data, test_values, scaled_forecasts, scaler):
    inverted = inv_scale(test_data[0], scaled_forecasts, scaler)

    rescaled_diff = inverted[:, test_data[0].shape[1]:]
    test_x = test_values[0]

    # last_ob - last object - czyli ostatni w wierszu X(None, window_size, 1)
    last_ob = test_x[:, -1]
    forecasts = list()
    for i in range(rescaled_diff.shape[0]):
        forecasts.append(inverse_difference(rescaled_diff[i], last_ob[i+1]))
    forecasts = np.array(forecasts)
    forecasts = forecasts.reshape(len(forecasts), scaled_forecasts.shape[1])
    return forecasts

    # for i in range(len(scaled_forecasts)):
    #     last_ob = test_data[i]
    #     inv_difference = inverse_difference(forecast=rescaled[i], last_ob=last_ob)
    #
    #     inverted.append(inv_difference)
    # return inverted


def invert_data_transformations2(real_x_values, scaled_forecasts, scaler):
    inverted = list()
    rescaled = scaler.inverse_transform(scaled_forecasts)
    for i in range(len(scaled_forecasts)):
        last_ob = real_x_values[i]
        inv_difference = inverse_difference(forecast=rescaled[i], last_ob=last_ob)

        inverted.append(inv_difference)
    return inverted


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inv_scale(test_x, forecasts, scaler):
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1])
    inverted = list()

    for i in range(test_x.shape[0]):
        new_row = [x for x in test_x[i]] + [y for y in forecasts[i]]
        new_row = np.array(new_row)
        new_row = new_row.reshape(1, len(new_row))
        inverted.append(new_row)

    inverted = np.array(inverted)
    inverted = inverted.reshape(inverted.shape[0], inverted.shape[2])
    inverted = scaler.inverse_transform(inverted)
    return inverted


def inverse_difference(forecast, last_ob):
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)

    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def split_data(series, window):
    return series[:, :window], series[:, window:]


def difference2(raw_values):
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    return diff_values

#
# def scale_train(diff_values):
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaled_values = scaler.fit_transform(diff_values)
#     return scaled_values, scaler
#
#
# def scale_test(diff_values, scaler):
#     scaled_values = scaler.transform(diff_values)
#     return scaled_values


#################################################################################################################
#
#
# def prepare_train_data(raw_values, window_size, sequence_len):
#     diff_values = difference2(raw_values)
#     scaled_values, scaler = scale_train(diff_values)
#
#     supervised = series_to_supervised(data=scaled_values, n_in=window_size, n_out=sequence_len)
#     supervised_values = supervised.values
#
#     supervised_x, supervised_y = split_data(supervised_values, window_size)
#     supervised_x = supervised_x.reshape(supervised_x.shape[0], supervised_x.shape[1], 1)
#     return scaler, supervised_x, supervised_y
#
#
# def prepare_test_data(raw_values, window_size, sequence_len, scaler):
#     diff_values = difference2(raw_values)
#     scaled_values = scale_test(diff_values, scaler)
#
#     supervised = series_to_supervised(data=scaled_values, n_in=window_size, n_out=sequence_len)
#     supervised_values = supervised.values
#
#     supervised_x, supervised_y = split_data(supervised_values, window_size)
#     supervised_x = supervised_x.reshape(supervised_x.shape[0], supervised_x.shape[1], 1)
#     return supervised_x, supervised_y
