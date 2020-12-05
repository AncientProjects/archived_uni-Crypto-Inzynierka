import csv

import requests

from data_client.data_client_dicts import base_path_dict, request_params_dict


class DataClient(object):

    def __init__(self):
        self.request = None
        self.data_dictionary = {}
        self.base_path = base_path_dict['DAY']
        self.params_dict = request_params_dict['default']
        self.filename = self.set_filename()
        self.fieldnames = self.set_fieldnames()

    def set_filename(self):
        return "data/" + str(self.params_dict['fsym']) + "-" + str(self.params_dict['tsym']) + "-" + str(
            self.params_dict['limit']) + "-" + "DAYasdf.csv"

    def set_fieldnames(self):
        return ['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 'conversionType', 'conversionSymbol']

    def request_and_return_df(self):
        self.request = requests.get(self.base_path, params=self.params_dict)
        self.data_dictionary = self.request.json()['Data']
        return self.data_dictionary

    def make_request_and_save(self):
        self.request = requests.get(self.base_path, params=self.params_dict)
        self.data_dictionary = self.request.json()['Data']
        self.save_data()

    def save_data(self):
        with open(self.filename, 'w', newline='') as new_file:
            csv_writer = csv.DictWriter(new_file, self.fieldnames, delimiter=',')
            csv_writer.writeheader()
            [csv_writer.writerow(one_data) for one_data in self.data_dictionary['Data']]
