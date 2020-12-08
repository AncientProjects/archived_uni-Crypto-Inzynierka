import data_client.data_client


def main():
    client = data_client.data_client.DataClient()
    client.make_request_and_save()
    df = client.request_and_return_df()
    print(df)


if __name__ == '__main__':
    main()