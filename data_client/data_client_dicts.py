api_key = "237a4305275ed87a72eeb7ec6618b207f68a2b58b702f5c1d73a61d2166a22d3"

endpoint_dict = {
    'BTCUSD100MINUT': "https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=USD&limit=100" \
                      "&extraParams=SuperPrzewidywanie&api_key=237a4305275ed87a72eeb7ec6618b207f68a2b58b702f5c1d73a61d2166a22d3",
    'BTCUSD2000GODZIN': "https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=2000" \
                        "&extraParams=SuperPrzewidywanie&api_key=237a4305275ed87a72eeb7ec6618b207f68a2b58b702f5c1d73a61d2166a22d3",
    'BTCUSD2000GODZINGEMINI': "https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=2000" \
                              "&extraParams=SuperPrzewidywanie&e=Gemini&api_key=237a4305275ed87a72eeb7ec6618b207f68a2b58b702f5c1d73a61d2166a22d3",
    'BTCUSD2000DNIBINANCE': "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000" \
                            "&extraParams=SuperPrzewidywanie&e=Kraken&api_key=237a4305275ed87a72eeb7ec6618b207f68a2b58b702f5c1d73a61d2166a22d3"}

base_path_dict = {
    'MINUTE': "https://min-api.cryptocompare.com/data/v2/histominute",
    'HOUR': "https://min-api.cryptocompare.com/data/v2/histohour",
    'DAY': "https://min-api.cryptocompare.com/data/v2/histoday"
}

request_params_dict = {
    'BTCUSDGemini': {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': 2000,
        'e': 'Gemini',
        'extraParams': 'SuperPrzewidywanie',
        'api_key': api_key
    },
    'BTCUSDGeminiDAY': {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': 2000,
        'e': 'Gemini',
        'extraParams': 'SuperPrzewidywanie',
        'api_key': api_key
    },
    'ETHUSD': {
        'fsym': 'ETH',
        'tsym': 'USD',
        'limit': 2000,
        'extraParams': 'SuperPrzewidywanie2',
        'api_key': api_key
    },
    'payload': {
        'api_key': api_key,
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': 2000
    },
    'ETHUSDtoTs': {
        'fsym': 'ETH',
        'tsym': 'USD',
        'limit': 2000,
        'extraParams': 'SuperPrzewidywanie2',
        'api_key': api_key
    },
    'payloadtoTs': {
        'api_key': api_key,
        'fsym': 'BTC',
        'tsym': 'USD',
        'toTs': 1431907200,
        'limit': 2000
    },
    'default': {  # default - BTC-USD-2000-DAYkuj.csv
        'tryConversion': 'true',
        'e': 'CCCAGG',
        'api_key': api_key,
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': 2000
    }
}
