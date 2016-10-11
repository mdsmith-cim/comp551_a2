import pandas as pd


def export(data, filename='export.csv'):

    df = pd.DataFrame(data, columns=['category'])

    df.to_csv(filename, index_label='id')