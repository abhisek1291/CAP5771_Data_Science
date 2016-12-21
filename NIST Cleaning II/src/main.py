import sys

import method1
import method2
import method3
from data import read_files
import pandas as pd


def main(argv):
    path = argv[1]
    print(path)
    df, result, flow_columns, prob_columns = read_files(path)
    dataset_number = path[-4:]

    # method 1
    correct_values, nan_values = setup_regression_data(result, flow_columns, prob_columns)
    regression_df = method1_regression(correct_values, flow_columns, prob_columns)

    # method 2
    neighbors_predict_df = method2.predict_flow_by_detector(result, flow_columns)

    # method 3
    method3_df = method3.predict_flow(result)

    print('\nmethod III finished. merging data...')

    joined_df = pd.merge(pd.merge(regression_df, neighbors_predict_df, on='index_col'), method3_df, on='index_col')
    joined_df = joined_df.drop_duplicates(keep="first")

    joined_df['total_confidence'] = joined_df.apply(lambda row: append_calculated_confidence(row), axis=1)
    output_results_to_file(dataset_number, joined_df)


index_to_header = {
    1: 'confidence_x',
    2: 'confidence_y',
    3: 'confidence'
}

index_to_flow_header = {
    1: '',
    2: 'ed_x',
    3: 'ed_y'
}


def output_results_to_file(dataset_number, joined_df):
    print('\ngenerating output file')
    # joined_df = pd.read_csv('/Users/abhisek/Desktop/final_joined1160.csv', delimiter="\t", error_bad_lines=False)
    joined_df = joined_df.sort_values("timestamp")
    final_output = pd.pivot_table(joined_df, values=['total_confidence'], index='timestamp', columns='detector')
    final_output = final_output.reset_index(drop=True)
    final_output = final_output.xs('total_confidence', axis=1, drop_level=True)
    final_output.to_csv(dataset_number + '.flow.txt', sep='\t', header=False, index=False)


def append_calculated_confidence(row):
    total_confidence = 0
    sum_confidence = row["confidence_x"] + row["confidence_y"] + row["confidence"]
    for x in range(1, 4, 1):
        header_name = index_to_header[x]
        confidence = row[header_name]
        if sum_confidence == 0:
            total_confidence = 0
        else:
            total_confidence = total_confidence + row["flow_predict" + index_to_flow_header[x]] * (confidence / sum_confidence)

    return total_confidence


def method1_regression(df, flow_columns, prob_columns):
    if len(flow_columns) == 1:
        df[flow_columns[0]+'_predicted'] = df[flow_columns[0]]
        regression_df = method1.merge_columns_regression(df, flow_columns, prob_columns)
    else:
        flow_cols_set = set(flow_columns)
        for i in range(0, len(flow_columns)):
            current_column = [flow_columns[i]]
            other_columns = list(flow_cols_set - set(current_column))

            regression_df = method1.linear_reg(df, other_columns, current_column[0] + '_predicted', current_column[0])
        regression_df = method1.merge_columns_regression(regression_df, flow_columns, prob_columns)
    return regression_df


def setup_regression_data(result, flow_columns, prob_columns):
    result = result.reset_index(drop=True)
    result['index_col'] = result.index
    table = pd.pivot_table(result, values=['flow', 'probability', 'index_col'], index='timestamp', columns='detector')
    f_cols = []
    p_cols = []
    i_cols = []
    for i in range(1, len(flow_columns) + 1, 1):
        f_cols.append('flow' + str(i))
        p_cols.append('prob' + str(i))
        i_cols.append('idx' + str(i))

    columns = []
    columns.extend(f_cols + p_cols + i_cols)
    print(columns)
    table.columns = columns
    table = table.reset_index()
    nan_condition = False
    non_nan_condition = True
    # print(table.columns
    for col in f_cols:
        nan_condition |= (pd.isnull(table[col]) == True)
        non_nan_condition &= (pd.isnull(table[col]) == False)

    nan_values = table[nan_condition]
    table = table[non_nan_condition]

    return table, nan_values


if __name__ == "__main__":
    main(sys.argv)
