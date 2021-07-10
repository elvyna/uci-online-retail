from datetime import date
import pandas as pd

current_date = pd.to_datetime(date.today())
current_month_end_date = current_date.to_period("M").to_timestamp("M")

def ensure_complete_date(
    df,
    date_column,
    latest_date_sequence=None,
    value_columns=["quantity"],
    grouping_columns=None,
    is_date_index=False,
    date_freq="D",
    impute_missing=True,
    impute_method="linear",
    impute_method_param=2,
):
    """
    Ensure all dates exist in the dataset.
    If there are any missing dates, a full period will be generated.
    If impute_missing is True, missing values will be imputed, by default via linear interpolation.
    If impute_missing is False, missing values will be filled by 0.
    If grouping_columns are specified, we expect to get a dataframe that contains only one group as input.
    i.e., please use this function in conjunction with df.iterrows() -- loop over each group in your main function
    Check pandas.DataFrame.interpolate() for more reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
    Return a dataframe with complete dates.
    """
    if is_date_index:
        date_min = df.index.min()
        date_max = df.index.max()
    else:
        date_min = df[date_column].min()
        date_max = df[date_column].max()
        ## ensure df[date_column] has the same data type as date_sequence
        df[date_column] = pd.to_datetime(df[date_column])

    ## if grouping_columns is specified
    ## take the max date of all groups (to ensure each group has full records until the latest date)
    if latest_date_sequence is not None:
        date_max = max(date_max, latest_date_sequence)

    date_sequence = pd.date_range(start=date_min, end=date_max, freq=date_freq)

    ## check if any dates are missing from the generated date sequence
    if is_date_index:
        missing_date_count = (~date_sequence.isin(df.index)).sum()
    else:
        missing_date_count = (~date_sequence.isin(df[date_column])).sum()

    ## impute value, can be moved as another function?
    if missing_date_count > 0:
        print("There are missing dates in the dataframe. Full period is generated.")
        df_full = pd.DataFrame(data={"period_date": date_sequence})
        if grouping_columns is None:
            if is_date_index:
                df_full = df_full.merge(
                    df, how="left", left_on="period_date", right_index=True
                )
            else:
                df_full = df_full.merge(
                    df, how="inner", left_on="period_date", right_on=date_column
                )
        else:
            df = df.reset_index()
            if is_date_index:
                df_full = df_full.merge(
                    df[[date_column] + grouping_columns + value_columns],
                    how="left",
                    left_on="period_date",
                    right_index=True,
                )
            else:
                df_full = df_full.merge(
                    df[[date_column] + grouping_columns + value_columns],
                    how="left",
                    left_on="period_date",
                    right_on=date_column,
                )
                
            ## fill NA in grouping column
            for group_col in grouping_columns:
                fill_value = df[group_col].unique()[0]
                ## ensure the filled value is not na
                assert fill_value is not np.nan
                df_full[group_col].fillna(fill_value, inplace=True)
        
        ## impute value, either using interpolation or zero
        if impute_missing:
            for col in value_columns:
                if impute_method in ["polynomial", "spline"]:
                    df_full[["period_date", col]] = df_full[
                        ["period_date", col]
                    ].interpolate(method=impute_method, order=impute_method_param)
                else:
                    df_full[["period_date", col]] = df_full[
                        ["period_date", col]
                    ].interpolate(method=impute_method)
        else:
            for col in value_columns:
                df_full[col].fillna(0, inplace=True)
        return df_full
    else:
        return df


def generate_complete_records(
    df,
    main_grouping_columns=["stock_code"],
    date_columns=["invoice_date_week"],
    main_date_column="invoice_date_week",
    value_columns=["quantity"],
    date_freq="M",
    latest_date_sequence=None,
    impute_missing: str = False,
    impute_method: str = "linear",
    impute_method_param: int = 2,
):
    """
    Ensure all records (based on grouping columns) have complete records from their first date until the most recent date.
    """
    ## ensure grouping columns exist in the dm
    assert all(col in df.columns for col in main_grouping_columns)
    assert (
        main_grouping_columns == ["stock_code"]
    )
    assert main_date_column in date_columns

    ## ensure all dates exist for each item_code and segment, starting from the first transaction date
    ## using df.iterrows() may not be the best solution
    ## have tried using df.groupby(grouping_columns).apply(ensure_complete_date, *kwargs)
    ## but the operation is executed per row, not per grouped df

    df_unique_group_columns = df[main_grouping_columns].drop_duplicates()
    df_full = pd.DataFrame()

    for idx, row in df_unique_group_columns.iterrows():
        ## if the grouping columns are only ['item_code'],
        ## the logic for mask rows are different
        mask = df["stock_code"] == row["stock_code"]

        df_masked = df[mask].copy()
        if latest_date_sequence is None:
            max_date = df[main_date_column].max()
        else:
            max_date = current_month_end_date

        df_temp = ensure_complete_date(
            df_masked,
            date_column=main_date_column,
            value_columns=value_columns,
            grouping_columns=main_grouping_columns,
            ## to ensure all records are generated until the latest date
            latest_date_sequence=max_date,
            date_freq=date_freq,
            impute_missing=impute_missing,
            impute_method=impute_method,
            impute_method_param=impute_method_param,
        )
        df_full = pd.concat([df_full, df_temp])

    ## remove additional columns that are generated from ensure_complete_date()
    ## it is added only if there are missing dates
    if "period_date" in df_full.columns:
        conditions = [df_full["period_date"].isnull()]
        choices = [df_full[main_date_column]]
        df_full["period_date"] = np.select(
            conditions, choices, default=df_full["period_date"]
        )

        df_full = df_full.drop(labels=date_columns, axis=1).rename(
            columns={"period_date": main_date_column}
        )
    
    return df_full.reset_index(drop=True)