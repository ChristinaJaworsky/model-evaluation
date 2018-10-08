# Save Model Using Pickle
import pandas as pd
import os
import pickle
import datetime
import numpy as np


# Parses a csv of the form `customer_id | event_timestamp | event_value` and turns it into a dataframe
#   with one row per customer per day on which one or more events occured. This dataframe has two
#   synthesized columns: one with the sum of event values on each day, and another with the event counts on each day.
def read_event_stream_data_set(csv, event_name='purchase'):

    event_value = '%s_value' % event_name

    # Parse csv and rename columns
    names = ['customer_id', 'timestamp', event_value]
    df = pd.read_csv(csv)
    df.columns = names


    # Convert timestamp string and create a column with just the date
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.assign(date=df['timestamp'].apply(lambda x: np.datetime64(x.date())))

    # Get key customer-level aggregates and add as columns for later transformation
    daily_aggregate = {event_value:['sum','count']}
    df_rollup = df.groupby(['customer_id', 'date']).agg(daily_aggregate).reset_index().sort_index()

    # Rename columns for cleanliness
    df_rollup.columns = ['customer_id', 'date', ('total_%s_on_day' % event_value), ('%s_count_on_day' % event_name)]

    return df_rollup


# Given a list of event names, this will return the generated column names for each event name.
def get_event_name_columns(event_names=['purchase']):
    event_name_columns = []
    for event_name in event_names:
        event_name_columns.append('total_%s_value_on_day' % event_name)
        event_name_columns.append('%s_count_on_day' % event_name)
    return event_name_columns


# This takes the summarized dataframe from `read_event_stream_data_set` and turns it into a time series,
#   with one row per customer per day. The first date for a customer is the first day on which they had an event.
#   All customers have the same final day, and it is the max event date across all customers.
def generate_time_series(df, event_names=['purchase']):

    event_name_columns = get_event_name_columns(event_names)

    # Get meta-level metrics for lower and upper bounds of a future time-series
    min_date = df['date'].min()
    max_date = df['date'].max()

    # Create a bare dataframe of dates
    dates = pd.date_range(start=min_date, end=max_date).to_frame()
    dates.columns = ['date']

    # Get the first purchase of each customer so we know when to start the time series for each.
    aggregates = {'date':['min']}
    df_summarized = df.reset_index().groupby(['customer_id']).agg(aggregates).reset_index()

    df_summarized.columns = ['customer_id', 'first_event_timestamp']

    merged_df = df.merge(df_summarized, how='inner', on='customer_id').set_index(['customer_id', 'date']).sort_index()


    # Take the cartesian product to create a time series
    customer_ids = pd.DataFrame({'customer_id':df_summarized.customer_id, 'first_event_timestamp':df_summarized.first_event_timestamp})
    # Create temporary keys to join on for our cross product
    customer_ids['key'] = 0
    dates['key'] = 0
    df_time_series = pd.merge(dates, customer_ids, on='key', how='left')
    # Remove any customer/date rows that occured before the customer's first purchase
    df_time_series = df_time_series[df_time_series.date >= df_time_series.first_event_timestamp]
    # Drop the temporary key column now that it is no longer needed
    df_time_series.drop(['key'], 1, inplace=True)
    # Set the combination of the customer_id and date as the index of this dataframe
    df_time_series.set_index(['customer_id', 'date'], inplace=True)


    # Add key day-level metrics
    df_time_series = pd.merge(df_time_series, merged_df[event_name_columns], on=['customer_id', 'date'], how='left').sort_index()
    # Set default values for each column
    default_values = {}
    for event_column_name in event_name_columns:
        default_values[event_column_name] = 0.0

    # Default these metrics to 0.0
    df_time_series.fillna(value=default_values, inplace=True)

    # Return the time-series dataframe. This now has one row per customer per day since (and including) the first day on which they made a purchase
    return df_time_series


# Adds two columns to a time-series dataframe. One for the rolling x day sum of event values and another for the rolling x day event counts.
def add_rolling_x_day_sums(df, event_name, x_days):

    event_name_columns = get_event_name_columns([event_name])

    df_lxd = df[event_name_columns].groupby(level=[0], as_index=False).rolling(x_days).sum()

    value_lxd = '%s_value_l%sd' % (event_name, x_days)
    count_lxd = '%s_count_l%sd' % (event_name, x_days)

    df_lxd.columns = [value_lxd, count_lxd]

    merged = pd.merge(df, df_lxd[[value_lxd, count_lxd]], on=['customer_id', 'date'], how='inner')

    return merged


# This function adds multiple sets of rolling metrics to our time-series'ed dataframe. These columns will be used as model inputs (i.e. factors).
#   This is where one might add additional factors to try in training a model if we want to get more sophisticated.
def generate_metrics_for_use_as_factors(df, event_names=['purchase']):
    for event_name in event_names:
        df = add_rolling_x_day_sums(df, event_name, 1*30)
        df = add_rolling_x_day_sums(df, event_name, 3*30)
        df = add_rolling_x_day_sums(df, event_name, 6*30)
    return df


# This will generate the "target" column – the one that we want to use our model to predict.
def determine_if_event_occured_in_furture_x_days(df, event_name, x_days):
    # This line reverses the index and creates a forward-looking x-day window to sum across
    forward_looking_x_day_window = df[['%s_count_on_day' % event_name]].iloc[::-1].groupby(level=[0], as_index=False).shift(1).rolling(x_days, min_periods=0).sum()

    total_event_count_future_x_days = 'total_%s_count_future_x_days' % event_name
    forward_looking_x_day_window.columns = [total_event_count_future_x_days]

    # We then look to see if the sum is greater than 1 to create our boolean success metric (1 for true, 0 for false).
    had_event_in_future_x_days = 'had_%s_in_future_x_days' % event_name
    forward_looking_x_day_window[had_event_in_future_x_days] = forward_looking_x_day_window[total_event_count_future_x_days].apply(lambda x: (1 if x > 0.0 else 0))

    # Merge the result back into the rest of the original data frame and return
    merged = pd.merge(df, forward_looking_x_day_window[[had_event_in_future_x_days]], on=['customer_id', 'date'], how='inner')
    return merged


# This adds additional columns that help to determine if a given customer/date row is the right "Age" to be useful in our model.
#   Dates that are too close to a customer's first event haven't baked enough to have meaningful rolling metrics. Likewise, dates that
#   are too old, aka within 6 months of today, haven't had enough time to posibily have a purchase within a future 6 months.
def determine_age_viability_for_model(df, min_age_days, max_age_days):
    max_date = df.index.get_level_values('date').max()
    df['old_enough_for_model'] = ((df.index.get_level_values('date') - df.first_event_timestamp).dt.days + 1) >= min_age_days
    df['young_enough_for_model'] = ((max_date - df.index.get_level_values('date')).days) >= max_age_days

    return df


# Given the columns generated in `determine_age_viability_for_model()`, this filters our dataframe to only include customer/date
#   rows that are within the appropriate relative age window.
def get_eligible_training_set(df, event_names=['purchase']):

    filtered_df = df[(df.old_enough_for_model == True) & (df.young_enough_for_model == True)]

    # Drop columns that aren't needed for model training and usage
    cols_to_drop = ['first_event_timestamp', 'old_enough_for_model', 'young_enough_for_model']
    cols_to_drop.extend(get_event_name_columns(event_names))
    filtered_df = filtered_df.drop(cols_to_drop, 1)

    return filtered_df


# This breaks down the processing of a csv into its time-series'ed dataframe into discreet steps.
def prep_csv_for_model_evaluation(csv, pickle_file=None):

    ########################## STEP 1: CSV Parsing and Day-Level Summarization #########################
    # Read in the event stream data
    # Note: If we, in the future, wanted to account for mutliple csvs, each with different event types,
    #   we'd want to execute this function on them as well and join the resulting dataframes together.
    df = read_event_stream_data_set(csv, event_name='purchase')


    ############################## STEP 2: Generate Time Series Dataframe ##############################
    # First transform the event stream into time-series data

    df = generate_time_series(df, event_names=['purchase'])


    ################################## STEP 3: Compute Rolling Metrics #################################
    # Synthesize rolling metrics from the time series data for use as inputs into a model
    #   This is one place where you could experiment more to create a better model.

    df = generate_metrics_for_use_as_factors(df)


    ################################# STEP 4: Generate "Target" Column #################################
    # Add one last column that is the actual categorization that we want to predict.
    #   Did the customer purchase in the future 6 months as of that day?
    rolling_window_into_future_in_days =  6*30 # 6 months, assuming 30 days per month
    df = determine_if_event_occured_in_furture_x_days(df, 'purchase', rolling_window_into_future_in_days)


    ################### STEP 5: Filter Out Ineligibile Dates & Drop Unneeded Columns ##################
    # This is the final step. Our final dataframe only includes customer/date rows that are old
    #   enough and yound enough for the model. It also drops any columns that we don't want to
    #   use as factors or as the categorization.


    max_rolling_window_span_in_days = 6*30 # 6 months, assuming 30 days per month
    df = determine_age_viability_for_model(df, max_rolling_window_span_in_days, rolling_window_into_future_in_days)
    df = get_eligible_training_set(df)

    # Finally, we want to save our resulting dataframe to a pickle file so that we don't have to generate it again unnecessarily.
    if pickle_file:
        df.to_pickle(pickle_file)
        print "Saved dataframe to %s." % pickle_file

    return df


# This tries to load a dataframe from a pickle file, but falls back on
#   recreating it from the csv if none is found
def load_dataframe(pickle_file, csv=None, overwrite_pickle=False):

    if overwrite_pickle:
        return prep_csv_for_model_evaluation(csv, pickle_file)

    try:
        df = pd.read_pickle(pickle_file)
        print "Successfully loaded dataframe from pickle file at %s\n" % pickle_file
        return df
    except (IOError, AttributeError) as e:
        if csv:
            print "Could not load pickle file at %s.\n   Generating new dataframe from csv at %s.\n" % (pickle_file, csv)
            return prep_csv_for_model_evaluation(csv, pickle_file)
        else:
            print "Could not find pickle file at %s\n   Attempted to generate dataframe from a csv, but no csv was specified.\n" % pickle_file
            return None


if __name__ == "__main__":
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)


    pickle_file = './notebook/stored_dataframes/TransactionsCompany1.pkl'
    csv = './notebook/stored_csvs/TransactionsCompany1.csv'
    df = load_dataframe(pickle_file, csv=csv, overwrite_pickle=False)
    print df
