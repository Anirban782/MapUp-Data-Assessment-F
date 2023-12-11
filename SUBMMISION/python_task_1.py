import pandas as pd

df=pd.read_csv('dataset-1.csv')

def generate_car_matrix(df):
    pivot_df = df.pivot(index='id_1', columns='id_2', values='car')
    for col in pivot_df.columns:
        pivot_df.loc[pivot_df.index == col, col] = 0
    return pivot_df

def get_type_count(df):
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    type_counts = df['car_type'].value_counts().to_dict()
    sorted_counts = dict(sorted(type_counts.items()))
    return sorted_counts

def get_bus_indexes(df):
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    bus_indexes.sort()
    return bus_indexes

def filter_routes(df):
    route_avg_truck = df.groupby('route')['truck'].mean()
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    selected_routes.sort()
    return selected_routes

def multiply_matrix(dataframe):
    modified_df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    modified_df = modified_df.round(1)
    return modified_df

multiply_matrix(generate_car_matrix(df))


def time_check(df):
    # Convert the timestamp columns to datetime objects with a specified format
    df['startTimestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    df['endTimestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    # Calculate the time difference for each record
    df['duration'] = df['endTimestamp'] - df['startTimestamp']

    # Group by ('id', 'id_2') and check if the duration covers a full 24-hour period and spans all 7 days
    result = df.groupby(['id', 'id_2']).apply(check_time_range)

    return result

def check_time_range(group):
    # Check if the duration covers a full 24-hour period
    full_24_hours = group['duration'].min() >= pd.Timedelta(hours=24)

    # Check if the timestamps span all 7 days of the week
    all_days_present = set(group['startTimestamp'].dt.dayofweek.unique()) == set(range(7))

    return pd.Series({'time_check': not (full_24_hours and all_days_present)})

# Load the dataset
df = pd.read_csv(r'dataset-2.csv')

# Apply the time_check function to the DataFrame
result_series = time_check(df)

print(result_series)