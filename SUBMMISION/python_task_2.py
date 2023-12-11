import pandas as pd

def calculate_distance_matrix(df):
    # Create an empty DataFrame for the distance matrix
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids,dtype=float)
    distance_matrix = distance_matrix.fillna(0)  # Initialize with zeros

    # Populate the distance matrix
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], float(row['distance'])
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance  # Ensure symmetry

    # Calculate cumulative distances along known routes
    for i in unique_ids:
        for j in unique_ids:
            if i != j and distance_matrix.at[i, j] == 0:
                for k in unique_ids:
                    if i != k and j != k:
                        # If distances between toll locations A to B and B to C are known,
                        # then the distance from A to C should be the sum of these distances.
                        distance_matrix.at[i, k] += distance_matrix.at[i, j] + distance_matrix.at[j, k]

    return distance_matrix

# Read the dataset
df = pd.read_csv('dataset-3.csv')

# Call the function and display the result
result = calculate_distance_matrix(df)
print("Distance Matrix:Cumulative Distances between Toll Locations")
print(result)


def unroll_distance_matrix(df):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): Distance matrix DataFrame

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to store unrolled data
    unrolled_data = []

    # Iterate through the distance matrix to extract combinations and distances
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Assuming 'result' is the distance matrix obtained from Question 1
# You can replace it with the actual variable containing the distance matrix
result = calculate_distance_matrix(df)

# Call the unroll_distance_matrix function and display the result
unrolled_result = unroll_distance_matrix(result)
print("Unroll Distance Matrix: Expanded Combinations with Distances")
print(unrolled_result)



def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): Reference ID for calculating the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter DataFrame based on the reference_id
    reference_data = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    reference_average_distance = reference_data['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold_range = 0.1 * reference_average_distance

    # Find IDs within the threshold range
    ids_within_threshold = df[
        (df['id_start'] != reference_id) &
        (df['distance'] >= (reference_average_distance - threshold_range)) &
        (df['distance'] <= (reference_average_distance + threshold_range))
    ]['id_start'].unique()

    # Create a DataFrame with the result
    result_df = pd.DataFrame({'id_start': ids_within_threshold})

    return result_df

# Assuming 'unrolled_result' is the DataFrame obtained from Question 2
# You can replace it with the actual variable containing the unrolled DataFrame
unrolled_result = unroll_distance_matrix(result)

# Call the find_ids_within_ten_percentage_threshold function and display the result for a reference_id
reference_id = 1001420  # Replace with the desired reference ID
result_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_result, reference_id)
print("Find IDs within Ten Percentage Threshold: Identify Toll Locations with Similar Average Distances")
print(result_within_threshold)


def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with toll rates for each vehicle type.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Assuming 'unrolled_result' is the DataFrame obtained from Question 2
# You can replace it with the actual variable containing the unrolled DataFrame
unrolled_result = unroll_distance_matrix(result)

# Call the calculate_toll_rate function and display the result
result_with_toll_rate = calculate_toll_rate(unrolled_result)
print("Find IDs within Ten Percentage Threshold: Identify Toll Locations with Similar Average Distances")
print(result_with_toll_rate)


from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define time ranges and discount factors
    weekday_time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]
    weekend_discount_factor = 0.7

    # Initialize an empty list to store DataFrames for each time range
    dfs = []

    # Iterate through the DataFrame and create DataFrames for each time range
    for _, row in df.iterrows():
        id_start, id_end = row['id_start'], row['id_end']

        # Iterate through days of the week
        for i in range(7):
            current_day = datetime.strptime('2023-01-01', '%Y-%m-%d').date() + timedelta(days=i)
            start_datetime = datetime.combine(current_day, time(0, 0, 0))
            end_datetime = datetime.combine(current_day, time(23, 59, 59))

            # Iterate through time ranges and apply discount factors
            for start_time, end_time, discount_factor in weekday_time_ranges:
                current_start_datetime = datetime.combine(current_day, start_time)
                current_end_datetime = datetime.combine(current_day, end_time)

                if start_datetime <= current_start_datetime <= end_datetime or \
                   start_datetime <= current_end_datetime <= end_datetime:
                    # Create a copy of the row and update it
                    new_row = row.copy()
                    new_row['start_day'] = current_day.strftime('%A')  # Get the day name
                    new_row['start_time'] = start_time
                    new_row['end_day'] = current_day.strftime('%A')  # Get the day name
                    new_row['end_time'] = end_time

                    # Update vehicle columns based on the discount factor
                    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                        new_row[vehicle_type] *= discount_factor

                    # Append the updated row to the list
                    dfs.append(pd.DataFrame([new_row]))

    # Concatenate the DataFrames for each time range
    result_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns as per the desired order
    column_order = ['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']
    result_df = result_df[column_order]


    return result_df

# Assuming 'result_with_toll_rate' is the DataFrame obtained from Question 4
# You can replace it with the actual variable containing the DataFrame with toll rates
result_with_toll_rate = calculate_toll_rate(unrolled_result)

# Call the calculate_time_based_toll_rates function and display the result
result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)
print(result_with_time_based_toll_rates)
