from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    i = 0
    s = len(lst)
    while i < s:
        left = i 
        right = min(i + n - 1, s-1)

        while left < right:
            lst[left], lst[right] = lst[right], lst[left]
            left += 1
            right -= 1

        i += n

    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    d = {}
    for i in lst:
        d[len(i)] = d.get(len(i), []) + [i]
        
    dict = {}
    for key in sorted(d):
        dict[key] = d[key]
    
    return dict

def flatten_dict(nested_dict: Dict, sep: str = '.', prefix = '') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    dict = {}
    for key, value in nested_dict.items():
        if type(value) == dict:
            dict.update(flatten_dict(value, sep, prefix + key + sep))
        elif type(value) == list:
            for i in range(len(value)):
                dict.update(flatten_dict(value[i], sep, prefix + key + '[' + str(i) + ']' + sep))

        else:
            dict[prefix + key] = value
    return dict

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    res = []
    def permutations(num, ans):
        if len(num) == 0:
            if ans not in res:
                res.append(ans)   
            return
        
        for i in range(len(num)):
            c = num[i]
            rnum = num[:i] + num[i+1:] 
            permutations(rnum,ans + [c])

    permutations(nums, [])
    return res

import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'

    dates = re.findall(pattern, text)

    return dates

import polyline
import pandas as pd
from haversine import haversine

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    l = polyline.decode(polyline_str)
    df = pd.DataFrame(l, columns=['latitude', 'longitude'])
    df['distance'] = [0.0] * len(df)

    for i in range(1,len(df)):
        previous_row = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        current_row = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(current_row, previous_row, unit='m')
    
    return pd.Dataframe(df)


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    def rotate_matrix_90_clockwise(matrix):
        rotated_matrix = [list(row)[::-1] for row in zip(*matrix)]
        return rotated_matrix

    def sum_row_col_excluding_self(matrix):
        n = len(matrix)
        m = len(matrix[0])
        
        result = [[0] * m for i in range(n)]
        
        row_sums = [sum(row) for row in matrix]
        col_sums = [sum(col) for col in zip(*matrix)]
        
        # Calculate the sum of row and column excluding the element itself
        for i in range(n):
            for j in range(m):
                result[i][j] = (row_sums[i] + col_sums[j] - 2 * matrix[i][j])
    
        return result
    
    rotated_matrix = rotate_matrix_90_clockwise(matrix)
    result_matrix = sum_row_col_excluding_self(rotated_matrix)
    return result_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time
    grouped = df.groupby(['id', 'id_2'])

    results = pd.Series(dtype=bool)

    for name, group in grouped:
        if (group['startTime'].min() <= pd.to_datetime('00:00:00', format='%H:%M:%S').time() and
            group['endTime'].max() >= pd.to_datetime('23:59:59', format='%H:%M:%S').time() and
            set(group['startDay']).union(set(group['endDay'])) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])):
            results[name] = False
        else:   
            results[name] = True

    return results
