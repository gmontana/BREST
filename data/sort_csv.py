#!/usr/bin/env python3
"""
Script to sort a CSV file according to a custom ranking of the 'ViewPosition' and 'ImageLaterality'
fields, grouped by 'ClientID' and 'EpisodeID'.

The script performs the following:
  1. Loads the input CSV file.
  2. Defines a custom ranking order:
       - ('CC', 'L'): rank 1
       - ('CC', 'R'): rank 2
       - ('MLO', 'L'): rank 3
       - ('MLO', 'R'): rank 4
  3. Adds a temporary 'Rank' column to each row based on the custom order.
  4. Sorts the DataFrame by 'ClientID', 'EpisodeID', and the computed 'Rank'.
  5. Drops the temporary 'Rank' column.
  6. Saves the sorted DataFrame back to a CSV file.

Usage Example:
    python sort_csv.py --input ../CSVs/initial_test_metadata.csv --output ../CSVs/sorted_test_metadata.csv
"""

import argparse
import pandas as pd

def sort_csv(input_file: str, output_file: str) -> None:
    """
    Sorts the CSV file based on a custom ranking order for 'ViewPosition' and 'ImageLaterality',
    grouping by 'ClientID' and 'EpisodeID'.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path where the sorted CSV will be saved.
    """
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)

    # Define the custom ranking order for ViewPosition and ImageLaterality
    custom_order = {
        ('CC', 'L'): 1,
        ('CC', 'R'): 2,
        ('MLO', 'L'): 3,
        ('MLO', 'R'): 4
    }

    # Add a 'Rank' column based on the custom order.
    # For any row that does not match a key in the custom order, use infinity (float('inf')) so that
    # these rows appear at the end after sorting.
    data['Rank'] = data.apply(
        lambda row: custom_order.get((row['ViewPosition'], row['ImageLaterality']), float('inf')),
        axis=1
    )

    # Sort the data by 'ClientID', 'EpisodeID', and the computed 'Rank'
    sorted_data = data.sort_values(by=['ClientID', 'EpisodeID', 'Rank']).reset_index(drop=True)

    # Remove the temporary 'Rank' column
    sorted_data = sorted_data.drop(columns=['Rank'])

    # Save the sorted data to the output CSV file.
    print(f"Saving sorted data to {output_file}...")
    sorted_data.to_csv(output_file, index=False)
    print("Sorting complete.")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Sort a CSV file according to custom ranking based on ViewPosition and ImageLaterality."
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help="Path to the input CSV file (e.g. ../CSVs/Oxford_CAD_1to3_V4.csv)."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path to save the sorted CSV file (e.g. ../CSVs/Oxford_CAD_1to3_V4_sorted.csv)."
    )
    args = parser.parse_args()

    # Perform the CSV sorting process
    sort_csv(args.input, args.output)

if __name__ == "__main__":
    main()
