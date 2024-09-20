# import the necessary libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

from pdb import set_trace as bp

class CustomDataset(Dataset):
    def __init__(self, data, binary_cols):

        if isinstance(data, pd.DataFrame):
            all_cols = data.columns
            real_cols = [col for col in all_cols if col not in binary_cols]
            binary_data = data[binary_cols].values
            real_data = data[real_cols].values
        else:
            all_cols = np.arange(data.shape[1])
            real_cols = list(set(all_cols) - set(binary_cols))
            binary_data = data[:, binary_cols]
            real_data = data[:, real_cols]

        # Create the mask before filling NaNs
        self.binary_mask = (~np.isnan(binary_data)).astype(float)  # Mask of available data (1 where data is present, 0 where it's NaN)
        self.real_mask = (~np.isnan(real_data)).astype(float)  # Mask of available data (1 where data is present, 0 where it's NaN)

        self.binary_data = np.nan_to_num(binary_data, nan=0.5)  # Replace NaNs with -1, after the mask is created
        self.real_data = np.nan_to_num(real_data, nan=0.5)  # Replace NaNs with -1, after the mask is created

        # print the overall shape of the data
        print("BINARY data report...")
        print(f"Data shape: {self.binary_data.shape}")

        # print the fraction of columns that have at least one NaN value
        print(f"Fraction of columns with NaNs: {np.mean(np.isnan(binary_data).any(axis=0))}")

        # print out each binary column that has missing values
        print("Columns with missing values:")
        for i, col in enumerate(binary_cols):
            if np.isnan(binary_data[:, i]).any():
                print(f"  - {col}")

        # print the fraction of rows that have at least one NaN value
        print(f"Fraction of rows with NaNs: {np.mean(np.isnan(binary_data).any(axis=1))}")

        # print the fraction of missing values in the dataset
        print(f"Overall fraction of missing values: {np.mean(np.isnan(binary_data))}")


        # print the overall shape of the data
        print("REAL data report...")
        print(f"Data shape: {self.real_data.shape}")

        # print the fraction of columns that have at least one NaN value
        print(f"Fraction of columns with NaNs: {np.mean(np.isnan(real_data).any(axis=0))}")

        # print the fraction of rows that have at least one NaN value
        print(f"Fraction of rows with NaNs: {np.mean(np.isnan(real_data).any(axis=1))}")

        # print the fraction of missing values in the dataset
        print(f"Overall fraction of missing values: {np.mean(np.isnan(real_data))}")

    def __len__(self):
        return len(self.binary_data)

    def __getitem__(self, idx):
        # Return binary and real data, along with their respective masks
        x_binary = torch.tensor(self.binary_data[idx], dtype=torch.float32)
        x_real = torch.tensor(self.real_data[idx], dtype=torch.float32)
        binary_mask = torch.tensor(self.binary_mask[idx], dtype=torch.float32)
        real_mask = torch.tensor(self.real_mask[idx], dtype=torch.float32)

        return x_binary, x_real, binary_mask, real_mask


class CustomMinMaxScalerWithGroups:
    def __init__(self, feature_groups, feature_range=(0, 1)):
        """
        feature_groups: List of lists, where each sublist contains column names of features that should be treated as a group.
        Example: [['feature1', 'feature2'], ['feature3', 'feature4']]
        feature_range: Desired range of transformed data (default: (0, 1))
        """
        self.feature_groups = feature_groups
        self.feature_range = feature_range
        self.group_mins = {}
        self.group_maxs = {}
        self.columns_ = None  # To store column names when the input is a Pandas DataFrame
        self.feature_groups_indices = []  # To store column indices for each group

    # Fit the scaler by computing the min and max for each group of features (using column names)
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()

            # Convert feature groups from column names to indices
            self.feature_groups_indices = [
                [self.columns_.index(col) for col in group] for group in self.feature_groups
            ]
            X = X.values  # Convert to NumPy array for processing

        # Compute min and max for each group of features
        for i, group_indices in enumerate(self.feature_groups_indices):
            group_data = X[:, group_indices]
            self.group_mins[i] = np.nanmin(group_data)
            self.group_maxs[i] = np.nanmax(group_data)
        return self

    # Transform the data by applying the min-max scaling to each group of features
    def transform(self, X, return_df=False):
        if not self.group_mins or not self.group_maxs:
            raise Exception("Scaler has not been fitted yet!")

        is_df = isinstance(X, pd.DataFrame)

        if is_df:
            X = X.values  # Convert DataFrame to NumPy array for transformation

        X_scaled = np.copy(X)
        min_r, max_r = self.feature_range  # Desired range of transformation

        for i, group_indices in enumerate(self.feature_groups_indices):
            group_data = X[:, group_indices]
            group_min = self.group_mins[i]
            group_max = self.group_maxs[i]

            # Standard min-max scaling to [0, 1]
            X_scaled[:, group_indices] = (group_data - group_min) / (group_max - group_min)

            # Rescale to desired feature_range (min_r, max_r)
            X_scaled[:, group_indices] = X_scaled[:, group_indices] * (max_r - min_r) + min_r

        if return_df and self.columns_:
            return pd.DataFrame(X_scaled, columns=self.columns_)

        return X_scaled

    # Fit and transform the data in one step
    def fit_transform(self, X, return_df=False):
        """Combines the fit and transform methods."""
        self.fit(X)
        return self.transform(X, return_df=return_df)

    # Reverse the scaling from transformed data back to the original scale
    def inverse_transform(self, X_scaled, return_df=False):
        if not self.group_mins or not self.group_maxs:
            raise Exception("Scaler has not been fitted yet!")

        is_df = isinstance(X_scaled, pd.DataFrame)

        if is_df:
            X_scaled = X_scaled.values  # Convert DataFrame to NumPy array for inverse transformation

        X_original = np.copy(X_scaled)
        min_r, max_r = self.feature_range  # Desired range of transformation

        for i, group_indices in enumerate(self.feature_groups_indices):
            group_data = X_scaled[:, group_indices]
            group_min = self.group_mins[i]
            group_max = self.group_maxs[i]

            # Rescale from [min_r, max_r] back to [0, 1]
            X_original[:, group_indices] = (group_data - min_r) / (max_r - min_r)

            # Reverse scaling from [0, 1] back to the original scale
            X_original[:, group_indices] = X_original[:, group_indices] * (group_max - group_min) + group_min

        if return_df and self.columns_:
            return pd.DataFrame(X_original, columns=self.columns_)

        return X_original


# 3. Data preprocessing and split
def prepare_data(filename="data/combined_data_subset_1_percent",
                 batch_size=None,
                 shuffle_train=True):
    # fname = "data/combined_data_subset_1_percent.csv"
    # fname = "data/combined_data_subset_10_percent.csv"
    data = pd.read_csv(filename, low_memory=False)

    # Drop columns "suid" and "eir" columns
    data = data.drop(columns=["suid", "eir"])

    # plot_missingness(data)

    # Identify columns that have only 1 unique value other than NaN, print them and drop
    single_value_cols = data.columns[data.nunique(dropna=True) == 1]

    # print all droppable columns even if it is a long list
    print("Droppable columns:")
    for col in single_value_cols:
        print(col)

    data = data.drop(columns=single_value_cols)
    print(f"Dropped {len(single_value_cols)} columns.")

    # Split into train/test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # First get a list of all boolean columns by checking if all values are 0 or 1
    boolean_cols = [
        col
        for col in train_data.columns
        if set(train_data[col].dropna().unique()) == {0, 1}
    ]

    print(f"Found {len(boolean_cols)} non-degenerate boolean columns.")

    # Next, get a list of all columns that are bounded between 0 and 1 (but are not boolean)
    zero_to_one_cols = [
        col
        for col in train_data.columns
        if train_data[col].dropna().between(0, 1).all()
    ]

    # Now get all remaining columns
    remaining_cols = [
        col
        for col in train_data.columns
        if col not in boolean_cols and col not in zero_to_one_cols
    ]

    # Assert that none of the remaining columns have negative values
    assert all(
        train_data[remaining_cols].min() >= 0
    ), "Some columns have negative values!"

    # start with susceptibility columns that need grouping
    grouped_cols = {
        # MSP
        "m_MSP_antibodies_{i}_present": [
            f"m_MSP_antibodies_{i}_present" for i in range(32)
        ],
        "m_MSP_antibodies_{i}_m_antibody_capacity": [
            f"m_MSP_antibodies_{i}_m_antibody_capacity" for i in range(32)
        ],
        "m_MSP_antibodies_{i}_m_antibody_concentration": [
            f"m_MSP_antibodies_{i}_m_antibody_concentration" for i in range(32)
        ],
        # minor
        "m_PfEMP1_minor_antibodies_{i}_present": [
            f"m_PfEMP1_minor_antibodies_{i}_present" for i in range(385)
        ],
        "m_PfEMP1_minor_antibodies_{i}_m_antibody_capacity": [
            f"m_PfEMP1_minor_antibodies_{i}_m_antibody_capacity" for i in range(385)
        ],
        "m_PfEMP1_minor_antibodies_{i}_m_antibody_concentration": [
            f"m_PfEMP1_minor_antibodies_{i}_m_antibody_concentration"
            for i in range(385)
        ],
        # major
        "m_PfEMP1_major_antibodies_{i}_present": [
            f"m_PfEMP1_major_antibodies_{i}_present" for i in range(1070)
        ],
        "m_PfEMP1_major_antibodies_{i}_m_antibody_capacity": [
            f"m_PfEMP1_major_antibodies_{i}_m_antibody_capacity" for i in range(1070)
        ],
        "m_PfEMP1_major_antibodies_{i}_m_antibody_concentration": [
            f"m_PfEMP1_major_antibodies_{i}_m_antibody_concentration"
            for i in range(1070)
        ],
    }

    infection_fill_names = [
        "m_start_measuring",
        "m_is_symptomatic",
        "m_is_newly_symptomatic",
        "total_duration",
        "m_temp_duration",
        "m_measured_duration",
        "m_IRBCtimer",
        "m_gametosexratio",
        "m_gametorate",
        "infectiousness",
        "infectious_timer",
        "incubation_timer",
        "duration",
        "sim_time_created",
        "m_nonspectype",
        "m_MSPtype",
        "m_max_parasites",
        "m_inv_microliters_blood",
        "m_hepatocytes",
        "m_asexual_phase",
        "m_asexual_cycle_count",
    ]

    double_names = {
        "m_minor_epitope_type_{i}": [f"m_minor_epitope_type_{i}" for i in range(50)],
        "m_malegametocytes_{i}": [f"m_malegametocytes_{i}" for i in range(6)],
        "m_femalegametocytes_{i}": [f"m_femalegametocytes_{i}" for i in range(6)],
        "m_IRBCtype_{i}": [f"m_IRBCtype_{i}" for i in range(50)],
        "m_IRBC_count_{i}": [f"m_IRBC_count_{i}" for i in range(50)],
    }

    infection_list = [1, 2, 3]
    for nm in infection_fill_names:
        grouped_cols[nm] = [f"infection_{i}_{nm}" for i in infection_list]

    for nm in double_names:
        grouped_cols[nm] = [
            f"infection_{i}_{sub_nm}"
            for i in infection_list
            for sub_nm in double_names[nm]
        ]

    # Now, check whether any of the grouped columns are actually in single_value_cols
    # That is, the grouped columns is a hand-crafted list of columns.
    # The dataset may dictate that we throw out some of these columns.
    for group_name, group_cols in grouped_cols.items():
        if any(col in single_value_cols for col in group_cols):
            print(f"Group {group_name} contains single-value columns:")
            print([col for col in group_cols if col in single_value_cols])
            print("Removing them...")
            grouped_cols[group_name] = [
                col for col in group_cols if col not in single_value_cols
            ]

    # Check if any values in grouped_cols are empty lists. If so, remove them.
    empty_groups = [
        group_name for group_name, group_cols in grouped_cols.items() if not group_cols
    ]
    for group_name in empty_groups:
        print(f"Removing empty group: {group_name}")
        del grouped_cols[group_name]

    # build a single list of all columns that are grouped (i.e. unroll the dictionary)
    grouped_cols_list = [col for cols in grouped_cols.values() for col in cols]

    # now loop over all columns in the actual dataset.
    # if they are not in the grouped columns, then add them as their own group
    for col in data.columns:
        if col not in grouped_cols_list:
            grouped_cols[col] = [col]

    print(f"We have distilled the dataset into {len(grouped_cols)} groups of columns.")
    print(
        "We will apply the same transformation to all columns in each group. This will help get robust statistics, and simplify our interpretations."
    )

    # First, we will apply a log10(1+x) transformation every element of the dataset.
    train_data = np.log10(1 + train_data.astype(float))
    test_data = np.log10(1 + test_data.astype(float))

    # Then we will apply a MinMaxScaling that computes min/max values for each column group.
    # This will ensure that the same scaling is applied to all columns in a group.
    scaler = CustomMinMaxScalerWithGroups(
        feature_groups=list(grouped_cols.values()), feature_range=(0, 1)
    )
    train_data_scaled = scaler.fit_transform(train_data, return_df=True)
    test_data_scaled = scaler.transform(test_data, return_df=True)

    # find_problematic_columns(train_data_scaled, test_data_scaled)

    # Create Datasets without indices, working with NumPy arrays
    # re-build the grouped_cols_list
    all_cols = [col for cols in grouped_cols.values() for col in cols]
    columns_by_type = {
        "binary": [col for col in all_cols if col in boolean_cols],
        "real": [col for col in all_cols if col not in boolean_cols],
    }

    print("Creating training dataset...")
    train_dataset_pre_split = CustomDataset(
        train_data_scaled, binary_cols=columns_by_type["binary"]
    )
    print("Creating test dataset...")
    test_dataset = CustomDataset(
        test_data_scaled, binary_cols=columns_by_type["binary"]
    )

    # Split train data into train and validation: # 80% train, 20% validation
    train_dataset, val_dataset = random_split(train_dataset_pre_split, [0.8, 0.2])

    if batch_size is None:
        train_batch_size = len(train_dataset)
        val_batch_size = len(val_dataset)
        test_batch_size = len(test_dataset)
    else:
        # take minimum of batch size and data size
        train_batch_size = min(len(train_dataset), batch_size)
        val_batch_size = min(len(val_dataset), batch_size)
        test_batch_size = min(len(test_dataset), batch_size)

    print(f"Using batch size of {train_batch_size} for training, {val_batch_size} for validation, and {test_batch_size} for testing.")

    # no reason to shuffle the data if we are using the full batch size
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def find_problematic_columns(train_data, test_data, relative_threshold=2, absolute_threshold=1000):
    # Initialize empty lists to track columns with significant differences in min/max values
    problematic_columns = []

    # Iterate over each column
    for col in train_data.columns:
        # Get min/max for the training and test data for this column
        train_min, train_max = train_data[col].min(), train_data[col].max()
        test_min, test_max = test_data[col].min(), test_data[col].max()

        # Calculate relative differences
        relative_diff_max = (test_max - train_max) / (abs(train_max) + 1e-9)  # Adding a small number to avoid division by zero
        relative_diff_min = (train_min - test_min) / (abs(train_min) + 1e-9)

        # Check if the relative differences are above the threshold
        if abs(relative_diff_max) > relative_threshold or abs(relative_diff_min) > relative_threshold:
            problematic_columns.append({
                'column': col,
                'train_min': train_min,
                'train_max': train_max,
                'test_min': test_min,
                'test_max': test_max,
                'reason': 'Relative threshold exceeded'
            })

        # Check if the absolute differences exceed the absolute threshold
        elif abs(test_max - train_max) > absolute_threshold or abs(test_min - train_min) > absolute_threshold:
            problematic_columns.append({
                'column': col,
                'train_min': train_min,
                'train_max': train_max,
                'test_min': test_min,
                'test_max': test_max,
                'reason': 'Absolute threshold exceeded'
            })

    # Print out the problematic columns
    if problematic_columns:
        print("Problematic columns (where test values differ significantly from training values):")
        for col_info in problematic_columns:
            print(f"Column: {col_info['column']}")
            print(f"  Reason: {col_info['reason']}")
            print(f"  Train Min: {col_info['train_min']}, Train Max: {col_info['train_max']}")
            print(f"  Test Min: {col_info['test_min']}, Test Max: {col_info['test_max']}")
    else:
        print("No problematic columns found.")
