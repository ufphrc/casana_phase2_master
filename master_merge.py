import streamlit as st
import pandas as pd
from datetime import datetime
import io

# ----------------------------- Helper Functions ----------------------------- #

def combine_checkbox_columns(df, base_column_name, column_labels, new_column_name):
    """
    Combines multiple checkbox columns into a single column with concatenated labels.

    Parameters:
    - df: DataFrame containing the columns to be combined.
    - base_column_name: The base name of the checkbox columns.
    - column_labels: A dictionary mapping the checkbox indices to their respective labels.
    - new_column_name: The name of the new combined column.
    """
    df[new_column_name] = ""
    for idx, label in column_labels.items():
        column_name = f"{base_column_name}___{idx}"
        if column_name in df.columns:
            df[new_column_name] += df[column_name].apply(lambda x: label if x == 1 else "")
            df[new_column_name] += df[column_name].apply(lambda x: ";" if x == 1 else "")
    df[new_column_name] = df[new_column_name].str.rstrip(";")
    df.drop([f"{base_column_name}___{idx}" for idx in column_labels.keys() if f"{base_column_name}___{idx}" in df.columns], axis=1, inplace=True)

def move_column(df, column_to_move, target_column):
    """
    Moves a column to a new position next to the target column.

    Parameters:
    - df: DataFrame containing the columns.
    - column_to_move: The name of the column to move.
    - target_column: The name of the column to place the column_to_move next to.
    """
    if column_to_move in df.columns and target_column in df.columns:
        cols = list(df.columns)
        target_index = cols.index(target_column) + 1
        cols.insert(target_index, cols.pop(cols.index(column_to_move)))
        df = df[cols]
    return df

def filter_by_date(df, start_date, end_date):
    """
    Filters the DataFrame based on a user-provided date range.

    Parameters:
    - df: DataFrame containing the date column 'time_1'.
    - start_date: The start date for filtering.
    - end_date: The end date for filtering.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    mask = (df['time_1'] >= start_date) & (df['time_1'] < end_date)
    return df[mask]

# ----------------------------- Streamlit User Interface ----------------------------- #

st.title("Casana Phase 2: Master Data Set Application")

# Allow the user to select a date range
start_date = st.date_input("Select Start Date", datetime.now())
end_date = st.date_input("Select End Date", datetime.now())

# File upload inputs
uploaded_df1 = st.file_uploader("Upload Visit 2 REDCap Measures", type="csv")
uploaded_df2 = st.file_uploader("Upload Google sheets BP", type="csv")
uploaded_med_df = st.file_uploader("Upload Medical Reference", type="csv")
uploaded_v1_df = st.file_uploader("Upload Visit 1 Master CSV", type="csv")

if uploaded_df1 and uploaded_df2 and uploaded_med_df and uploaded_v1_df:
    # ----------------------------- Data Loading and Initial Processing ----------------------------- #
    
    # Load the CSV files
    df1 = pd.read_csv(uploaded_df1)
    df2 = pd.read_csv(uploaded_df2)
    med_df = pd.read_csv(uploaded_med_df)
    v1_df = pd.read_csv(uploaded_v1_df)
    
    # ----------------------------- Duplicate Check in Google Sheets Data ----------------------------- #
    
    # Check for duplicates in 'rec_id' column of df2
    duplicate_rec_ids = df2[df2['rec_id'].duplicated()]['rec_id'].unique()
    if len(duplicate_rec_ids) > 0:
        st.write("**The 'rec_id' column has duplicates.**")
        st.write(f"Duplicate 'rec_id' values: {duplicate_rec_ids}")
    else:
        st.write("**The 'rec_id' column has no duplicates.**")
        
    #----------------------DATA VALIDATION: BP READINGS------------------------------------------------------#
    # Function to adjust minutes in the time string
    def adjust_minutes(time_str):
        minutes, seconds = time_str.split(':')
        minutes = int(minutes)
        if minutes in [0, 1, 2, 3]:
            minutes += 60
        return f'{minutes:02}:{seconds}'

    # Function to adjust minutes in the link time string
    def adjust_link_minutes(link_time_str):
        link_minutes, link_seconds = link_time_str.split('-')
        link_minutes = int(link_minutes)
        if link_minutes in [0, 1, 2, 3]:
            link_minutes += 60
        return f'{link_minutes:02}-{link_seconds}'

    # Function to check for duplicate links
    def check_duplicate_links(data, link_columns):
        all_links = data[link_columns].values.flatten()
        unique_links = set()
        duplicates = set()
        
        for link in all_links:
            if link in unique_links:
                duplicates.add(link)
            else:
                unique_links.add(link)
        
        return list(duplicates)

    # Function to create a table with time and link consistency details
    def create_time_link_table(data, time_columns, link_columns):
        table_data = []

        for idx, row in data.iterrows():
            for i in range(1, 11):
                time_col = f'time_{i}'
                link_col = f'ths_link_{i}'
                if pd.notnull(row[time_col]) and pd.notnull(row[link_col]):
                    if isinstance(row[time_col], pd.Timestamp):
                        time_str = row[time_col].strftime('%M:%S')  # Convert timestamp to string (minutes:seconds)
                    else:
                        time_str = row[time_col][-5:]  # Assume it's already a string (minutes:seconds)
                    time_str = adjust_minutes(time_str)
                    link_parts = row[link_col].split('-')
                    if len(link_parts) >= 2:
                        link_time_str = link_parts[-2] + '-' + link_parts[-1].split('.')[0]  # Extract the last two parts before .pb.bin
                        link_time_str = adjust_link_minutes(link_time_str)
                        try:
                            time_minutes, time_seconds = map(int, time_str.split(':'))
                            link_minutes, link_seconds = map(int, link_time_str.split('-'))

                            # Convert both times to seconds
                            time_in_seconds = time_minutes * 60 + time_seconds
                            link_in_seconds = link_minutes * 60 + link_seconds

                            # Add row to the table data
                            table_data.append([time_str, link_time_str, time_in_seconds, link_in_seconds])
                        except ValueError:
                            # Skip rows with invalid time formats
                            table_data.append([time_str, link_time_str, 'Invalid', 'Invalid'])

        return pd.DataFrame(table_data, columns=['time_str', 'link_time_str', 'time_in_seconds', 'link_in_seconds'])

    # Function to check time and link consistency
    def check_time_link_consistency(data, time_columns, link_columns):
        inconsistent_links = []

        for idx, row in data.iterrows():
            for i in range(1, 11):
                time_col = f'time_{i}'
                link_col = f'ths_link_{i}'
                if pd.notnull(row[time_col]) and pd.notnull(row[link_col]):
                    if isinstance(row[time_col], pd.Timestamp):
                        time_str = row[time_col].strftime('%M:%S')  # Convert timestamp to string (minutes:seconds)
                    else:
                        time_str = row[time_col][-5:]  # Assume it's already a string (minutes:seconds)
                    time_str = adjust_minutes(time_str)
                    link_parts = row[link_col].split('-')
                    if len(link_parts) >= 2:
                        link_time_str = link_parts[-2] + '-' + link_parts[-1].split('.')[0]  # Extract the last two parts before .pb.bin
                        link_time_str = adjust_link_minutes(link_time_str)
                        try:
                            time_minutes, time_seconds = map(int, time_str.split(':'))
                            link_minutes, link_seconds = map(int, link_time_str.split('-'))

                            # Convert both times to seconds
                            time_in_seconds = time_minutes * 60 + time_seconds
                            link_in_seconds = link_minutes * 60 + link_seconds
                            difference_seconds = time_in_seconds - link_in_seconds

                            # Check if time is 1:10 to 3:20 minutes ahead
                            if not (70 <= difference_seconds <= 200):
                                inconsistent_links.append((row[link_col], idx, i, difference_seconds))
                                
                        except ValueError:
                            # Skip rows with invalid time formats
                            inconsistent_links.append((row[link_col], idx, i, 'Invalid'))

        return inconsistent_links

    # Process the filtered data for duplicate links and time-link consistency
    link_columns = [col for col in df2.columns if col.startswith('ths_link_')]
    time_columns = [f'time_{i}' for i in range(1, 11)]

    # Check for duplicate links
    duplicate_links = check_duplicate_links(df2, link_columns)

    # Check for time and link consistency
    inconsistent_links = check_time_link_consistency(df2, time_columns, link_columns)

    # Extract additional information for inconsistent links
    inconsistent_links_info = []
    for link, idx, i, difference_seconds in inconsistent_links:
        rec_id = df2.at[idx, 'rec_id'] if 'rec_id' in df2.columns else None
        inconsistent_links_info.append({'links': link, 'rec_id': rec_id, 'link_#': i, 'difference_seconds': difference_seconds})

    # Create a DataFrame for inconsistent links
    df_inconsistent = pd.DataFrame(inconsistent_links_info)

    # Display the results    
    st.subheader("Duplicate Links")
    st.write(duplicate_links)

    st.subheader("Inconsistent Links")
    st.write("Below are the links and sit numbers which didn't satisfy the rule of 70 seconds and 320 seconds:")
    st.write(df_inconsistent)
    
    # ----------------------------- Cleaning and Preparing Visit 2 REDCap Data ----------------------------- #

    # List of columns to remove from df1
    columns_to_remove_df1 = [
        'redcap_survey_identifier', 'initial_phone_eligibility_assessment_timestamp', 
        'initial_phone_eligibility_assessment_complete', 'demographics_timestamp', 
        'demographics_complete', 'medical_history_timestamp', 'medical_history_complete', 
        'self_reported_health_social_engagement_timestamp', 
        'montreal_cognitive_assessment_moca_timestamp', 'moca_edu', 'moca_1', 'moca_2', 
        'moca_3', 'visuospatial', 'moca_4', 'moca_5', 'moca_6', 'naming', 'moca_7', 'moca_8', 
        'moca_9', 'moca_10', 'attention', 'moca_11', 'moca_12', 'language', 'moca_13', 
        'abstraction', 'moca_14', 'delayed_recall', 'moca_15', 'orientation', 'total_moca', 
        'montreal_cognitive_assessment_moca_complete', 'physical_health_assessment_timestamp', 
        'physical_health_assessment_complete', 'final_eligibility_assessment_timestamp', 
        'final_eligibility_assessment_complete', 'bathroom_data_collection_timestamp', 
        'data_bathroom_id', 'data_start_time1', 'std_sys_bp1', 'std_dia_bp1', 'data_time_link', 
        'data_signal', 'data_start_time2', 'std_sys_bp2', 'std_dia_bp2', 'data_time_link_2', 
        'data_signal_2', 'data_start_time3', 'std_sys_bp3', 'std_dia_bp3', 'data_time_link_3', 
        'data_signal_3', 'bathroom_data_collection_complete', 
        'participant_experience_timestamp', 'participant_experience_complete', 
        'self_reported_health_social_engagement_complete', 'medical_history_1_item_complete',
        'demo_fam_med___1', 'demo_fam_med___2', 'demo_fam_med___3', 'demo_fam_med___4', 
        'demo_fam_med___5', 'demo_fam_med___6', 'demo_fam_med_other', 'med_his_diabetes', 
        'med_his_dia_medic___1', 'med_his_dia_medic___2', 'med_his_dia_medic___3', 
        'med_his_dia_medic___4', 'med_his_dia_medic___5', 'med_his_dia_other', 'med_his_bp', 
        'med_his_bp_medic___1', 'med_his_bp_medic___2', 'med_his_bp_medic___3', 
        'med_his_bp_medic___4', 'med_his_bp_medic___5', 'med_his_bp_other', 'med_his_cardio', 
        'med_his_cardio_medic___1', 'med_his_cardio_medic___2', 'med_his_cardio_medic___3', 
        'med_his_cardio_medic___4', 'med_his_cardio_medic___5', 'med_his_cardio_other', 
        'med_his_cardio_attack', 'med_his_cardio_aortic', 'med_his_cardio_artery', 
        'med_his_cardio_enlarge', 'med_his_cardio_regur', 'med_his_cardio_fail', 
        'med_his_cardio_breath', 'med_his_cardio_fluid', 'med_his_cardio_diuretics', 
        'med_his_arry', 'med_his_afib', 'med_his_aflutter', 'med_his_svt', 'med_his_vt', 
        'med_his_vfib', 'med_his_pacs', 'med_his_pvcs', 'med_his_arr_other', 'med_his_que_other___1', 
        'med_his_que_other___2', 'med_his_que_other___3', 'med_his_que_other___4', 
        'med_his_que_other___5', 'med_his_que_other___6', 'med_his_que_other___7', 
        'med_his_question_other1', 'med_his_arrhy_device', 'med_his_arrhy_neuro', 
        'med_his_allergy', 'med_his_childbearing', 'med_his_preg', 'medical_history_1_item_timestamp','med_one', 'med_his_arr_other1', 'phy_skin', 'coord_perc'
    ]

    # Remove the specified columns from df1
    df1 = df1.drop(columns=list(set(columns_to_remove_df1)))

    # Combine rows for each record_id
    df1_combined = df1.groupby('record_id').first().reset_index()

    # Drop the redcap_event_name column
    if 'redcap_event_name' in df1_combined.columns:
        df1_combined = df1_combined.drop(columns=['redcap_event_name'])

    # ----------------------------- Processing Google Sheets BP Data ----------------------------- #

    # Create the pass_fail column in df2
    df2['pass_fail'] = df2['good_readings'].apply(lambda x: 'pass' if pd.notnull(x) and x.strip() else 'fail')

    # Move the pass_fail column right next to good_readings
    good_readings_index = df2.columns.get_loc('good_readings')
    df2.insert(good_readings_index + 1, 'pass_fail', df2.pop('pass_fail'))

    # Convert time_1 to datetime
    df2['time_1'] = pd.to_datetime(df2['time_1'])

    # Filter the Google Sheets data based on the selected date range
    filtered_df = filter_by_date(df2, start_date, end_date)

    # Perform the left join using different column names
    merged_df = pd.merge(df1_combined, filtered_df, left_on='record_id', right_on='rec_id', how='right')

    # Further cleaning
    if 'rec_id' in merged_df.columns:
        merged_df = merged_df.drop(columns=['rec_id'])
    merged_df['pe_easy'] = merged_df['pe_easy'].fillna(0).astype(int)
    merged_df['pe_overall_exp'] = merged_df['pe_overall_exp'].fillna(0).astype(int)
    merged_df['pe_easy'] = merged_df['pe_easy'].apply(lambda x: 11 - x)
    merged_df['pe_overall_exp'] = merged_df['pe_overall_exp'].apply(lambda x: 11 - x)
    merged_df['pe_valuable'] = merged_df['pe_valuable'].fillna(0).astype(int)

    # ----------------------------- Eligibility Processing ----------------------------- #

    # Processing 'exc_eligible_2' column based on provided criteria
    exc_eligible_2_index = merged_df.columns.get_loc('exc_eligible_2')
    merged_df = merged_df.drop(columns=['exc_eligible_2'])
    merged_df.insert(exc_eligible_2_index, 'exc_eligible_2', '')
    merged_df['exc_eligible_2'] = merged_df.apply(lambda row: 'Eligible' if (
        row['exc_eligible'] == 'Eligible' and
        40 <= row['phy_hr'] <= 110 and
        row['phy_spo2'] >= 90 and
        90 <= row['phy_weight_lb'] <= 350 and
        row['phy_bmi'] < 45 and
        row['total_moca_2'] >= 18
    ) else 'Ineligible', axis=1)

    # ----------------------------- Processing Medical Reference Data ----------------------------- #

    # Prepare med_df
    med_df['record_id'] = med_df['record_id'].astype(str) + '-B'
    
    # Assume filtered_med_df is just med_df for this example
    filtered_med_df = med_df
    
    # Combine checkbox columns
    family_med_labels = {
        1: "Cancer",
        2: "Cardiovascular Disease (CVD)",
        3: "Neurological Illness",
        4: "Dementia / Alzheimer's",
        5: "Other",
        6: "Not applicable"
    }
    combine_checkbox_columns(filtered_med_df, 'demo_fam_med', family_med_labels, 'demo_fam_med')

    # Reordering columns
    filtered_med_df = move_column(filtered_med_df, 'demo_fam_med', 'med_any_change')
    
    # Define the mapping of column names to their corresponding labels
    column_label_mapping = {
        'med_his_diabetes': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_bp': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_attack': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_aortic': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_artery': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_enlarge': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_regur': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_fail': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_breath': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_fluid': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_cardio_diuretics': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_arry': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_afib': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_aflutter': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_svt': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_vt': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_vfib': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_pacs': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_pvcs': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_arr_other': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_arrhy_device': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_arrhy_neuro': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_allergy': {1: 'Yes', 0: 'No', 2: 'Unsure'},
        'med_his_childbearing': {1: 'Yes', 0: 'No'},
        'med_his_preg': {1: 'Yes', 0: 'No'}
    }
    
    for column, mapping in column_label_mapping.items():
        if column in filtered_med_df.columns:
            filtered_med_df[column] = filtered_med_df[column].map(mapping)

    # Remove unnecessary columns
    columns_to_remove_med_df = [
        'medical_history_complete', 'confirmation_timestamp', 'change_med_his', 
        'confirmation_complete', 'redcap_survey_identifier', 
        'medical_history_timestamp', 'med_any_change'
    ]
    filtered_med_df.drop(columns=columns_to_remove_med_df, inplace=True)
    
    # Merge dataframes
    result_df = pd.merge(merged_df, filtered_med_df, on='record_id', how='left')
    if 'record_id_x' in result_df.columns and 'record_id_y' in result_df.columns:
        result_df.drop(['record_id_y'], axis=1, inplace=True)
        result_df.rename(columns={'record_id_x': 'record_id'}, inplace=True)
    
    # Reorder columns near 'demo_gender'
    columns_to_move_near_demo_gender = [
        'demo_fam_med', 'demo_fam_med_other', 'med_his_diabetes', 'med_his_dia_medic', 'med_his_dia_other',
        'med_his_bp', 'med_his_bp_medic', 'med_his_bp_other', 'med_his_cardio', 'med_his_cardio_medic',
        'med_his_cardio_other', 'med_his_cardio_attack', 'med_his_cardio_aortic', 'med_his_cardio_artery',
        'med_his_cardio_enlarge', 'med_his_cardio_regur', 'med_his_cardio_fail', 'med_his_cardio_breath',
        'med_his_cardio_fluid', 'med_his_cardio_diuretics', 'med_his_arry', 'med_his_afib', 'med_his_aflutter',
        'med_his_svt', 'med_his_vt', 'med_his_vfib', 'med_his_pacs', 'med_his_pvcs', 'med_his_arr_other',
        'med_his_arr_other1', 'med_his_que_other___1', 'med_his_que_other___2', 'med_his_que_other___3',
        'med_his_que_other___4', 'med_his_que_other___5', 'med_his_que_other___6', 'med_his_que_other___7',
        'med_his_question_other1', 'med_his_arrhy_device', 'med_his_arrhy_neuro', 'med_his_allergy',
        'med_his_childbearing', 'med_his_preg'
    ]
    cols = list(result_df.columns)
    demo_gender_index = cols.index('demo_gender') + 1
    for col in columns_to_move_near_demo_gender:
        if col in cols:
            cols.insert(demo_gender_index, cols.pop(cols.index(col)))
            demo_gender_index += 1
    result_df = result_df[cols]

    # ----------------------------- Processing Visit 1 Master Data ----------------------------- #

    # Prepare v1_df
    v1_df['record_id'] = v1_df['record_id'].astype(str) + '-B'
    unique_record_ids = result_df['record_id'].unique()
    filtered_v1_df = v1_df[v1_df['record_id'].isin(unique_record_ids)]
    columns_to_pull_v1 = [
        'record_id', 'demo_race', 'demo_highest_edu', 'demo_current_employ', 'demo_military',
        'demo_military_active', 'demo_years_served', 'demo_current_marital', 'demo_marital_other',
        'demo_living_arrange', 'demo_living_other', 'demo_children', 'demo_disability',
        'demo_disability_spec', 'demo_lived_in_villages', 'demo_residency',
        'demo_years_lived_villages', 'demo_surrounding', 'phy_skin'
    ]
    v1_df_filtered = filtered_v1_df[columns_to_pull_v1]
    
    # Merge result_df with v1_df_filtered
    final_df = pd.merge(result_df, v1_df_filtered, on='record_id', how='left')
    if 'record_id_x' in final_df.columns and 'record_id_y' in final_df.columns:
        final_df.drop(['record_id_y'], axis=1, inplace=True)
        final_df.rename(columns={'record_id_x': 'record_id'}, inplace=True)

    # Reorder columns near 'demo_gender'
    columns_to_move_near_demo_gender = [
        'demo_race', 'demo_highest_edu', 'demo_current_employ', 'demo_military',
        'demo_military_active', 'demo_years_served', 'demo_current_marital',
        'demo_marital_other', 'demo_living_arrange', 'demo_living_other',
        'demo_children', 'demo_disability', 'demo_disability_spec',
        'demo_lived_in_villages', 'demo_residency', 'demo_years_lived_villages',
        'demo_surrounding'
    ]
    cols = list(final_df.columns)
    demo_gender_index = cols.index('demo_gender') + 1
    for col in columns_to_move_near_demo_gender:
        if col in cols:
            cols.insert(demo_gender_index, cols.pop(cols.index(col)))
            demo_gender_index += 1

    # Move 'phy_skin' just beside 'phy_arm'
    if 'phy_skin' in cols and 'phy_arm' in cols:
        phy_arm_index = cols.index('phy_arm') + 1
        cols.insert(phy_arm_index, cols.pop(cols.index('phy_skin')))
    final_df = final_df[cols]

    # ----------------------------- Column Mappings ----------------------------- #

    # Define column mappings
    column_label_mapping = {
        'exc_age': {1: 'Yes', 0: 'No'},
        'exc_eng': {1: 'Yes', 0: 'No'},
        'exc_loc': {1: 'Yes', 0: 'No'},
        'exc_sit': {1: 'Yes', 0: 'No'},
        'exc_stand': {1: 'Yes', 0: 'No'},
        'exc_mob': {1: 'Yes', 0: 'No'},
        'exc_res': {1: 'Yes', 0: 'No'},
        'exc_consent': {1: 'Yes', 0: 'No'},
        'exc_mcs': {1: 'Yes', 0: 'No'},
        'exc_valve': {1: 'Yes', 0: 'No'},
        'exc_afib': {1: 'Yes', 0: 'No'},
        'exc_dialysis': {1: 'Yes', 0: 'No'},
        'exc_allergies': {1: 'Yes', 0: 'No'},
        'exc_skin': {1: 'Yes', 0: 'No'},
        'exc_hr': {1: 'Yes', 0: 'No'},
        'exc_spo2': {1: 'Yes', 0: 'No'},
        'exc_weight': {1: 'Yes', 0: 'No'},
        'exc_bmi': {1: 'Yes', 0: 'No'},
        'exc_eligible': {1: 'Eligible', 0: 'Ineligible'},
        'demo_gender': {0: 'Male', 1: 'Female'}
    }

    # Apply column mappings
    for column, mapping in column_label_mapping.items():
        if column in final_df.columns:
            final_df[column] = final_df[column].map(mapping)

    # ----------------------------- Combining Multi-Checkbox Columns ----------------------------- #

    # Function to combine multi-checkbox columns into a single column
    def combine_checkbox_columns(df, base_column_name, column_labels, new_column_name):
        """
        Combines multiple checkbox columns into a single column with concatenated labels.

        Parameters:
        - df: DataFrame containing the columns to be combined.
        - base_column_name: The base name of the checkbox columns.
        - column_labels: A dictionary mapping the checkbox indices to their respective labels.
        - new_column_name: The name of the new combined column.
        """
        df[new_column_name] = ""
        for idx, label in column_labels.items():
            column_name = f"{base_column_name}___{idx}"
            if column_name in df.columns:
                df[new_column_name] += df[column_name].apply(lambda x: label if x == 1 else "")
                df[new_column_name] += df[column_name].apply(lambda x: ";" if x == 1 else "")
        df[new_column_name] = df[new_column_name].str.rstrip(";")
        df.drop([f"{base_column_name}___{idx}" for idx in column_labels.keys() if f"{base_column_name}___{idx}" in df.columns], axis=1, inplace=True)

    # Combine med_his_que_other columns into a single column
    med_his_que_other_labels = {
        1: "High Cholesterol",
        2: "Sleep Apnea",
        3: "Kidney disease",
        4: "Neurological illness",
        5: "Mental illness",
        6: "Other",
        7: "Not applicable"
    }
    combine_checkbox_columns(final_df, 'med_his_que_other', med_his_que_other_labels, 'med_his_que_other')

    # Move the combined column and med_his_question_other1 to the correct position
    final_df = move_column(final_df, 'med_his_que_other', 'med_his_arr_other1')
    final_df = move_column(final_df, 'med_his_question_other1', 'med_his_que_other')

    # Combine med_his_dia_medic columns into a single column
    med_his_dia_medic_labels = {
        1: "I do not take any prescription medications for my Diabetes",
        2: "Metformin",
        3: "Insulin (various types, e.g. rapid-acting, long-acting, etc.)",
        4: "Sulfonylureas (e.g. glipizide, glyburide)",
        5: "Are you taking any other medications for your Diabetes not listed above?"
    }
    combine_checkbox_columns(final_df, 'med_his_dia_medic', med_his_dia_medic_labels, 'med_his_dia_medic')

    # Move the combined column after med_his_diabetes
    final_df = move_column(final_df, 'med_his_dia_medic', 'med_his_diabetes')

    # Combine med_his_bp_medic columns into a single column
    med_his_bp_medic_labels = {
        1: "I do not take any prescription medications for my high blood pressure",
        2: "Angiotensin-converting enzyme (ACE) inhibitors (e.g.: lisinopril, enalapril)",
        3: "Angiotensin II receptor blockers (ARBs) (e.g.: losartan, valsartan)",
        4: "Calcium channel blockers (e.g.: amlodipine, nifedipine)",
        5: "Are you taking any other medications for your high blood pressure not listed above?"
    }
    combine_checkbox_columns(final_df, 'med_his_bp_medic', med_his_bp_medic_labels, 'med_his_bp_medic')

    # Move the combined column after med_his_bp
    final_df = move_column(final_df, 'med_his_bp_medic', 'med_his_bp')

    # Combine med_his_cardio_medic columns into a single column
    med_his_cardio_medic_labels = {
        1: "I do not take any prescription medications for my cardiovascular disease",
        2: "Beta-blockers (e.g.: Metoprolol, Atenolol)",
        3: "Statins (e.g.: Atorvastatin, Simvastatin)",
        4: "Aspirin (for Anti-platelet therapy - as a blood thinner/an anticoagulant, not for pain relief)",
        5: "Are you taking any other medications for your cardiovascular disease not listed above?"
    }
    combine_checkbox_columns(final_df, 'med_his_cardio_medic', med_his_cardio_medic_labels, 'med_his_cardio_medic')

    # Move the combined column after med_his_cardio
    final_df = move_column(final_df, 'med_his_cardio_medic', 'med_his_cardio')
    
    # ----------------------------- Export to Excel ----------------------------- #

    # Format the filename using the selected start and end dates
    start_date_str = start_date.strftime('%m%d')
    end_date_str = end_date.strftime('%m%d')
    file_name_xlsx = f"master-{start_date_str}-{end_date_str}.xlsx"

    # Convert the final DataFrame to an Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    
    # Allow the user to download the processed Excel file
    st.download_button(label="Download Excel File", data=output, file_name=file_name_xlsx, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
