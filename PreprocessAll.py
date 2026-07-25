import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.signal import filtfilt

# =====================================================
# Load preprocessing objects once
# =====================================================

LABEL_ENCODER_PATH = "preprocessing_objects/label_encoder.joblib"
SEX_ENCODER_PATH = "preprocessing_objects/sex_encoder.joblib"
LEAD_ENCODER_PATH = "preprocessing_objects/lead_encoder.joblib"
FILTER_PATH = "preprocessing_objects/butterworth_filter.joblib"


label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

sex_encoder = joblib.load(
    SEX_ENCODER_PATH
)

lead_encoder = joblib.load(
    LEAD_ENCODER_PATH
)

filter_params = joblib.load(
    FILTER_PATH
)


b = filter_params["b"]
a = filter_params["a"]


# =====================================================
# Constants
# =====================================================

VALID_SYMBOLS = [
    "N",
    "A",
    "V",
    "F",
    "L",
    "R"
]


INPUT_DIR = "./DataSet/MIT-BIH"

OUTPUT_DIR = "MIT_Preprocessed_Datasets"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================================================
# Preprocessing Function
# =====================================================

def preprocess_record(record_number):

    print(f"\nProcessing record {record_number}")


    # -------------------------------
    # Load ECG CSV
    # -------------------------------

    ekg_file = os.path.join(
        INPUT_DIR,
        f"{record_number}_ekg.csv"
    )

    json_file = os.path.join(
        INPUT_DIR,
        f"{record_number}_ekg.json"
    )


    ekg = pd.read_csv(
        ekg_file
    )


# -------------------------------
# Select ECG lead
# -------------------------------

    preferred_leads = [
    "MLII",
    "V5",
    "V2",
    "V1",
    "V4"
    ]

    signal = None
    selected_lead = None

    for lead in preferred_leads:
        if lead in ekg.columns:
            signal = ekg[lead]
            selected_lead = lead
            break

    if signal is None:
        raise ValueError(
            f"No suitable ECG lead found in record {record_number}")

    print(f"Using {selected_lead} for record {record_number}")

    lead_encoded = lead_encoder.transform([selected_lead])[0]

    # -------------------------------
    # Apply Butterworth filter
    # -------------------------------

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )



    # -------------------------------
    # Extract annotated beats
    # -------------------------------

    features = []
    labels = []


    annotations = ekg[
        ekg["symbol"].isin(VALID_SYMBOLS)
    ]


    for _, row in annotations.iterrows():

        r_peak = int(
            row["Unnamed: 0"]
        )


        # Avoid incomplete windows
        if (
            r_peak < 100
            or r_peak + 150 >= len(filtered_signal)
        ):
            continue


        beat = filtered_signal[r_peak-100:r_peak+150]


        features.append(beat)

        labels.append(row["symbol"])

    if len(features) == 0:
        print(f"No valid beats found in record {record_number}")
        return
    
    x = np.array(features)

    y = np.array(labels)



    # -------------------------------
    # Load patient metadata
    # -------------------------------

    with open(json_file, "r") as f:

        metadata = json.load(f)


    patient_info = metadata["comments"][0]

    parts = patient_info.split()

    comments = metadata.get("comments", [])

    if len(comments) > 0:
        patient_info = comments[0]
        parts = patient_info.split()

        age = int(parts[0])
        sex = parts[1]

    else:
        print(f"No metadata found for record {record_number}")

        age = -1
        sex = "M"   # default fallback


    if sex in sex_encoder.classes_:
        sex_encoded = sex_encoder.transform([sex])[0]
    else:
        sex_encoded = -1



    # -------------------------------
    # Add metadata
    # -------------------------------

    age_column = np.full((len(x), 1),age)


    sex_column = np.full((len(x), 1),sex_encoded)


    lead_column = np.full((len(x), 1),lead_encoded)


    X_final = np.hstack(
        (
            x,
            age_column,
            sex_column,
            lead_column
        )
    )

    # -------------------------------
    # Encode labels
    # -------------------------------

    y_encoded = label_encoder.transform(y)

    # -------------------------------
    # Create dataframe
    # -------------------------------

    columns = [
        f"ecg_{i}"
        for i in range(x.shape[1])
    ]


    columns += ["age","sex","lead"]


    df_processed = pd.DataFrame(
        X_final,
        columns=columns
    )


    df_processed["label"] = y_encoded
    df_processed["lead_name"] = selected_lead
    df_processed["record"] = int(record_number)

    # Convert types
    df_processed["age"] = (df_processed["age"].astype(int))
    df_processed["lead"] = (df_processed["lead"].astype(int))
    df_processed["sex"] = (df_processed["sex"].astype(int))
    df_processed["label"] = (df_processed["label"].astype(int))
    # -------------------------------
    # Save
    # -------------------------------
    output_file = os.path.join(OUTPUT_DIR,f"MITBIH_preprocessed_{record_number}.csv")


    df_processed.to_csv(output_file,index=False)
    print(f"Saved: {output_file}")

    print("Shape:",df_processed.shape)
    return df_processed

# =====================================================
# Command line execution
# =====================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:

        print("Usage: python preprocess_mitbih.py <record_number>")
        sys.exit(1)

    record_number = str(sys.argv[1])
    df=preprocess_record(record_number)
    print(df.head())