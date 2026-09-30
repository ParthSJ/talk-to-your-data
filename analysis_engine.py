"""
analysis_engine.py — Person B's model functions, copied verbatim from
Hackathon_Work_Updated.ipynb (cells 21-23), unmodified except for the data
path. The notebook reads from a Google-Drive mount; this reads from a
local CSV next to this file, since a deployed server has no Drive to
mount.

This is the second pass: top_drivers is now computed (coefficients pulled
off the fitted pipeline, numeric ones un-scaled back to real dollar-per-
unit terms), and predict now only requires age/bmi/smoker, defaulting the
other three and reporting which were defaulted. Both match the locked
contract. Verified against this file's own test cells before it was even
wired in here (see the notebook's cells 26-28).
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

analysis_results = {}


# --- verbatim from cell-21, only project_dir -> DATA_DIR ---------------

def get_dataset_context(dataset_id='med_insurance'):
    if dataset_id != 'med_insurance':
        return {
            "ok": False,
            "message": "Ask the user to choose an available dataset.",
            "reason": "unknown_dataset",
            "available_datasets": ['med_insurance']
        }

    data = pd.read_csv(f"{DATA_DIR}/insurance.csv").drop_duplicates().reset_index(drop=True)
    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    categorical_columns = ["sex", "smoker", "region"]
    numerical_columns = ["age", "bmi", "children"]

    dataset_context = {
        "dataset_id": "med_insurance",
        "name": "Medical Insurance charges - USA",
        "description": "Dataset containing personal characteristics and medical insurance charges for records from United States of America",
        "row_count": len(data),
        "target": "charges",
        "currency": "USD",
        "task_type": "regression",
        "model_type": "linear_regression",
        "input_descriptions": {
            "age": "Age in years",
            "sex": "Sex recorded in the dataset",
            "bmi": "Body Mass Index",
            "children": "Number of children or dependants covered",
            "smoker": "Smoking status",
            "region": "US residential region"
        },
        "feature_columns": features,
        "required_inputs": ["age", "bmi", "smoker"],
        "optional_inputs": ["sex", "children", "region"],
        "column_types": {"categorical": categorical_columns, "numerical": numerical_columns},
        "allowed_categories": {
            "sex": ["female", "male"],
            "smoker": ["no", "yes"],
            "region": ["northeast", "northwest", "southeast", "southwest"]
        },
        "missing_input_policy": {
            "required": "Ask the user for any missing required input.",
            "optional": "Use training-data defaults when omitted, and tell the user which defaults were used."
        }
    }
    return {
        "ok": True,
        "message": "Dataset context retrieved successfully.",
        "reason": None,
        "context": dataset_context
    }


# --- verbatim from cell-22, only project_dir -> DATA_DIR ---------------

def run_analysis(dataset_id='med_insurance', task_type='regression', model_type='linear_regression'):
    context_result = get_dataset_context(dataset_id)

    if not context_result['ok']:
        return context_result

    dataset_context = context_result['context']

    if dataset_context["task_type"] != task_type:
        return {
            "ok": False,
            "message": "Requested task is not supported for this dataset.",
            "reason": "unsupported_task",
            "supported_task": dataset_context["task_type"]
        }

    if dataset_context["model_type"] != model_type:
        return {
            "ok": False,
            "message": "Requested model is not supported for this dataset.",
            "reason": "unsupported_model",
            "supported_model": dataset_context["model_type"]
        }

    med_file = pd.read_csv(f"{DATA_DIR}/insurance.csv").drop_duplicates().reset_index(drop=True)

    X = med_file[dataset_context['feature_columns']]
    y = med_file['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    categorical_columns = ["sex", "smoker", "region"]
    numerical_columns = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(drop="first"), categorical_columns),
        ("numerical", StandardScaler(), numerical_columns)
    ])

    model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    fitted_preprocessor = model_pipeline.named_steps['preprocessor']
    fitted_model = model_pipeline.named_steps['model']
    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = fitted_model.coef_.copy()

    fitted_scaler = fitted_preprocessor.named_transformers_['numerical']

    for column, scale_value in zip(numerical_columns, fitted_scaler.scale_):
        feature_name = f"numerical__{column}"
        feature_index = list(feature_names).index(feature_name)
        coefficients[feature_index] = coefficients[feature_index] / scale_value

    driver_descriptions = {
        "categorical__sex_male": "Male compared with female",
        "categorical__smoker_yes": "Smoker compared with non-smoker",
        "categorical__region_northwest": "Northwest compared with northeast",
        "categorical__region_southeast": "Southeast compared with northeast",
        "categorical__region_southwest": "Southwest compared with northeast",
        "numerical__age": "One additional year of age",
        "numerical__bmi": "One additional BMI point",
        "numerical__children": "One additional child"
    }

    driver_results = []

    for feature_name, coefficient in zip(feature_names, coefficients):
        driver_results.append({
            "feature": feature_name,
            "description": driver_descriptions.get(feature_name, feature_name),
            "estimated_change_usd": round(float(coefficient), 2),
            "direction": "higher charges" if coefficient > 0 else "lower charges"
        })

    positive_drivers = [driver for driver in driver_results if driver["estimated_change_usd"] > 0]

    top_drivers = sorted(positive_drivers, key=lambda driver: driver["estimated_change_usd"], reverse=True)[:4]

    reliability_statement = f"On held-out test data, predictions differed from actual charges by ${mae:,.2f} on average. Individual errors can be larger."

    optional_defaults = {
        "sex": str(X_train["sex"].mode().iloc[0]),
        "children": int(X_train["children"].mode().iloc[0]),
        "region": str(X_train["region"].mode().iloc[0])
    }
    analysis_results[dataset_id] = {
        "pipeline": model_pipeline,
        "task_type": task_type,
        "model_type": model_type,
        "feature_columns": dataset_context["feature_columns"],
        "required_inputs": dataset_context["required_inputs"],
        "optional_inputs": dataset_context["optional_inputs"],
        "target": dataset_context["target"],
        "currency": dataset_context["currency"],
        "test_mae": float(mae),
        "defaults": optional_defaults,
        "top_drivers": top_drivers,
        "reliability_statement": reliability_statement
    }

    return {
        "ok": True,
        "message": "Regression analysis completed successfully.",
        "reason": None,
        "dataset_id": dataset_id,
        "task_type": task_type,
        "model_type": model_type,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_mae": round(float(mae), 2),
        "currency": dataset_context["currency"],
        "defaults": optional_defaults,
        "top_drivers": top_drivers,
        "reliability_statement": reliability_statement
    }


# --- verbatim from cell-23 ----------------------------------------------

def predict_targetval(dataset_id, supplied_inputs):
    if dataset_id not in analysis_results:
        return {
            "ok": False,
            "message": "Dataset ID is not available",
            "reason": "datasetid_not_available"
        }
    data_analysis = analysis_results[dataset_id]

    if "pipeline" not in data_analysis:
        return {
            "ok": False,
            "message": "Fitted pipeline is not available for this dataset",
            "reason": "pipeline_not_available"
        }
    model_pipeline = data_analysis['pipeline']

    if not isinstance(supplied_inputs, dict):
        return {
            "ok": False,
            "message": "Provided inputs are not in the correct format.",
            "reason": "invalid_input_format"
        }
    required_inputs = data_analysis["required_inputs"]
    optional_inputs = data_analysis["optional_inputs"]
    feature_columns = data_analysis["feature_columns"]

    missing_required_inputs = [feature for feature in required_inputs if feature not in supplied_inputs or supplied_inputs[feature] is None]

    if missing_required_inputs:
        missing_field = missing_required_inputs[0]
        return {
            "ok": False,
            "message": f"Ask the user for the missing {missing_field}.",
            "reason": "missing_required_field",
            "field": missing_field
        }

    prediction_inputs = supplied_inputs.copy()
    defaults_used = {}

    for field in optional_inputs:
        if field not in prediction_inputs or prediction_inputs[field] is None:
            default_value = data_analysis['defaults'][field]

            prediction_inputs[field] = default_value
            defaults_used[field] = default_value

    input_data = pd.DataFrame([prediction_inputs], columns=feature_columns)

    predicted_charge = model_pipeline.predict(input_data)[0]

    return {
        "ok": True,
        "message": "Prediction completed successfully.",
        "reason": None,
        "dataset_id": dataset_id,
        "predicted_charge": round(float(predicted_charge), 2),
        "currency": data_analysis["currency"],
        "inputs_used": prediction_inputs,
        "defaults_used": defaults_used,
        "reliability_statement": data_analysis["reliability_statement"]
    }
