import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import os
from currency_converter import CurrencyConverter


# Initialize currency converter
c = CurrencyConverter()

# Set page title and layout
st.set_page_config(
    page_title="Data Science Salary Predictor",
    layout="wide",
    page_icon="💵"
)


# Initialize session state to store prediction results and prevent UI flashing
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.median_pred = None
    st.session_state.lower_bound = None
    st.session_state.upper_bound = None
    # Add default values for selectboxes to maintain consistency
    st.session_state.default_job_title_index = 6  # You can adjust these default indices
    st.session_state.default_experience_index = 0
    st.session_state.default_employment_index = 0
    st.session_state.default_company_size_index = 1
    st.session_state.default_remote_ratio = 0  # Default to no remote work (0%)
    st.session_state.currency = 'USD'  # Default currency
    st.session_state.n_bootstrap = 10  # Default number of bootstraps

# --- SIDEBAR ---
st.sidebar.markdown('<h2 style="color: #00A651;">⚙️ Settings</h2>', unsafe_allow_html=True)

# Currency conversion section
st.sidebar.markdown('<h3 style="color: #00A651; font-size: 1.2rem;">💵 Currency</h3>', unsafe_allow_html=True)
available_currencies = sorted(['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'SGD'])
selected_currency = st.sidebar.selectbox("Display Salary In:", available_currencies, index=available_currencies.index(st.session_state.currency))

# Update currency in session state if changed
if selected_currency != st.session_state.currency:
    st.session_state.currency = selected_currency

# Bootstrap sample control - commented out as per request and fixed at 10
#st.sidebar.markdown('<h3 style="color: #00A651; font-size: 1.2rem;">📊 Model Settings</h3>', unsafe_allow_html=True)
# n_bootstrap = st.sidebar.slider(
#     "Number of Bootstrap Samples for CI:", 
#     min_value=5, 
#     max_value=100, 
#     value=st.session_state.n_bootstrap,
#     help="Higher values give more accurate confidence intervals but take longer to compute."
# )

#st.sidebar.markdown("""
#<div style="background-color: #F0FAF5; padding: 1rem; border-radius: 5px; border-left: 3px solid #00A651; margin-top: 1rem;">
#    <p style="margin: 0; color: #007840;">This app predicts data science salaries based on various job factors using XGBoost regression with bootstrap confidence intervals.</p>
#</div>
#""", unsafe_allow_html=True)

# Update bootstrap samples in session state if changed - commented out and fixed at 10
# if n_bootstrap != st.session_state.n_bootstrap:
#     st.session_state.n_bootstrap = n_bootstrap

# Function to convert USD to selected currency
def convert_currency(amount_usd, target_currency='USD'):
    if target_currency == 'USD':
        return amount_usd, '$'
    
    try:
        converted_amount = c.convert(amount_usd, 'USD', target_currency)
        
        # Currency symbols for display
        currency_symbols = {
            'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 
            'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF', 
            'CNY': '¥', 'INR': '₹', 'SGD': 'S$'
        }
        
        symbol = currency_symbols.get(target_currency, target_currency)
        return converted_amount, symbol
    except:
        st.sidebar.warning(f"Could not convert to {target_currency}. Showing USD.")
        return amount_usd, '$'

# --- Updated Employee Residence Mapping (numeric index to country code) ---
employee_residence_map = {
    0: 'AE', 1: 'AM', 2: 'AR', 3: 'AS', 4: 'AT', 5: 'AU', 6: 'BA', 7: 'BE', 8: 'BG', 9: 'BO',
    10: 'BR', 11: 'CA', 12: 'CF', 13: 'CH', 14: 'CL', 15: 'CN', 16: 'CO', 17: 'CR', 18: 'CY', 
    19: 'CZ', 20: 'DE', 21: 'DK', 22: 'DO', 23: 'DZ', 24: 'EE', 25: 'EG', 26: 'ES', 27: 'FI', 
    28: 'FR', 29: 'GB', 30: 'GH', 31: 'GR', 32: 'HK', 33: 'HN', 34: 'HR', 35: 'HU', 36: 'ID', 
    37: 'IE', 38: 'IL', 39: 'IN', 40: 'IQ', 41: 'IR', 42: 'IT', 43: 'JE', 44: 'JP', 45: 'KE', 
    46: 'KW', 47: 'LT', 48: 'LU', 49: 'LV', 50: 'MA', 51: 'MD', 52: 'MK', 53: 'MT', 54: 'MX', 
    55: 'MY', 56: 'NG', 57: 'NL', 58: 'NZ', 59: 'PH', 60: 'PK', 61: 'PL', 62: 'PR', 63: 'PT', 
    64: 'RO', 65: 'RS', 66: 'RU', 67: 'SE', 68: 'SG', 69: 'SI', 70: 'SK', 71: 'TH', 72: 'TN', 
    73: 'TR', 74: 'UA', 75: 'US', 76: 'UZ', 77: 'VN'
}

# --- Updated Company Location Mapping (numeric index to country code) ---
company_location_map = {
    0: 'AE', 1: 'AL', 2: 'AM', 3: 'AR', 4: 'AS', 5: 'AT', 6: 'AU', 7: 'BA', 8: 'BE', 9: 'BO',
    10: 'BR', 11: 'BS', 12: 'CA', 13: 'CF', 14: 'CH', 15: 'CL', 16: 'CN', 17: 'CO', 18: 'CR', 
    19: 'CZ', 20: 'DE', 21: 'DK', 22: 'DZ', 23: 'EE', 24: 'EG', 25: 'ES', 26: 'FI', 27: 'FR', 
    28: 'GB', 29: 'GH', 30: 'GR', 31: 'HK', 32: 'HN', 33: 'HR', 34: 'HU', 35: 'ID', 36: 'IE', 
    37: 'IL', 38: 'IN', 39: 'IQ', 40: 'IR', 41: 'IT', 42: 'JP', 43: 'KE', 44: 'LT', 45: 'LU', 
    46: 'LV', 47: 'MA', 48: 'MD', 49: 'MK', 50: 'MT', 51: 'MX', 52: 'MY', 53: 'NG', 54: 'NL', 
    55: 'NZ', 56: 'PH', 57: 'PK', 58: 'PL', 59: 'PR', 60: 'PT', 61: 'RO', 62: 'RU', 63: 'SE', 
    64: 'SG', 65: 'SI', 66: 'SK', 67: 'TH', 68: 'TR', 69: 'UA', 70: 'US', 71: 'VN'
}

# Create reverse mappings (country code to numeric index)
employee_residence_code_to_idx = {v: k for k, v in employee_residence_map.items()}
company_location_code_to_idx = {v: k for k, v in company_location_map.items()}

# Country code to full name mapping
country_code_to_name = {
    'AE': 'United Arab Emirates', 'AL': 'Albania', 'AM': 'Armenia', 'AR': 'Argentina', 'AS': 'American Samoa', 
    'AT': 'Austria', 'AU': 'Australia', 'BA': 'Bosnia and Herzegovina', 'BE': 'Belgium', 
    'BG': 'Bulgaria', 'BO': 'Bolivia', 'BR': 'Brazil', 'BS': 'Bahamas', 'CA': 'Canada', 
    'CF': 'Central African Republic', 'CH': 'Switzerland', 'CL': 'Chile', 'CN': 'China', 
    'CO': 'Colombia', 'CR': 'Costa Rica', 'CY': 'Cyprus', 
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark', 'DO': 'Dominican Republic', 
    'DZ': 'Algeria', 'EE': 'Estonia', 'EG': 'Egypt', 'ES': 'Spain', 'FI': 'Finland', 
    'FR': 'France', 'GB': 'United Kingdom', 'GH': 'Ghana', 
    'GR': 'Greece', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croatia', 'HU': 'Hungary', 
    'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel', 'IN': 'India', 'IQ': 'Iraq', 
    'IR': 'Iran', 'IT': 'Italy', 'JE': 'Jersey', 'JP': 'Japan', 'KE': 'Kenya', 
    'KW': 'Kuwait', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MA': 'Morocco', 
    'MD': 'Moldova', 'MK': 'North Macedonia', 'MT': 'Malta', 'MX': 'Mexico', 'MY': 'Malaysia', 
    'NG': 'Nigeria', 'NL': 'Netherlands', 'NZ': 'New Zealand', 'PH': 'Philippines', 
    'PK': 'Pakistan', 'PL': 'Poland', 'PR': 'Puerto Rico', 'PT': 'Portugal', 'RO': 'Romania', 
    'RS': 'Serbia', 'RU': 'Russia', 'SE': 'Sweden', 'SG': 'Singapore', 'SI': 'Slovenia', 
    'SK': 'Slovakia', 'TH': 'Thailand', 'TN': 'Tunisia', 'TR': 'Turkey', 'UA': 'Ukraine', 
    'US': 'United States', 'UZ': 'Uzbekistan', 'VN': 'Vietnam'
}

# Create mapping from country name to country code (for UI selection)
country_name_to_code = {v: k for k, v in country_code_to_name.items()}

# --- MAIN PAGE ---
st.markdown('<h1 class="money-text">Data Science Salary Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="card" style="background-color: #F0FAF5; border-left: 4px solid #00A651; padding: 1rem;">
    <p>Enter job details below to estimate your data science salary. The prediction includes a confidence interval showing the possible salary range.</p>
</div>
""", unsafe_allow_html=True)

# Function to load or train the model
@st.cache_resource
def load_or_train_model():
    model_path = "xgboost_model.pkl"
    df_encoded = pd.read_csv('data/encoded_data.csv')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        train_df, _ = train_test_split(df_encoded, test_size=0.2, random_state=123)
        features = [col for col in train_df.columns if col != 'salary_in_usd']
        model = xgb.XGBRegressor(
            colsample_bytree=0.8, 
            learning_rate=0.1, 
            max_depth=5, 
            min_child_weight=1, 
            n_estimators=75, 
            subsample=0.8,
            random_state=123
        )
        model.fit(train_df[features], train_df['salary_in_usd'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model, df_encoded

model, df_encoded = load_or_train_model()
features = [col for col in df_encoded.columns if col != 'salary_in_usd']

# --- FRIENDLY LABELS AND MAPPINGS ---
# Job Titles
job_title_cols = [col for col in features if col.startswith('job_title_')]
job_title_names = [col.replace('job_title_', '') for col in job_title_cols]
job_title_map = {name: col for name, col in zip(job_title_names, job_title_cols)}

# Experience Levels
exp_level_map = {
    'Entry-level': 'experience_level_EN',
    'Mid-level': 'experience_level_MI',
    'Senior': 'experience_level_SE',
    'Executive': 'experience_level_EX',
}
exp_level_names = list(exp_level_map.keys())

# Employment Types
emp_type_map = {
    'Full-time': 'employment_type_FT',
    'Part-time': 'employment_type_PT',
    'Contract': 'employment_type_CT',
    'Freelance': 'employment_type_FL',
}
emp_type_names = list(emp_type_map.keys())

# Company Sizes
company_size_map = {
    'Small': 'company_size_S',
    'Medium': 'company_size_M',
    'Large': 'company_size_L',
}
company_size_names = list(company_size_map.keys())

# Remote work values with captions for endpoints
remote_ratio_values = list(range(0, 101))
remote_ratio_format = lambda x: f"{x}% (No remote work)" if x == 0 else f"{x}% (Fully remote)" if x == 100 else f"{x}%"

# Get lists of country names for dropdowns (sorted alphabetically)
employee_country_names = sorted([country_code_to_name.get(code, code) for code in set(employee_residence_map.values())])
company_country_names = sorted([country_code_to_name.get(code, code) for code in set(company_location_map.values())])

# Function to make prediction and store in session state
def make_prediction(selected_job_title, selected_experience, selected_employment, 
                  employee_residence_name, company_location_name, selected_company_size,
                  remote_ratio):
    # Convert country names to codes
    employee_residence_code = country_name_to_code.get(employee_residence_name)
    company_location_code = country_name_to_code.get(company_location_name)
    
    # Convert country codes to numeric indices for model input
    employee_residence_idx = employee_residence_code_to_idx.get(employee_residence_code)
    company_location_idx = company_location_code_to_idx.get(company_location_code)
    
    input_data = pd.DataFrame(columns=features, data=np.zeros((1, len(features))))
    # Set the selected features to 1 (for one-hot encoded features)
    input_data[job_title_map[selected_job_title]] = 1
    input_data[exp_level_map[selected_experience]] = 1
    input_data[emp_type_map[selected_employment]] = 1
    input_data[company_size_map[selected_company_size]] = 1
    input_data['employee_residence'] = employee_residence_idx
    input_data['company_location'] = company_location_idx
    input_data['remote_ratio'] = remote_ratio
    
    # Make prediction with bootstrap (using fixed value of 10 instead of session state)
    median_pred, lower_bound, upper_bound = bootstrap_predictions(model, input_data, n_bootstrap=10)
    
    # Store in session state
    st.session_state.median_pred = median_pred
    st.session_state.lower_bound = lower_bound
    st.session_state.upper_bound = upper_bound
    st.session_state.prediction_made = True

# Bootstrap predictions function
def bootstrap_predictions(model, input_data, n_bootstrap=10):
    """
    Generate bootstrap predictions for confidence intervals.
    Using a fixed number of bootstrap samples (default is 10).
    """
    df = df_encoded.copy()
    bootstrap_preds = np.zeros(n_bootstrap)
    
    # If we're in a resource-constrained environment like Docker,
    # we can use more efficient parameters
    bootstrap_model_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'n_estimators': 75,
        'subsample': 0.8,
        'random_state': 123,
        'n_jobs': 4,  # Use multiple cores efficiently
        'verbosity': 0  # Reduce logging overhead
    }
    
    for i in range(n_bootstrap):
        bootstrap_sample = df.sample(frac=1, replace=True, random_state=i)
        features = [col for col in bootstrap_sample.columns if col != 'salary_in_usd']
        bootstrap_model = xgb.XGBRegressor(**bootstrap_model_params)
        bootstrap_model.fit(bootstrap_sample[features], bootstrap_sample['salary_in_usd'])
        bootstrap_preds[i] = bootstrap_model.predict(input_data)[0]
    
    median_pred = np.median(bootstrap_preds)
    lower_bound = np.percentile(bootstrap_preds, 2.5)
    upper_bound = np.percentile(bootstrap_preds, 97.5)
    return median_pred, lower_bound, upper_bound

# --- TABS ---
tabs = st.tabs(["Prediction", "About/Extra"])

# --- PREDICTION TAB ---
with tabs[0]:
    st.markdown('<div class="stForm">', unsafe_allow_html=True)
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<h3 style="color: #00A651; text-align: center; margin-bottom: 1.5rem;">Job Details</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            selected_job_title = st.selectbox("Job Title", job_title_names, index=st.session_state.default_job_title_index)
            selected_experience = st.selectbox("Experience Level", exp_level_names, index=st.session_state.default_experience_index)
            selected_employment = st.selectbox("Employment Type", emp_type_names, index=st.session_state.default_employment_index)
            
            # Use select_slider with custom formatting function for remote work
            remote_ratio = st.select_slider(
                "Remote Work Percentage",
                options=remote_ratio_values,
                value=st.session_state.default_remote_ratio,
                format_func=remote_ratio_format
            )
                
        with col2:
            # Use country names for display
            selected_employee_residence = st.selectbox("Employee Residence Country", employee_country_names, index=employee_country_names.index("United States"))
            selected_company_location = st.selectbox("Company Location Country", company_country_names, index=company_country_names.index("United States"))
            selected_company_size = st.selectbox("Company Size", company_size_names, index=st.session_state.default_company_size_index)
        
        st.markdown('<div style="text-align: center; margin-top: 1.5rem;">', unsafe_allow_html=True)
        submitted = st.form_submit_button("📊 Predict Salary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submitted:
            # Use the slider value directly
            make_prediction(selected_job_title, selected_experience, selected_employment,
                         selected_employee_residence, selected_company_location, selected_company_size,
                         remote_ratio)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-predict on startup if no prediction has been made yet
    if not st.session_state.prediction_made:
        # Get default values
        default_job_title = job_title_names[st.session_state.default_job_title_index]
        default_experience = exp_level_names[st.session_state.default_experience_index]
        default_employment = emp_type_names[st.session_state.default_employment_index]
        default_employee_residence = "United States"
        default_company_location = "United States" 
        default_company_size = company_size_names[st.session_state.default_company_size_index]
        default_remote_ratio = st.session_state.default_remote_ratio
        
        # Make prediction with default values
        make_prediction(default_job_title, default_experience, default_employment,
                     default_employee_residence, default_company_location, default_company_size,
                     default_remote_ratio)
    
    # Display results if prediction was made
    if st.session_state.prediction_made:
        # Convert predictions to selected currency
        median_pred = st.session_state.median_pred
        lower_bound = st.session_state.lower_bound
        upper_bound = st.session_state.upper_bound
        
        # Convert to selected currency
        median_converted, currency_symbol = convert_currency(median_pred, st.session_state.currency)
        lower_converted, _ = convert_currency(lower_bound, st.session_state.currency)
        upper_converted, _ = convert_currency(upper_bound, st.session_state.currency)
        
        # Display currency info
        st.markdown(f"""
        <div style="background-color: #F0FAF5; border-left: 4px solid #00A651; padding: 1rem; margin: 1.5rem 0; border-radius: 4px;">
            <p style="margin: 0; color: #007840; font-weight: 500;">
                <span style="font-size: 1.1rem;">💱</span> Displaying salary in <span style="font-weight: bold;">{st.session_state.currency}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 style="color: #00A651; text-align: center;">Salary Prediction</h2>', unsafe_allow_html=True)
        
        # Calculate interval width as percentage of median for visual scaling
        interval_width = upper_converted - lower_converted
        median_value = median_converted
        relative_width_pct = min(95, max(30, (interval_width / median_value) * 100))
        
        # Calculate position for the marker within the interval
        relative_position = ((median_converted - lower_converted) / (upper_converted - lower_converted)) * 100
        
        # Determine the visual width of the interval bar
        visual_width = f"{relative_width_pct}%"
        
        # Calculate margins to center the bar
        margin_left = f"{(100 - relative_width_pct) / 2}%"
        
        st.markdown(f"""
        <div class="prediction-container" style="padding: 0 5%; margin: 0 auto; max-width: 1200px;">
            <div style="display: flex; align-items: center;">
                <div style="flex: 2; text-align: right; padding-right: 15px;">
                    <div class="currency-value" style="font-size: 16px; padding-top: 40px; color: #333;">{currency_symbol}{lower_converted:,.0f}</div>
                </div>
                <div style="flex: 8; position: relative;">
                    <div style="
                        position: relative;
                        height: 100px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 30px 0 10px 0;
                        width: 100%;
                    ">
                        <div class="prediction-range" style="
                            background-color: rgba(0, 166, 81, 0.2); 
                            border-radius: 10px; 
                            padding: 10px; 
                            text-align: center;
                            position: relative;
                            height: 80px;
                            width: {visual_width};
                            margin-left: {margin_left};
                        ">
                            <div class="prediction-value" style="
                                position: absolute;
                                top: -30px;
                                left: {relative_position}%;
                                transform: translateX(-50%);
                                background-color: #00A651;
                                color: white;
                                padding: 8px 16px;
                                border-radius: 15px;
                                font-weight: bold;
                                font-size: 20px;
                                white-space: nowrap;
                                z-index: 10;
                                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
                            ">
                                {currency_symbol}{median_converted:,.0f}
                            </div>
                            <div class="prediction-marker" style="
                                position: absolute;
                                top: 50%;
                                left: {relative_position}%;
                                transform: translate(-50%, -50%);
                                width: 6px;
                                height: 60px;
                                background-color: #00A651;
                                border-radius: 3px;
                                z-index: 5;
                            "></div>
                        </div>
                    </div>
                </div>
                <div style="flex: 2; text-align: left; padding-left: 15px;">
                    <div class="currency-value" style="font-size: 16px; padding-top: 40px; color: #333;">{currency_symbol}{upper_converted:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add explanation about the interval width
        interval_size_description = ""
        if relative_width_pct > 70:
            interval_size_description = "The wide prediction interval indicates higher uncertainty in the estimate."
        elif relative_width_pct > 40:
            interval_size_description = "This prediction has a moderate confidence interval."
        else:
            interval_size_description = "The narrow prediction interval suggests higher confidence in the estimate."
        
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; margin-top: 20px;">
            <div style="font-weight: bold; color: #00A651; font-size: 18px; margin-bottom: 10px;">
                Possible Salary Range
            </div>
            <div style="font-size: 14px; color: #555; margin-bottom: 15px; text-align: center;">
                {interval_size_description} 
                Interval width: <span class="currency-value" style="color: #00A651;">{currency_symbol}{interval_width:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: center; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
                <div style="margin-right: 30px;">
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: #00A651; margin-right: 8px; border-radius: 2px;"></span>
                    <span style="font-size: 16px;">Average Predicted Salary</span>
                </div>
                <div>
                    <span style="display: inline-block; width: 14px; height: 14px; background-color: rgba(0, 166, 81, 0.2); margin-right: 8px; border-radius: 2px;"></span>
                    <span style="font-size: 16px;">Possible Range</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- ABOUT/EXTRA TAB ---
with tabs[1]:
    st.markdown('<h2 style="color: #00A651;">Model Details & Information</h2>', unsafe_allow_html=True)
    
    # Model details card
    st.markdown("""
    <div class="card" style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
        <h3 style="color: #00A651; margin-bottom: 1rem;">💻 Technical Details</h3>
        <p>This salary prediction tool uses <strong>XGBoost regression</strong> with the following configuration:</p>
        <ul style="margin-bottom: 1rem;">
            <li><strong>Learning rate:</strong> 0.1</li>
            <li><strong>Max depth:</strong> 5</li>
            <li><strong>Number of estimators:</strong> 75</li>
            <li><strong>Model error (MAE):</strong> ~35,000 USD</li>
        </ul>
        <p>Confidence intervals are calculated using <strong>bootstrap resampling</strong>, where the model is retrained multiple times on resampled data to capture prediction uncertainty.</p>
    </div>
    """, unsafe_allow_html=True)
