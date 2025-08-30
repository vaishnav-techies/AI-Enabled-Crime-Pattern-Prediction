# Data Preprocessing & Feature Engineering Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import joblib
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

class CrimeDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        
    def generate_synthetic_dataset(self, n_samples=10000):
        """
        Generate synthetic crime dataset for demonstration
        In production, replace with actual data loading
        """
        np.random.seed(42)
        
        print(" Generating synthetic crime dataset...")
        
        # Define categories
        locations = ['Downtown', 'North District', 'South District', 'East District', 
                    'West District', 'Suburban Area', 'Industrial Zone', 'University Area',
                    'Commercial District', 'Residential Area']
        
        crime_types = ['Theft', 'Vandalism', 'Assault', 'Burglary', 'Fraud', 
                      'Drug Offense', 'Vehicle Theft', 'Robbery', 'Domestic Violence',
                      'Cybercrime', 'Trespassing', 'Public Disorder']
        
        # Time-based features
        hours = list(range(24))
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        months = list(range(1, 13))
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        # Generate dates for the last 3 years
        start_date = datetime.now() - timedelta(days=3*365)
        
        data = []
        
        for i in range(n_samples):
            # Generate random date
            random_days = np.random.randint(0, 3*365)
            incident_date = start_date + timedelta(days=random_days)
            
            # Extract features
            hour = np.random.choice(hours, p=self._get_hour_probability())
            day_of_week = incident_date.strftime('%A')
            month = incident_date.month
            season = self._get_season(month)
            
            # Location with realistic distribution
            location = np.random.choice(locations, p=self._get_location_probability())
            
            # Crime type based on location and time
            crime_type = self._get_crime_type_by_context(location, hour, day_of_week)
            
            # Generate severity (1-10 scale)
            severity = self._calculate_severity(crime_type, location, hour)
            
            # Arrest probability
            arrest_made = np.random.random() < self._get_arrest_probability(crime_type, location)
            
            # Population density (crimes per 1000 people)
            pop_density = self._get_population_density(location)
            
            data.append({
                'Date': incident_date.strftime('%Y-%m-%d'),
                'Time': f"{hour:02d}:00",
                'Hour': hour,
                'Day_of_Week': day_of_week,
                'Month': month,
                'Season': season,
                'Location': location,
                'Crime_Type': crime_type,
                'Severity': severity,
                'Arrest_Made': arrest_made,
                'Population_Density': pop_density,
                'Weekend': day_of_week in ['Saturday', 'Sunday'],
                'Peak_Hours': hour in [18, 19, 20, 21, 22, 23],
                'Late_Night': hour in [0, 1, 2, 3, 4, 5]
            })
        
        df = pd.DataFrame(data)
        print(f" Generated {len(df)} crime records")
        return df
    
    def _get_hour_probability(self):
        # Higher probability during evening/night hours
        probs = np.array([0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
                         0.03, 0.04, 0.05, 0.04, 0.04, 0.04,  # 6-11
                         0.05, 0.05, 0.06, 0.06, 0.07, 0.08,  # 12-17
                         0.09, 0.08, 0.07, 0.06, 0.04, 0.03]) # 18-23
        return probs / probs.sum()
    
    def _get_location_probability(self):
        # Realistic crime distribution by area type
        return [0.18, 0.12, 0.10, 0.08, 0.08, 0.06, 0.15, 0.08, 0.10, 0.05]
    
    def _get_crime_type_by_context(self, location, hour, day):
        crime_weights = {
            'Theft': 0.25, 'Vandalism': 0.15, 'Assault': 0.12, 'Burglary': 0.10,
            'Fraud': 0.08, 'Drug Offense': 0.08, 'Vehicle Theft': 0.07,
            'Robbery': 0.06, 'Domestic Violence': 0.05, 'Cybercrime': 0.02,
            'Trespassing': 0.02, 'Public Disorder': 0.01
        }
        
        # Adjust weights based on context
        if location == 'Downtown':
            crime_weights['Theft'] *= 1.5
            crime_weights['Robbery'] *= 1.3
        elif location == 'Suburban Area':
            crime_weights['Burglary'] *= 2.0
            crime_weights['Theft'] *= 0.7
        elif location == 'Industrial Zone':
            crime_weights['Vandalism'] *= 1.8
            crime_weights['Trespassing'] *= 2.5
        
        if hour >= 22 or hour <= 4:
            crime_weights['Burglary'] *= 1.5
            crime_weights['Vehicle Theft'] *= 1.3
        
        crimes = list(crime_weights.keys())
        weights = list(crime_weights.values())
        weights = np.array(weights) / np.sum(weights)
        
        return np.random.choice(crimes, p=weights)
    
    def _calculate_severity(self, crime_type, location, hour):
        base_severity = {
            'Theft': 4, 'Vandalism': 3, 'Assault': 7, 'Burglary': 6,
            'Fraud': 5, 'Drug Offense': 6, 'Vehicle Theft': 5,
            'Robbery': 8, 'Domestic Violence': 8, 'Cybercrime': 4,
            'Trespassing': 2, 'Public Disorder': 3
        }
        
        severity = base_severity.get(crime_type, 5)
        
        # Adjust based on location
        if location in ['Downtown', 'Industrial Zone']:
            severity += 1
        elif location == 'Suburban Area':
            severity -= 1
        
        # Adjust based on time
        if hour >= 22 or hour <= 4:
            severity += 1
        
        return np.clip(severity + np.random.randint(-1, 2), 1, 10)
    
    def _get_arrest_probability(self, crime_type, location):
        base_prob = {
            'Theft': 0.35, 'Vandalism': 0.25, 'Assault': 0.65, 'Burglary': 0.45,
            'Fraud': 0.40, 'Drug Offense': 0.70, 'Vehicle Theft': 0.30,
            'Robbery': 0.55, 'Domestic Violence': 0.60, 'Cybercrime': 0.20,
            'Trespassing': 0.50, 'Public Disorder': 0.45
        }
        
        prob = base_prob.get(crime_type, 0.4)
        
        if location == 'Downtown':
            prob += 0.1
        elif location == 'Suburban Area':
            prob += 0.05
        
        return np.clip(prob, 0, 1)
    
    def _get_population_density(self, location):
        density_map = {
            'Downtown': 8500, 'North District': 4200, 'South District': 3800,
            'East District': 3500, 'West District': 4000, 'Suburban Area': 2100,
            'Industrial Zone': 1200, 'University Area': 6800,
            'Commercial District': 7200, 'Residential Area': 2800
        }
        base_density = density_map.get(location, 3000)
        return base_density + np.random.randint(-500, 500)
    
    def _get_season(self, month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'
    
    def preprocess_data(self, df):
        """
        Handle missing values and data cleaning
        """
        print(" Preprocessing data...")
        
        # Handle missing values
        df = df.copy()
        
        # Fill missing categorical values with mode
        categorical_cols = ['Location', 'Crime_Type', 'Day_of_Week', 'Season']
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Fill missing numerical values with median
        numerical_cols = ['Hour', 'Severity', 'Population_Density']
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        print(f" Cleaned dataset: {len(df)} records")
        return df
    
    def feature_engineering(self, df):
        """
        Create new features for better model performance
        """
        print(" Engineering features...")
        
        df = df.copy()
        
        # Time-based features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # Day of week features
        df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Month cyclical features
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Crime severity index based on type
        severity_index = {
            'Theft': 1.2, 'Vandalism': 0.8, 'Assault': 2.5, 'Burglary': 1.8,
            'Fraud': 1.5, 'Drug Offense': 2.0, 'Vehicle Theft': 1.6,
            'Robbery': 3.0, 'Domestic Violence': 2.8, 'Cybercrime': 1.3,
            'Trespassing': 0.6, 'Public Disorder': 1.0
        }
        df['Crime_Severity_Index'] = df['Crime_Type'].map(severity_index)
        
        # Location risk score
        location_risk = {
            'Downtown': 2.5, 'North District': 1.8, 'South District': 1.5,
            'East District': 1.3, 'West District': 1.6, 'Suburban Area': 0.8,
            'Industrial Zone': 2.0, 'University Area': 1.4,
            'Commercial District': 2.2, 'Residential Area': 1.0
        }
        df['Location_Risk_Score'] = df['Location'].map(location_risk)
        
        # Interaction features
        df['Risk_Time_Interaction'] = df['Location_Risk_Score'] * df['Peak_Hours'].astype(int)
        df['Severity_Density_Ratio'] = df['Severity'] / (df['Population_Density'] / 1000)
        
        # Clustering locations by crime patterns
        location_features = df.groupby('Location').agg({
            'Severity': 'mean',
            'Arrest_Made': 'mean',
            'Peak_Hours': lambda x: x.astype(int).mean()
        }).reset_index()
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        location_features['Cluster'] = kmeans.fit_predict(
            location_features[['Severity', 'Arrest_Made', 'Peak_Hours']]
        )
        
        # Map clusters back to main dataframe
        cluster_map = dict(zip(location_features['Location'], location_features['Cluster']))
        df['Location_Cluster'] = df['Location'].map(cluster_map)
        
        # Save cluster model
        joblib.dump(kmeans, 'models/location_cluster_model.pkl')
        joblib.dump(cluster_map, 'models/cluster_mapping.pkl')
        
        print(f" Feature engineering complete. New features: {len(df.columns)} total")
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical variables for machine learning
        """
        print(" Encoding categorical features...")
        
        df = df.copy()
        categorical_features = ['Location', 'Crime_Type', 'Day_of_Week', 'Season']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_Encoded'] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
        
        # One-hot encode high cardinality features
        location_encoded = pd.get_dummies(df['Location'], prefix='Loc')
        crime_encoded = pd.get_dummies(df['Crime_Type'], prefix='Crime')
        
        # Combine with main dataframe
        df = pd.concat([df, location_encoded, crime_encoded], axis=1)
        
        print(" Categorical encoding complete")
        return df
    
    def create_risk_levels(self, df):
        """
        Create risk level target variable
        """
        print(" Creating risk level targets...")
        
        df = df.copy()
        
        # Calculate composite risk score
        risk_components = [
            df['Severity'] / 10,  # Normalize severity
            df['Location_Risk_Score'] / 3,  # Normalize location risk
            df['Peak_Hours'].astype(int) * 0.3,  # Time factor
            df['Weekend'].astype(int) * 0.2,  # Weekend factor
        ]
        
        df['Risk_Score'] = sum(risk_components) / len(risk_components)
        
        # Create risk level categories
        df['Risk_Level'] = pd.cut(
            df['Risk_Score'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Encode risk levels
        le_risk = LabelEncoder()
        df['Risk_Level_Encoded'] = le_risk.fit_transform(df['Risk_Level'])
        self.label_encoders['Risk_Level'] = le_risk
        
        print(" Risk level creation complete")
        return df
    
    def prepare_model_data(self, df):
        """
        Prepare features and targets for model training
        """
        print(" Preparing model data...")
        
        # Feature columns for different prediction tasks
        base_features = [
            'Hour', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
            'Month', 'Month_Sin', 'Month_Cos', 'Severity', 'Population_Density',
            'Location_Encoded', 'Day_of_Week_Encoded', 'Season_Encoded',
            'Crime_Severity_Index', 'Location_Risk_Score',
            'Risk_Time_Interaction', 'Severity_Density_Ratio',
            'Location_Cluster', 'Weekend', 'Peak_Hours', 'Late_Night'
        ]
        
        # Risk prediction features (without crime type)
        risk_features = [f for f in base_features if 'Crime' not in f]
        
        # Crime type prediction features (including risk level)
        crime_features = base_features + ['Risk_Level_Encoded']
        
        # Prepare datasets
        X_risk = df[risk_features]
        y_risk = df['Risk_Level_Encoded']
        
        X_crime = df[crime_features]
        y_crime = df['Crime_Type_Encoded']
        
        print(f" Model data prepared - Risk features: {len(risk_features)}, Crime features: {len(crime_features)}")
        return X_risk, y_risk, X_crime, y_crime
    
    def train_models(self, X_risk, y_risk, X_crime, y_crime):
        """
        Train multiple models for different prediction tasks
        """
        print(" Training machine learning models...")
        
        # Split data
        X_risk_train, X_risk_test, y_risk_train, y_risk_test = train_test_split(
            X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk
        )
        
        X_crime_train, X_crime_test, y_crime_train, y_crime_test = train_test_split(
            X_crime, y_crime, test_size=0.2, random_state=42, stratify=y_crime
        )
        
        # Scale features
        X_risk_train_scaled = self.scaler.fit_transform(X_risk_train)
        X_risk_test_scaled = self.scaler.transform(X_risk_test)
        
        # Models for risk prediction
        risk_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        print(" Training risk prediction models...")
        best_risk_model = None
        best_risk_score = 0
        
        for name, model in risk_models.items():
            if name == 'Logistic Regression':
                model.fit(X_risk_train_scaled, y_risk_train)
                score = model.score(X_risk_test_scaled, y_risk_test)
            else:
                model.fit(X_risk_train, y_risk_train)
                score = model.score(X_risk_test, y_risk_test)
            
            print(f"  {name}: {score:.4f}")
            
            if score > best_risk_score:
                best_risk_score = score
                best_risk_model = (name, model)
        
        # Crime type prediction
        print(" Training crime type prediction models...")
        crime_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        crime_model.fit(X_crime_train, y_crime_train)
        crime_score = crime_model.score(X_crime_test, y_crime_test)
        
        print(f"  Crime Type Model: {crime_score:.4f}")
        
        # Store models
        self.models['risk'] = best_risk_model[1]
        self.models['crime'] = crime_model
        
        # Generate detailed evaluation
        self._evaluate_models(X_risk_test, y_risk_test, X_crime_test, y_crime_test)
        
        print(f" Model training complete - Best risk model: {best_risk_model[0]}")
        return self.models
    
    def _evaluate_models(self, X_risk_test, y_risk_test, X_crime_test, y_crime_test):
        """
        Detailed model evaluation
        """
        print("\n DETAILED MODEL EVALUATION")
        print("="*50)
        
        # Risk model evaluation
        if isinstance(self.models['risk'], LogisticRegression):
            X_risk_test = self.scaler.transform(X_risk_test)
        
        risk_pred = self.models['risk'].predict(X_risk_test)
        
        print("\n RISK PREDICTION MODEL:")
        print(f"Accuracy: {accuracy_score(y_risk_test, risk_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_risk_test, risk_pred, 
                                   target_names=['Low', 'Medium', 'High']))
        
        # Crime type model evaluation
        crime_pred = self.models['crime'].predict(X_crime_test)
        
        print("\n CRIME TYPE PREDICTION MODEL:")
        print(f"Accuracy: {accuracy_score(y_crime_test, crime_pred):.4f}")
        
        # Feature importance
        if hasattr(self.models['risk'], 'feature_importances_'):
            print("\n TOP RISK PREDICTION FEATURES:")
            feature_names = X_risk_test.columns if hasattr(X_risk_test, 'columns') else [f'Feature_{i}' for i in range(len(X_risk_test.columns))]
            importances = self.models['risk'].feature_importances_
            feature_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            
            for name, importance in feature_imp[:10]:
                print(f"  {name}: {importance:.4f}")
    
    def save_models(self, output_dir='models'):
        """
        Save trained models and encoders
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f" Saving models to {output_dir}/...")
        
        # Save models
        joblib.dump(self.models['risk'], f'{output_dir}/risk_prediction_model.pkl')
        joblib.dump(self.models['crime'], f'{output_dir}/crime_type_model.pkl')
        
        # Save encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f'{output_dir}/{name.lower()}_encoder.pkl')
        
        # Save scaler
        joblib.dump(self.scaler, f'{output_dir}/feature_scaler.pkl')
        
        print(" All models and encoders saved successfully")
    
    def generate_processed_dataset(self, output_file='crime_data_processed.csv'):
        """
        Complete pipeline: generate, preprocess, and save dataset
        """
        print(" STARTING CRIME DATA PROCESSING PIPELINE")
        print("="*50)
        
        # Generate synthetic data
        df = self.generate_synthetic_dataset(n_samples=10000)
        
        # Preprocessing steps
        df = self.preprocess_data(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        df = self.create_risk_levels(df)
        
        # Save processed dataset
        df.to_csv(output_file, index=False)
        print(f" Processed dataset saved as: {output_file}")
        
        # Prepare model data and train
        X_risk, y_risk, X_crime, y_crime = self.prepare_model_data(df)
        self.train_models(X_risk, y_risk, X_crime, y_crime)
        
        # Save models
        self.save_models()
        
        # Generate summary statistics
        self._generate_summary_report(df)
        
        print("\n PIPELINE COMPLETED SUCCESSFULLY!")
        return df
    
    def _generate_summary_report(self, df):
        """
        Generate comprehensive data summary
        """
        print("\n DATASET SUMMARY REPORT")
        print("="*50)
        
        print(f" Total Records: {len(df):,}")
        print(f" Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f" Unique Locations: {df['Location'].nunique()}")
        print(f" Crime Types: {df['Crime_Type'].nunique()}")
        
        print(f"\n Risk Distribution:")
        risk_dist = df['Risk_Level'].value_counts(normalize=True) * 100
        for level, pct in risk_dist.items():
            print(f"  {level}: {pct:.1f}%")
        
        print(f"\n Top Crime Types:")
        crime_dist = df['Crime_Type'].value_counts().head()
        for crime, count in crime_dist.items():
            print(f"  {crime}: {count:,}")
        
        print(f"\n High-Risk Locations:")
        location_risk = df.groupby('Location')['Risk_Score'].mean().sort_values(ascending=False)
        for location, risk in location_risk.head().items():
            print(f"  {location}: {risk:.3f}")

def main():
    """
    Main execution function
    """
    processor = CrimeDataProcessor()
    
    # Run complete pipeline
    processed_df = processor.generate_processed_dataset('crime_data_clean.csv')
    
    # Optional: Generate visualizations
    print("\n Generating visualizations...")
    create_eda_plots(processed_df)

def create_eda_plots(df):
    """
    Create exploratory data analysis plots
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Crime Data Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Crime distribution by type
    crime_counts = df['Crime_Type'].value_counts()
    axes[0,0].bar(range(len(crime_counts)), crime_counts.values)
    axes[0,0].set_title('Crime Distribution by Type')
    axes[0,0].set_xlabel('Crime Type')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Risk level distribution
    df['Risk_Level'].value_counts().plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
    axes[0,1].set_title('Risk Level Distribution')
    
    # Hourly crime pattern
    hourly_crimes = df.groupby('Hour').size()
    axes[0,2].plot(hourly_crimes.index, hourly_crimes.values, marker='o')
    axes[0,2].set_title('Crime Patterns by Hour')
    axes[0,2].set_xlabel('Hour of Day')
    axes[0,2].set_ylabel('Number of Crimes')
    axes[0,2].grid(True)
    
    # Location risk heatmap
    location_time = df.pivot_table(values='Risk_Score', index='Location', columns='Peak_Hours', aggfunc='mean')
    sns.heatmap(location_time, ax=axes[1,0], cmap='YlOrRd', annot=True, fmt='.2f')
    axes[1,0].set_title('Risk Score: Location vs Peak Hours')
    
    # Seasonal patterns
    seasonal_crimes = df.groupby('Season')['Crime_Type'].count()
    axes[1,1].bar(seasonal_crimes.index, seasonal_crimes.values)
    axes[1,1].set_title('Seasonal Crime Patterns')
    axes[1,1].set_ylabel('Number of Crimes')
    
    # Severity distribution
    axes[1,2].hist(df['Severity'], bins=10, edgecolor='black', alpha=0.7)
    axes[1,2].set_title('Crime Severity Distribution')
    axes[1,2].set_xlabel('Severity (1-10)')
    axes[1,2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('crime_data_analysis.png', dpi=300, bbox_inches='tight')
    print(" Visualization saved as 'crime_data_analysis.png'")
    plt.show()

if __name__ == "__main__":
    main()
