from flask import Flask, request, render_template, url_for, redirect, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import numpy as np
from models import db, User
import os

# Define the path to the Pictures folder
app = Flask(__name__, 
            static_url_path='', 
            static_folder='static')

# Configure the app
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the models and data
best_model = joblib.load('traffic_incident_model.pkl')
kmeans = joblib.load('location_cluster_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')
model_config = joblib.load('model_config.pkl')

# Get feature lists from config
categorical_features = model_config['categorical_features']
numeric_features = model_config['numeric_features']

@app.route('/')
def home():
    return render_template('TrafficWatch.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/data')
def data():
    return render_template('Data.html')

@app.route('/predict-tool')
@login_required
def predict_tool():
    return render_template('Prediction.html')

@app.route('/manage-db', methods=['GET'])
@login_required
def manage_db():
    if current_user.username != 'admin':
        flash('You are not authorized to access this page.')
        return redirect(url_for('home'))
    users = User.query.all()
    return render_template('manage_db.html', users=users)

@app.route('/add-user-admin', methods=['POST'])
@login_required
def add_user_admin():
    if current_user.username != 'admin':
        flash('You are not authorized to perform this action.')
        return redirect(url_for('manage_db'))

    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if User.query.filter_by(username=username).first():
        flash('Username already exists.')
        return redirect(url_for('manage_db'))
    
    if User.query.filter_by(email=email).first():
        flash('Email already registered.')
        return redirect(url_for('manage_db'))

    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    flash('User added successfully.')
    return redirect(url_for('manage_db'))

@app.route('/delete-user-admin/<int:user_id>', methods=['POST'])
@login_required
def delete_user_admin(user_id):
    if current_user.username != 'admin':
        flash('You are not authorized to perform this action.')
        return redirect(url_for('manage_db'))

    user_to_delete = User.query.get_or_404(user_id)
    if user_to_delete.username == 'admin':
        flash('Admin user cannot be deleted.')
        return redirect(url_for('manage_db'))
        
    db.session.delete(user_to_delete)
    db.session.commit()
    flash('User deleted successfully.')
    return redirect(url_for('manage_db'))

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get data from form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        hour = int(request.form['hour'])
        day_of_week = int(request.form['day_of_week'])
        is_weekend = int(request.form['is_weekend'])
        is_rush_hour = int(request.form['is_rush_hour'])
        city = request.form['city']
        
        # Determine time of day
        if 0 <= hour < 6:
            time_of_day = 'Night'
        elif 6 <= hour < 12:
            time_of_day = 'Morning'
        elif 12 <= hour < 18:
            time_of_day = 'Afternoon'
        else:
            time_of_day = 'Evening'
        
        # Create input data
        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'IsWeekend': [is_weekend],
            'IsRushHour': [is_rush_hour],
            'City': [city],
            'TimeOfDay': [time_of_day],
            'Month': [8],  # Default to August as in the data
            'Direction': ['Unknown']  # Default direction
        })
        
        # Determine location cluster
        input_data['LocationCluster'] = kmeans.predict(input_data[['Latitude', 'Longitude']])
        
        # One-hot encode the features
        input_encoded = pd.get_dummies(input_data[categorical_features + numeric_features])
        
        # Make sure all columns match the training columns
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Predict incident type
        prediction_idx = best_model.predict(input_encoded)[0]
        prediction = label_encoder.inverse_transform([prediction_idx])[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(input_encoded)[0]
            probabilities = {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}
            probabilities = {k: round(v * 100, 2) for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)}
        
        return render_template('Prediction.html', 
                              prediction=prediction, 
                              probabilities=probabilities,
                              form_data=request.form)
    
    except Exception as e:
        return render_template('Prediction.html', error=str(e), form_data=request.form)

# Create database tables before first request
with app.app_context():
    db.create_all()  # Actually execute the create_all() command
    # Create admin user if it doesn't exist
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        admin = User(username='admin', email='admin@example.com')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created.")

if __name__ == '__main__':
    app.run(debug=True)