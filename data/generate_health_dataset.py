"""
Comprehensive Health Fitness Dataset Generator
Creates a realistic health fitness dataset for federated learning
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_health_fitness_dataset(num_participants=3000, days_per_participant=200):
    """Generate comprehensive health fitness dataset"""
    
    print(f"Generating health fitness dataset for {num_participants} participants...")
    
    all_data = []
    
    # Activity types
    activity_types = [
        'Swimming', 'Basketball', 'Weight Training', 'Yoga', 'Dancing', 
        'Running', 'Cycling', 'Walking', 'Tennis', 'Pilates'
    ]
    
    # Health conditions
    health_conditions = ['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Obesity']
    
    # Smoking status
    smoking_statuses = ['Never', 'Former', 'Current']
    
    # Genders
    genders = ['M', 'F', 'Other']
    
    for participant_id in range(1, num_participants + 1):
        if participant_id % 500 == 0:
            print(f"  Generated data for {participant_id} participants...")
        
        # Participant characteristics
        age = random.randint(18, 80)
        gender = random.choice(genders)
        height_cm = random.normalvariate(170, 10) if gender == 'M' else random.normalvariate(160, 8)
        weight_kg = random.normalvariate(75, 15) if gender == 'M' else random.normalvariate(65, 12)
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Health characteristics
        health_condition = random.choice(health_conditions)
        smoking_status = random.choice(smoking_statuses)
        
        # Base fitness level (0-20 scale)
        base_fitness = random.normalvariate(10, 3)
        base_fitness = max(0, min(20, base_fitness))
        
        # Generate daily data
        start_date = datetime(2024, 1, 1)
        
        for day in range(days_per_participant):
            current_date = start_date + timedelta(days=day)
            
            # Daily variations
            daily_fitness_variation = random.normalvariate(0, 1)
            fitness_level = max(0, min(20, base_fitness + daily_fitness_variation))
            
            # Activity
            activity_type = random.choice(activity_types)
            duration_minutes = random.normalvariate(45, 20)
            duration_minutes = max(10, min(180, duration_minutes))
            
            # Intensity based on activity type
            if activity_type in ['Swimming', 'Basketball', 'Running']:
                intensity = random.choice(['High', 'Medium'])
            elif activity_type in ['Yoga', 'Pilates', 'Walking']:
                intensity = random.choice(['Low', 'Medium'])
            else:
                intensity = random.choice(['Low', 'Medium', 'High'])
            
            # Calories burned based on activity and intensity
            base_calories = {
                'Swimming': 8, 'Basketball': 7, 'Weight Training': 5,
                'Yoga': 3, 'Dancing': 4, 'Running': 10, 'Cycling': 6,
                'Walking': 3, 'Tennis': 7, 'Pilates': 2
            }
            
            intensity_multiplier = {'Low': 0.7, 'Medium': 1.0, 'High': 1.3}
            calories_burned = base_calories[activity_type] * intensity_multiplier[intensity] * (duration_minutes / 60)
            
            # Heart rate based on activity and fitness level
            resting_hr = random.normalvariate(70, 10)
            if intensity == 'High':
                avg_heart_rate = resting_hr + random.normalvariate(60, 15)
            elif intensity == 'Medium':
                avg_heart_rate = resting_hr + random.normalvariate(40, 10)
            else:
                avg_heart_rate = resting_hr + random.normalvariate(20, 8)
            
            avg_heart_rate = max(60, min(200, avg_heart_rate))
            
            # Sleep and stress
            hours_sleep = random.normalvariate(7.5, 1.5)
            hours_sleep = max(4, min(12, hours_sleep))
            
            stress_level = random.normalvariate(5, 2)
            stress_level = max(1, min(10, stress_level))
            
            # Daily steps
            base_steps = random.normalvariate(8000, 2000)
            if activity_type in ['Running', 'Walking', 'Cycling']:
                base_steps += random.normalvariate(3000, 1000)
            
            daily_steps = max(1000, int(base_steps))
            
            # Hydration
            hydration_level = random.normalvariate(2.5, 0.5)
            hydration_level = max(1, min(5, hydration_level))
            
            # Blood pressure
            blood_pressure_systolic = random.normalvariate(120, 15)
            blood_pressure_diastolic = random.normalvariate(80, 10)
            
            # Create record
            record = {
                'participant_id': participant_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': gender,
                'height_cm': round(height_cm, 1),
                'weight_kg': round(weight_kg, 1),
                'activity_type': activity_type,
                'duration_minutes': int(duration_minutes),
                'intensity': intensity,
                'calories_burned': round(calories_burned, 1),
                'avg_heart_rate': int(avg_heart_rate),
                'hours_sleep': round(hours_sleep, 1),
                'stress_level': round(stress_level, 1),
                'daily_steps': daily_steps,
                'hydration_level': round(hydration_level, 1),
                'bmi': round(bmi, 1),
                'resting_heart_rate': int(resting_hr),
                'blood_pressure_systolic': round(blood_pressure_systolic, 1),
                'blood_pressure_diastolic': round(blood_pressure_diastolic, 1),
                'health_condition': health_condition,
                'smoking_status': smoking_status,
                'fitness_level': round(fitness_level, 2)
            }
            
            all_data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_path = 'data/health_fitness_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df):,} records for {num_participants} participants")
    print(f"📁 Saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"  Total records: {len(df):,}")
    print(f"  Participants: {df['participant_id'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()} years")
    print(f"  Fitness level range: {df['fitness_level'].min():.2f}-{df['fitness_level'].max():.2f}")
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    df = generate_health_fitness_dataset(num_participants=3000, days_per_participant=200)
    
    print("\n🎯 Dataset generation complete!")
    print("You can now run the enhanced federated learning pipeline.")
