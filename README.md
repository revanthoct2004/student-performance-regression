⚙️ Installation and Setup
Step 1:
Clone this repository
git clone https://github.com/<your-username>/student-score-predictor.git
cd student-score-predictor

Step 2:
Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate     # For Windows

Step 3: 
Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt file yet, create one with this content:

pandas
numpy
matplotlib
scikit-learn

Step 4:
Run the project
python student_score_predictor.py

🧪 Example Output
Dataset (first 5 rows):
   Hours  Score
0    2.5     21
1    5.1     47
...

Model coefficient (slope): 9.84
Model intercept: 3.03
Predicted score for 9.25 study hours: 92.56

MSE: 19.12
R2 Score: 0.95


The program also displays a graph showing:<img width="1397" height="947" alt="image" src="https://github.com/user-attachments/assets/d7d1d951-81fa-4b51-a57f-f653f95f46c4" /


Data points (hours vs score)

Regression line (model prediction)
