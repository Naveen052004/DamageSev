import os
import cv2
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key in production

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Sample car models (replace with your actual models)
CAR_MODELS = [
    'Toyota Camry',
    'Honda Civic',
    'Ford Mustang',
    'BMW 3 Series',
    'Tesla Model 3',
    'Mercedes-Benz C-Class',
    'Audi A4',
    'Hyundai Sonata'
]

model = YOLO('bestOfNewNano.pt')
car_detect = YOLO('DetectCar.pt')

# Mock damage detection function (replace with your actual ML model)
def detect_damage_car(image_path, choosemodel):
    # Make prediction with YOLO model
    print(choosemodel)
    if choosemodel == 1:
        results = model.predict(source=image_path)
    else:
        results = car_detect.predict(source=image_path)
    prediction = []
    
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class indices

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])  # Convert to integers for OpenCV
            confidence = confidences[i]
            class_id = int(classes[i])
            label = f"{result.names[class_id]}: {confidence:.2f}"

            # Draw bounding boxes and labels on the image
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Append to prediction list with serializable data
            prediction.append({
                'damage_type': label,
                'confidence': float(confidence),  # Convert NumPy float to Python float
                'coordinates': [x1, y1, x2, y2],  # Convert tuple to list
            })
    
    # Save the image with bounding boxes
    output_path = image_path.replace(".jpg", "_annotated.jpg")  # Change the filename to avoid overwriting
    cv2.imwrite(output_path, img)
    
    return prediction, output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-name', methods=['POST'])
def submit_name():
    name = request.form.get('name')
    if name:
        session['user_name'] = name
        return redirect(url_for('upload'))
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_name' not in session:
        return redirect(url_for('index'))
    action = request.args.get('action')
    print(action, "hello")
    print(action is None, 'This is about action')
    return render_template('upload.html', choosing = 'detect_damage' if action is None else 'detect_car', user_name=session['user_name'])

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    print("Entered")
    if 'image' not in request.files: 
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Get prediction from ML model
        action = request.args.get('action')
        print(action, "Naveen")
        prediction, annotated_image_path = detect_damage_car(filepath, 1 if action =='detect_damage' else 2)
        session['prediction'] = prediction
        # session['image_path'] = os.path.join('uploads', filename).replace("\\", "/")
        print(annotated_image_path.replace("\\", "/"))
        session['image_path'] = annotated_image_path[7:].replace("\\", "/")
        # print(session['image_path'])
        return redirect(url_for('select_model'))
        # return jsonify({'error': 'Invalid file type'}), 400

    
    return jsonify({'error': 'Invalid file type'}), 400

# @app.route('/select-model')
# def select_model():
#     if 'user_name' not in session or 'prediction' not in session:
#         return redirect(url_for('index'))
#     return render_template('select_model.html', 
#                          user_name=session['user_name'],
#                          car_models=CAR_MODELS,
#                          prediction=session['prediction'],
#                          image_path=session['image_path'])

@app.route('/confirm-prediction')
def select_model():
    if 'user_name' not in session or 'prediction' not in session:
        return redirect(url_for('index'))
    
    # Check if 'prediction' is already a list, if not, deserialize it
    prediction = session['prediction']
    print("Before testing")
    # if isinstance(prediction, str):
    #     prediction = json.loads(prediction)  # Deserialize if it's a string
    # print("After testing")
    # print(prediction)
    # return jsonify({'error': 'Invalid file type'}), 400
    return render_template('confirm-prediction.html', 
                           user_name=session['user_name'],
                           car_models=CAR_MODELS,
                           prediction=prediction,  # Pass the prediction to the template
                           image_path=session['image_path'])


@app.route('/confirm-prediction', methods=['POST'])
def confirm_prediction():
    is_correct = request.form.get('is_correct') == 'true'
    selected_model = request.form.get('car_model')
    
    if is_correct and selected_model:
        # Store the confirmation (implement your storage logic here)
        return jsonify({'status': 'success', 'message': 'Prediction confirmed'})
    
    return jsonify({'status': 'error', 'message': 'Invalid submission'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)