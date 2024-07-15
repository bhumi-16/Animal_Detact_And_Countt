
from flask import Flask, request, render_template
import cv2
import numpy as np
import base64
from image_predictor import get_prediction

app = Flask(__name__)

@app.route('/')
def index():
    ing = "500px" 
    inw = "500px"
    return render_template('home.html', ing=ing,inw = inw,uptext = "Upload Image hare")

@app.route('/prediction', methods=['POST'])
def prediction():

    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    # Read the image file into a numpy array
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    img_rgb, animal_name = get_prediction(img)

    result_img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # Encode the original image to base64
    _, img_encoded = cv2.imencode('.png', img)
    original_image_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Encode the processed image to base64
    _, result_img_encoded = cv2.imencode('.png', result_img)
    processed_image_base64 = base64.b64encode(result_img_encoded).decode('utf-8')

    return render_template('home.html', 
                           original_image=f'data:image/png;base64,{original_image_base64}',
                           processed_image=f'data:image/png;base64,{processed_image_base64}',animal_name = f"Animal capture = {animal_name}",ing = "1%")

if __name__ == '__main__':
    app.run(debug=True)
