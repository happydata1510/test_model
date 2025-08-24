
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# 학습된 모델 로드
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json
            features = data['features']
            
            if len(features) != 8:
                return jsonify({'error': f'8개의 특성이 필요합니다. 받은 개수: {len(features)}'}), 400

            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            price_in_dollars = int(prediction * 100000)

            result = {
                'price': price_in_dollars,
                'success': True
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # GET 요청 시 예측 페이지를 렌더링합니다.
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
