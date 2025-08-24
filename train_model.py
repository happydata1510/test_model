
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ====== 집값 예측 모델 학습 ======
print("집값 예측 모델 학습 중...")

# California Housing 데이터 로드
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest 모델 학습
model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# 성능 평가
test_score = model.score(X_test, y_test)
print(f"모델 학습 완료! 테스트 R² 점수: {test_score:.3f}")

# ====== 모델 저장 ======
joblib.dump(model, 'model.pkl')
print("학습된 모델을 model.pkl 파일로 저장했습니다.")
