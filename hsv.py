import cv2
import numpy as np
import dlib

def extract_cheek(image_path, landmark_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)

    # 얼굴 랜드마크(landmark) 예측을 위해 dlib의 얼굴 탐지기 로드
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_path)

    # 이미지에서 얼굴 탐지
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # 첫 번째 얼굴에 대한 랜드마크 예측
    shape = predictor(gray, faces[0])
    landmarks = shape.parts()

    # 뺨 영역 추출을 위한 좌표 계산
    x1 = landmarks[4].x
    y1 = landmarks[2].y
    x2 = landmarks[7].x
    y2 = landmarks[8].y

    # 뺨 영역 추출
    cheek_region = image[y1:y2, x1:x2]

    return cheek_region

def extract_skin_color(cheek_region):
    # 뺨 영역에서 평균 피부 색상 추출
    average_color = np.mean(cheek_region, axis=(0, 1)).astype(int)

    return average_color

# 이미지 파일 경로 설정
image_path = 'C:/DEV/workspace/smart_mirror/landmark/asset/people_3.jpg'
# 얼굴 랜드마크 파일 경로 설정
landmark_path = 'C:/DEV/workspace/smart_mirror/landmark/asset/shape_predictor_68_face_landmarks.dat'

# 뺨 영역 추출
cheek_region = extract_cheek(image_path, landmark_path)

# 피부 색상 추출
skin_color = extract_skin_color(cheek_region)
print("피부 색상(RGB):", skin_color)

# 이미지 출력
cv2.imshow("Skin Detection", cheek_region)
cv2.waitKey(0)
cv2.destroyAllWindows()
