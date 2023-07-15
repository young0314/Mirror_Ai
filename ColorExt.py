import cv2
import numpy as np

# 이미지를 로드합니다.
image = cv2.imread('image.jpg')

# 이미지를 BGR에서 HSV로 변환합니다.
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 피부색의 범위를 정의합니다.
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# 피부색 마스크를 생성합니다.
skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

# 원본 이미지에서 피부색만 추출합니다.
skin = cv2.bitwise_and(image, image, mask=skin_mask)

# 결과 이미지를 출력합니다.
cv2.imshow('Skin Detection', skin)
cv2.waitKey(0)
cv2.destroyAllWindows()