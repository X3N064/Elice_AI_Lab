import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from elice_utils import EliceUtils

elice_utils = EliceUtils()

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)


# MNIST 데이터 세트를 불러옵니다.
mnist = tf.keras.datasets.mnist

# MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    

# Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
train_images, train_labels = train_images[:5000], train_labels[:5000]
test_images, test_labels = test_images[:1000], test_labels[:1000]


print("원본 학습용 이미지 데이터 형태: ",train_images.shape)
print("원본 평가용 이미지 데이터 형태: ",test_images.shape)
print("원본 학습용 label 데이터: ",train_labels)

# 첫 번째 샘플 데이터를 출력합니다.
plt.figure(figsize=(10, 10))
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.title("Training Data Sample")
plt.savefig("sample1.png")
elice_utils.send_image("sample1.png")

# 9개의 학습용 샘플 데이터를 출력합니다.
class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("sample2.png")
elice_utils.send_image("sample2.png")

"""
1. CNN 모델의 입력으로 사용할 수 있도록 (샘플개수, 가로픽셀, 세로픽셀, 1) 형태로 변환합니다.
"""
train_images = tf.expand_dims(train_images, -1)
test_images = tf.expand_dims(test_images, -1)

print("변환한 학습용 이미지 데이터 형태: ",train_images.shape)
print("변환한 평가용 이미지 데이터 형태: ",test_images.shape)