
### Đoạn 1
import os
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 1. Kết nối Drive
drive.mount('/content/drive')

# 2. Đổi đường dẫn trỏ thẳng vào thư mục chứa tiền
# Đảm bảo thư mục "money" nằm ngay ngoài cùng của Drive
train_dir = "/content/drive/MyDrive/Bài giảng/DATA_AI/money"

if os.path.exists(train_dir):
    print("✅ Đã tìm thấy kho tiền!")
    print("Các mệnh giá nhận diện được:", os.listdir(train_dir))
else:
    print("❌ Không tìm thấy thư mục money, hãy kiểm tra lại đường dẫn.")

# 3. Tiền xử lý và tăng cường ảnh
img_width, img_height = 128, 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255, 
    rotation_range=20,       # Giảm độ xoay một chút vì tiền thường nằm ngang
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2 
)

print("\n--- Đang nạp dữ liệu Huấn luyện ---")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training' 
)

print("\n--- Đang nạp dữ liệu Kiểm tra ---")
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation' 
)

### Đoạn 2


num_classes = train_generator.num_classes 
print(f"Hệ thống tự động phát hiện {num_classes} mệnh giá tiền.")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax') 
])

model.summary()

### Đoạn 3


# Cấu hình AI
custom_optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=custom_optimizer,
              loss="categorical_crossentropy",
              metrics=['accuracy'])

# Phanh tự động
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,       
    verbose=1,        
    factor=0.5,       
    min_lr=0.00001    
)

epochs = 30 

print("BẮT ĐẦU HUẤN LUYỆN NHẬN DIỆN TIỀN...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[lr_reduction]
)

# Lưu lại với tên mới để không đè lên bài nhận diện khuôn mặt
model_save_path = "/content/drive/MyDrive/MoHinh_PhanLoai_Tien.h5"
model.save(model_save_path)

print(f"Đã huấn luyện xong và lưu mô hình tại: {model_save_path}")

### Đoạn 4


# 1. Nạp lại mô hình (Nếu bạn vừa chạy Ô số 3 xong thì biến 'model' vẫn còn,
# nhưng lệnh này giúp bạn mở lại Colab vào ngày mai vẫn test được luôn)
model_path = "/content/drive/MyDrive/MoHinh_PhanLoai_Tien.h5"

if os.path.exists(model_path):
    # Tải lại mô hình từ Drive
    # Bỏ qua cảnh báo "Compiled the loaded model..." nếu có xuất hiện nhé
    model_tien = load_model(model_path) 
else:
    print("Không tìm thấy file mô hình. Bạn hãy kiểm tra lại Ô code số 3!")

# Re-define necessary variables for train_generator
# These are typically defined in the data loading/preprocessing cell.
# Make sure these match the original definitions in cell z8fIGwYqQS8m if that cell is executed first.
train_dir = "/content/drive/MyDrive/Bài giảng/DATA_AI/money"
img_width, img_height = 128, 128
batch_size = 32

# Re-instantiate ImageDataGenerator (without validation split for prediction)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Re-instantiate train_generator to get class_indices
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Shuffle is not needed for getting class_indices
)

# 2. Đường dẫn đến bức ảnh tiền test
test_img_path = "/content/5000.jpg" 

if os.path.exists(test_img_path):
    # Tiền xử lý bức ảnh để ép về đúng chuẩn 128x128
    img = image.load_img(test_img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch_size
    img_array /= 255.0 # Chuẩn hóa điểm ảnh giống lúc train

    # 3. Yêu cầu AI dự đoán
    predictions = model_tien.predict(img_array)
    predicted_class_index = np.argmax(predictions) # Tìm vị trí có xác suất cao nhất

    # 4. Ánh xạ vị trí đó ra tên mệnh giá tiền
    # Lấy danh sách tên thư mục (chính là tên các mệnh giá: 1000, 2000, 5000...)
    class_labels = list(train_generator.class_indices.keys())
    predicted_money = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100 # Độ tự tin

    print(f"AI dự đoán đây là tờ: {predicted_money} VNĐ")
    print(f"Độ tự tin: {confidence:.2f}%")
else:
    print("Không tìm thấy ảnh test! Bạn hãy kiểm tra lại tên file và đường dẫn nhé.")