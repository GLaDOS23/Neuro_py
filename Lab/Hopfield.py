import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Загрузка изображений
def load_and_preprocess_images():
    images = []
    for i in range(5):
        img = cv2.imread(f'{i}.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (5, 6))  #6 на 5
        _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        images.append(binary_img)
    return images

'''
def load_images_from_folder(folder, txt_file, img_size=(5, 6)):
    images_dict = {}
   
    #with open(txt_file, 'r', encoding='utf-8') as file:
        #lines = file.readlines()
    #for line in lines:
        #filename, class_id = line.strip().split(':')
        #class_id = int(class_id)
        #image_path = os.path.join(folder, filename)
        #image = cv2.imread(image_path)
       
        image = cv2.resize(image, img_size)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flattened_image = gray_image.flatten()
        if class_id not in images_dict:
            images_dict[class_id] = []
        images_dict[class_id].append(flattened_image)
    images_array = []
    labels_array = []
    for class_id in sorted(images_dict.keys()):
        images_array.extend(images_dict[class_id])
        labels_array.extend([class_id] * len(images_dict[class_id]))
    return np.array(images_array), np.array(labels_array)
'''

#сеть Хопфилда
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = torch.zeros(size, size)
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = 2 * pattern - 1  # Преобразование в -1 1
            self.weights += torch.outer(pattern, pattern)
        self.weights /= len(patterns)
        
        self.weights.fill_diagonal_(0)  #обнуление диагонали
    
    def recall(self, input_pattern, max_steps=10):
        pattern = 2 * input_pattern - 1 
        for _ in range(max_steps):
            new_pattern = torch.sign(self.weights @ pattern)
            if torch.all(new_pattern == pattern):
                break
            pattern = new_pattern
        return (pattern + 1) / 2  # Обратное преобразование в {0, 1}

# Визуализация результатов
def plot_results(original, distorted, recalled, error_rate):
    plt.figure(figsize=(12, 4))
    
    # Инвертируем изображения для правильного отображения
    original = 1 - original
    distorted = 1 - distorted
    recalled = 1 - recalled
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='binary')
    plt.title('Вход')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(distorted, cmap='binary')
    plt.title(f'Искажённый ({error_rate:.0%})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(recalled, cmap='binary')
    plt.title('Выход')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # загрузка и подготовка данных
    images = load_and_preprocess_images()
    patterns = [ torch.from_numpy(img.flatten()).float() for img in images]
    
    #Создание модели
    network = HopfieldNetwork(6 * 5)
    #обучение
    network.train(patterns)
    
    # тестирование с нарастающими ошибками
    test_idx = 0  #тестовый символ
    original_img = images[test_idx]
    original_vec = patterns[test_idx]
    
    for error_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #Создание шума на изображении
        distortion_mask = torch.rand_like(original_vec) < error_rate
        distorted_vec = original_vec.clone()
        distorted_vec[distortion_mask] = 1 - distorted_vec[distortion_mask]
        distorted_img = distorted_vec.view((6, 5)).numpy() 
        
        # восстановление
        recalled_vec = network.recall(distorted_vec)
        recalled_img = recalled_vec.view((6, 5)).numpy()
        
        plot_results(original_img, distorted_img, recalled_img, error_rate)
        
        # Проверка на химерные образы
        is_chimera = True
        for pattern in patterns:
            if torch.all(recalled_vec == pattern):
                is_chimera = False
                break
        
        if is_chimera:
            print(f"Обнаружен химерный образ при {error_rate:.0%}")

if __name__ == "__main__":
    main()
