import shutil
import cv2
import random
import os
import numpy as np

from alive_progress import alive_bar


def copy_files(input_folder, output_folder_no_mask, output_folder_mask):
    
    with alive_bar(force_tty=True) as bar:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                source_file_path = os.path.join(root, file)
                if file.endswith('_mask.tif'):
                    destination_folder = output_folder_mask
                else:
                    destination_folder = output_folder_no_mask
                os.makedirs(destination_folder, exist_ok=True)
                try:
                    shutil.copy(source_file_path, destination_folder)
                except shutil.SameFileError:
                   pass
                bar()
    
    print('Copied all files')

    
def convert_mask_to_poly_annotation(image_dir, output_dir):
    with alive_bar(force_tty=True) as bar:
        for filename in os.listdir(image_dir):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif'):
                image_id = os.path.splitext(filename)[0]
                image_path = os.path.join(image_dir, filename)
                seg_list = []

                # Load binary mask image and convert to numpy array
                mask = cv2.imread(image_path)
                img_h, img_w = mask.shape[:2]
                grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                (T, thresh) = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)

                # Create annotation for each object in the image
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
                    segmentation = contour.squeeze().tolist()
                    seg_list.append(segmentation)

                if seg_list:
                    max_poly = max(seg_list, key=len) 
                    normalized = []
                    for x,y in max_poly:
                        x = (x) / img_w
                        y = (y) / img_h
                        normalized.append(x)
                        normalized.append(y)

                    seg = '0 '+' '.join(str(item) for item in normalized)

                    # Save COCO dataset to file
                    with open(os.path.join(output_dir, f'{image_id}.txt'), 'w') as f:
                        f.write(seg)
                else:
                    with open(os.path.join(output_dir, f'{image_id}.txt'), 'w') as f:
                        f.write('')  
            
            bar()

            

def split_folder_content(input_folder_images, input_folder_labels, input_folder_masks, dataset_dir, split_percentage_val, split_percentage_test, file_ending):
    all_files = os.listdir(input_folder_images)
    file_list = [ fname for fname in all_files if fname.endswith(file_ending)]
    random.shuffle(file_list)
    
    train, val, test = np.split(file_list, [int(len(file_list)*split_percentage_val), int(len(file_list)*split_percentage_test)])

    # Training images folder 
    train_images = os.path.join(os.path.join(dataset_dir, 'train'),'images')
    train_labels = os.path.join(os.path.join(dataset_dir, 'train'),'labels')
    train_masks = os.path.join(os.path.join(dataset_dir, 'train'),'masks')

    # Validation images folder
    val_images = os.path.join(os.path.join(dataset_dir, 'valid'),'images')
    val_labels = os.path.join(os.path.join(dataset_dir, 'valid'),'labels')
    val_masks = os.path.join(os.path.join(dataset_dir, 'valid'),'masks')
    
    # test images folder
    test_images = os.path.join(os.path.join(dataset_dir, 'test'),'images')
    test_labels = os.path.join(os.path.join(dataset_dir, 'test'),'labels')
    test_masks = os.path.join(os.path.join(dataset_dir, 'test'),'masks')    
    
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(train_masks, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    os.makedirs(val_masks, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)
    os.makedirs(test_masks, exist_ok=True)
    
    
    with alive_bar(force_tty=True) as bar:
        for file in train:
            # copy image
            source_image_path = os.path.join(input_folder_images, file)
            destination_image_path = os.path.join(train_images, file)
            shutil.copy(source_image_path, destination_image_path)

            # copy corresponding label
            label = os.path.splitext(file)[0] + '_mask.txt'
            mask = os.path.splitext(file)[0] + '_mask.tif'
            source_label_path = os.path.join(input_folder_labels, label)
            source_mask_path = os.path.join(input_folder_masks, mask)
            destination_label_path = os.path.join(train_labels, os.path.splitext(file)[0] + '.txt')
            destination_mask_path = os.path.join(train_masks, os.path.splitext(file)[0] + '.tif')
            shutil.copy(source_label_path, destination_label_path)
            shutil.copy(source_mask_path, destination_mask_path)
            bar()


    with alive_bar(force_tty=True) as bar:
        for file in val:
            source_file_path = os.path.join(input_folder_images, file)
            destination_file_path = os.path.join(val_images, file)
            shutil.copy(source_file_path, destination_file_path)

            # copy corresponding label
            label = os.path.splitext(file)[0] + '_mask.txt'
            mask = os.path.splitext(file)[0] + '_mask.tif'
            source_label_path = os.path.join(input_folder_labels, label)
            source_mask_path = os.path.join(input_folder_masks, mask)
            destination_label_path = os.path.join(val_labels, os.path.splitext(file)[0] + '.txt')
            destination_mask_path = os.path.join(val_masks, os.path.splitext(file)[0] + '.tif')
            shutil.copy(source_label_path, destination_label_path)
            shutil.copy(source_mask_path, destination_mask_path)
            bar()
            
            
    with alive_bar(force_tty=True) as bar:
        for file in test:
            source_file_path = os.path.join(input_folder_images, file)
            destination_file_path = os.path.join(test_images, file)
            shutil.copy(source_file_path, destination_file_path)

            # copy corresponding label
            label = os.path.splitext(file)[0] + '_mask.txt'
            mask = os.path.splitext(file)[0] + '_mask.tif'
            source_label_path = os.path.join(input_folder_labels, label)
            source_mask_path = os.path.join(input_folder_masks, mask)
            destination_label_path = os.path.join(test_labels, os.path.splitext(file)[0] + '.txt')
            destination_mask_path = os.path.join(test_masks, os.path.splitext(file)[0] + '.tif')
            shutil.copy(source_label_path, destination_label_path)
            shutil.copy(source_mask_path, destination_mask_path)
            bar()