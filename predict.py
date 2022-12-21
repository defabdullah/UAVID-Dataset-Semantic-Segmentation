import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

#integer class matchings

integer_class_matches = [[0, 0, 0],[128, 0, 0],[128, 64, 128],[192, 0, 192],[0, 128, 0],[128, 128, 0],[64, 64, 0],[64, 0, 128]]

def predict(valid_gen,model,image_size=128):
    
    ONE_CHANNEL_IMAGE_SIZE=(image_size,image_size)
    THREE_CHANNEL_IMAGE_SIZE = (image_size,image_size,3)
    #get random image with label
    path = random.choice(valid_gen.train_files)
    real_image, label = valid_gen.__load__(path)

    img=np.expand_dims(real_image, axis=0)
    label = label > 0.5
    
    pred=(model.predict(img)>0.5)[0]

    #convert one hot predictions to 1 channel integers
    new_image = np.zeros(ONE_CHANNEL_IMAGE_SIZE)
    new_label_image = np.zeros(ONE_CHANNEL_IMAGE_SIZE)
    for c in range(pred.shape[2]):
        pred_channel=pred[:,:,c]
        label_channel=label[:,:,c]

        new_image[np.where(pred_channel==1)]=c
        new_label_image[np.where(label_channel==1)]=c

    # create labeled image with colors
    predicted_image = np.zeros(THREE_CHANNEL_IMAGE_SIZE,dtype=np.uint8)
    label_image = np.zeros(THREE_CHANNEL_IMAGE_SIZE,dtype=np.uint8)

    for id,val in enumerate(integer_class_matches):
      mask = (new_image==id)
      predicted_image[mask] = val

      label_mask = (new_label_image==id)
      label_image[label_mask] = val
    
    rate=0.998
    masked_image = (real_image*rate + predicted_image*(1-rate))

    #plot all photos
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('Testing Image')
    plt.imshow(real_image)
    plt.subplot(222)
    plt.title('Testing Label')
    plt.imshow(label_image)
    plt.subplot(223)
    plt.title('Prediction on test image')
    plt.imshow(predicted_image)
    plt.subplot(224)
    plt.title('Mask real image with test label')
    plt.imshow(masked_image)
    plt.show()
