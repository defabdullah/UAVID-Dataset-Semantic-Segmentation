import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
#integer class matchings

integer_class_matches = [[0, 0, 0],[128, 0, 0],[128, 64, 128],[192, 0, 192],[0, 128, 0],[128, 128, 0],[64, 64, 0],[64, 0, 128]]

def predict(valid_gen,model,image_size=256):
    
    THREE_CHANNEL_IMAGE_SIZE = (image_size,image_size,3)

    #get random image with label
    path = random.choice(valid_gen.train_files)
    real_image, label = valid_gen.__load__(path)

    img=np.expand_dims(real_image, axis=0)
    label=np.argmax(label, axis=2)
    
    pred=(model.predict(img))[0]
    pred=np.argmax(pred, axis=2)

    # create labeled image with colors
    pred_image = np.zeros(THREE_CHANNEL_IMAGE_SIZE,dtype=np.uint8)
    label_image = np.zeros(THREE_CHANNEL_IMAGE_SIZE,dtype=np.uint8)
    for id,val in enumerate(integer_class_matches):
        pred_image[np.where(pred==id)] = val
        label_image[np.where(label==id)] = val
    plot_graphs(real_image, label_image, pred_image)
    _, ax = plt.subplots(figsize=(10, 10))
    IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
    plot_confusion_matrix(IC, pred.flatten(), label.flatten(), normalize='true', values_format='.2%',ax=ax)


def plot_graphs(real_image, label_image, pred_image, rate=0.998):

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
    plt.imshow(pred_image)

    masked_image = (real_image*rate + pred_image*(1-rate))
    plt.subplot(224)
    plt.title('Mask real image with prediction')
    plt.imshow(masked_image)
    plt.show()

def plot_all_conf(valid_gen,model,image_size=256):
    length=len(valid_gen.train_files)

    real_images = np.zeros([length,image_size,image_size,3])
    labels = np.zeros([length,image_size,image_size,8])

    for i in range(length):
        real_image, label = valid_gen.__load__(valid_gen.train_files[i])
        real_images[i]=real_image
        labels[i]=label

    preds = model.predict(real_images)

    labels=np.argmax(labels, axis=3)
    preds=np.argmax(preds, axis=3)
    

    _, ax = plt.subplots(figsize=(10, 10))
    IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
    plot_confusion_matrix(IC, preds.flatten(), labels.flatten(), normalize='true', values_format='.2%',ax=ax)
    
