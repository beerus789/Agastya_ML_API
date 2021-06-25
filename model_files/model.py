#import modules
from PIL import Image
from keras.models import load_model, Model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#Load Model
# def load_model():
#     model = load_model('trained_Model.hdf5')
#     model.load_weights('trained_Model.hdf5')

#Main function to predict COVID
def predict_covid(img_loc):

    model = load_model('C:/DATA/Agastya-VC/Agastya_ML_API/model_files/trained_Model.hdf5')
    model.load_weights('C:/DATA/Agastya-VC/Agastya_ML_API/model_files/trained_Model.hdf5')

    label_dic = {0:'Covid19 Positive',
            1: 'Covid19 Negative',
            2: 'Pnemonia'}
    
    new_data2 = []

    # img_loc = 'https://firebasestorage.googleapis.com/v0/b/agastya-2021.appspot.com/o/1.jpeg?alt=media&token=2cc16723-8efb-4351-b57f-485c3a426a1f'
    # img_temp = url_to_image(img_loc)

    # img_loc.encode()
    img_loc = img_loc.decode('utf-8')
    print(type(img_loc))
    img_temp = io.imread(img_loc)


    # new_imag2 = cv2.imread(img_temp)
    new_imag2 = img_temp
    img = new_imag2.copy()
    new_imag2 = cv2.cvtColor(new_imag2, cv2.COLOR_BGR2RGB)
    new_imag2 = cv2.resize(new_imag2, (224,224))
    new_data2.append(new_imag2)

    new_data2 = np.array(new_data2)/255.0

    prediction = model.predict(new_data2, batch_size=8)
    result = np.argmax(prediction, axis=1)[0]
    accuracy = float(np.max(prediction, axis=1)[0])
    label = label_dic[result]
    # print(prediction, result, accuracy)
    return (label_dic[result], accuracy * 100)
    # print("{} \n{:.2f}%".format(label_dic[result], accuracy*100))


    # window_name = label_dic[result]
    # font = cv2.FONT_HERSHEY_SIMPLEX 
    # org = (80, 80) 
    # fontScale = 0.5
    # thickness = 1
    # rows = 1
    # columns = 3
    # fig = plt.figure(figsize=(10, 20))

    # img = cv2.putText(img, label_dic[result], org, font,
    #                     fontScale, (0, 0, 255), thickness, cv2.LINE_AA)
    # plt.imshow(img)
    # plt.axis('on')
    # plt.show()

# predict_covid(image_loc)