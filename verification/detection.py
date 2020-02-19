import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import ImageDraw, ImageFont
from skimage import measure
from tensorflow_core.python.keras.models import model_from_json

all_contours = []
temp_framed = []


def load_922_model(directory):
    # load json and create model
    json_file = open(directory + '_model.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(directory + "_weight.h5")
    return loaded_model


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))

def prep_meta_img(filename):
    #load metaphase
    img = image.load_img(filename, color_mode='grayscale')
    img = image.img_to_array(img, dtype='uint8')
    img = img.reshape(img.shape[0], img.shape[1])
    temp_framed.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    
    #find contour - crop - rotate
    ch_img = []
    cropped = []
    temp_contour = []
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    contours = measure.find_contours(img, 200)
    for j, con in enumerate(contours):
        w = max(con[:, 1]) - min(con[:, 1])
        h = max(con[:, 0]) - min(con[:, 0])
        if h > 10 and w > 10:
            temp_contour.append(con)
            temp_crop = img[int(min(con[:, 0]) - 2):int(max(con[:, 0]) + 2),
                        int(min(con[:, 1]) - 2):int(max(con[:, 1]) + 2)]
            temp_crop = cv2.copyMakeBorder(temp_crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
            max_diff = 0
            for angle in range(0, 180, 15):  # rotate
                rotated = rotate_bound(temp_crop, angle)
                contours2 = measure.find_contours(rotated, 200)
                for con2 in contours2:
                    w = max(con2[:, 1]) - min(con2[:, 1])
                    h = max(con2[:, 0]) - min(con2[:, 0])
                    if h > 10 and w > 10 and h - w > max_diff:
                        max_diff = h - w
                        right_rotated = rotated[int(min(con2[:, 0]) - 2):int(max(con2[:, 0]) + 2),
                                        int(min(con2[:, 1]) - 2):int(max(con2[:, 1]) + 2)]
            cropped.append(right_rotated)

    #sort crop image and contour 
    index_sorted = sorted(range(len(cropped)), key=lambda k: cropped[k].shape[0])
    cropped2 = []
    for i in index_sorted:
        all_contours.append(temp_contour[i])
        cropped2.append(cropped[i])

    #resize(add border)
    fixed_h, fixed_w = 100, 72
    temp = []
    for j, img in enumerate(cropped2):
        top = int((fixed_h - img.shape[0]) / 2)
        if top * 2 + img.shape[0] != fixed_h:
            bot = int((fixed_h - img.shape[0]) / 2 + 1)
        else:
            bot = int((fixed_h - img.shape[0]) / 2)

        left = int((fixed_w - img.shape[1]) / 2)
        if left * 2 + img.shape[1] != fixed_w:
            right = int((fixed_w - img.shape[1]) / 2 + 1)
        else:
            right = int((fixed_w - img.shape[1]) / 2)
        
        if top > 0 and bot > 0 and left > 0 and right > 0:
            temp = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_CONSTANT, value=255)
            ch_img.append(cv2.cvtColor(img_to_array(temp), cv2.COLOR_GRAY2RGB))
    
    ch_img = np.asarray(ch_img)

    return ch_img


def framing(no, chromosome):
    min_h = int(min(all_contours[no][:, 0])) - 2
    max_h = int(max(all_contours[no][:, 0])) + 2
    min_w = int(min(all_contours[no][:, 1])) - 2
    max_w = int(max(all_contours[no][:, 1])) + 2
    frame = []
    for k in range(max_h - min_h + 1):
        frame.append([min_h + k, min_w])
        frame.append([min_h + k, max_w])
    for k in range(max_w - min_w + 1):
        frame.append([min_h, min_w + k])
        frame.append([max_h, min_w + k])
    temp_index = (min_w - 10, min_h - 21)
    for pixel in frame:
        try:
            if chromosome == 9:
                temp_framed[0][pixel[0] - 5, pixel[1] - 5] = [255, 0, 0]
            else:
                temp_framed[0][pixel[0] - 5, pixel[1] - 5] = [0, 0, 255]
        except Exception:
            pass

    return temp_index


def label_framing(temp_index, chro):
    font = ImageFont.truetype('Roboto-Bold.ttf', size=10)
    draw = ImageDraw.Draw(temp_framed[0])
    message = "ABCDEFGHI"
    if chro == 9:
        color = 'rgb(255, 0, 0)'
    else:
        color = 'rgb(0, 0, 255)'
    for i, index in enumerate(temp_index):
        draw.text(index, str(chro) + message[i], fill=color, font=font) 


def predict_22(ch_img, model_find, model_classify):
    chromosome = 22
    n = 5
    
    find = model_find.predict_classes(ch_img[:n])
    classify = model_classify.predict_classes(ch_img[:n])
    prob = model_classify.predict(ch_img[:n])

    temp_index2 = []
    index_n, index_p, result_img, result_prob, result_pred = [], [], [], [], []
    for j in range(len(ch_img)):
        if j < n and len(result_img) < 4:
            img = array_to_img(ch_img[j])

            if find[j] == 1:  # chr 22
                temp_index = framing(j, chromosome)
                temp_index2.append(temp_index)
                if classify[j] == 1:  # phila 22
                    print(str(chromosome) + 'PH : ' + str(j))
                    result_img.append(img)
                    result_prob.append(prob[j][1])
                    result_pred.append(True)
                    index_p.append(j + 1)
                else:  # normal 22
                    print(str(chromosome) + 'NM : ' + str(j))
                    result_img.append(img)
                    result_prob.append(prob[j][0])
                    result_pred.append(False)
                    index_n.append(j + 1)
        else:
            break

    if np.any(index_n) and not np.any(index_p):
        print('Not found abnormal chromosome %d' % chromosome)
        result = 0
    elif np.any(index_p) and np.any(index_n):
        print('Found abnormal chromosome %d' % chromosome)
        result = 1
    else:
        print('cannot predict this metaphase')
        result = None

    return result_img, result_prob, result_pred, result, temp_index2


def predict_9(ch_img, model_n, model_p):
    chromosome = 9
    n = 36

    predicted_N = model_n.predict_classes(ch_img[:n])
    prob_n = model_n.predict(ch_img[:n])
    predicted_P = model_p.predict_classes(ch_img[:n])
    prob_p = model_p.predict(ch_img[:n])

    temp_index2 = []
    index_n, index_p, result_img, result_prob, result_pred = [], [], [], [], []
    for j in range(len(ch_img)):
        if j < n and len(result_img) < 4:
            img = array_to_img(ch_img[j])

            if predicted_N[j] == 1 or predicted_P[j] == 1:
                temp_index = framing(j, chromosome)
                temp_index2.append(temp_index)
            if predicted_N[j] == 1 and predicted_P[j] == 1:
                print(str(chromosome) + 'chromosome: both model predict same result at index ' + str(j))
                result_img.append(img)
                index_p.append(j + 1)
                result_prob.append(prob_p[j][1])
                result_pred.append(True)
            elif predicted_N[j] == 1:
                print(str(chromosome) + 'NM : ' + str(j))
                result_img.append(img)
                result_prob.append(prob_n[j][1])
                result_pred.append(False)
                index_n.append(j + 1)
            elif predicted_P[j] == 1:
                print(str(chromosome) + 'PH : ' + str(j))
                result_img.append(img)
                result_prob.append(prob_p[j][1])
                result_pred.append(True)
                index_p.append(j + 1)
        else:
            break

    if np.any(index_n) and not np.any(index_p):
        print('Not found abnormal chromosome %d' % chromosome)
        result = 0
    elif np.any(index_p):
        print('Found abnormal chromosome %d' % chromosome)
        result = 1
    else:
        print('cannot predict this metaphase')
        result = None

    return result_img, result_prob, result_pred, result, temp_index2

def nine_22(meta_filename):
    model_9n = load_922_model('models/9N')
    model_9p = load_922_model('models/9P')
    model_22f = load_922_model('models/22Find')
    model_22c = load_922_model('models/22Classify')

    img_9, prob_9, pred_9, result_9 = [], [], [], []
    img_22, prob_22, pred_22, result_22 = [], [], [], []
    framed = []

    for filename in meta_filename:
        all_contours.clear()
        temp_framed.clear()
        output = {9,22}    
        
        #preprocess metaphase image
        ch_img = prep_meta_img(filename)

        #predict 9,22 and framing
        img_9t,prob_9t,pred_9t,result_9t,index9 = predict_9(ch_img, model_9n, model_9p)
        img_22t,prob_22t,pred_22t,result_22t,index22 = predict_22(ch_img, model_22f, model_22c)
        temp_framed[0] = array_to_img(temp_framed[0])
        
        #label framing
        label_framing(index9, 9)
        label_framing(index22, 22)

        #save output
        img_9.append(img_9t)  # [[img1-1,img1-2], [img2-1,img2-2,img2-3], [img3-1,img3-2]]
        prob_9.append(prob_9t)
        pred_9.append(pred_9t)
        result_9.append(result_9t)

        img_22.append(img_22t)
        prob_22.append(prob_22t)
        pred_22.append(pred_22t)
        result_22.append(result_22t)

        framed.append(temp_framed[0])  # [img1,img2,img3]

    return img_9,prob_9,pred_9,result_9,\
    img_22, prob_22, pred_22, result_22,\
    framed