from verification.detection_van import prep_meta_img_van

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.ndimage as ndimage 
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from keras.models import Model,model_from_json


#show all img
def show_all_img(list_img):
  print(len(list_img))
  plt.figure(figsize=(12,12))
  columns = 10
  for i,ima in  enumerate(list_img):  
      plt.subplot(len(list_img) / columns + 1, columns, i + 1)
      plt.imshow(ima,cmap='gray')

def load_922_model(directory):
    # load json and create model
    json_file = open(directory + '_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(directory + "_weight.h5")
    return loaded_model


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


# Use the contour to mask the original image
def mask_image_by_contour(contour, original_image):
  image_mask = np.zeros_like(original_image, dtype='bool')
  image_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
  image_mask = ndimage.binary_fill_holes(image_mask)
  copy_image = np.copy(original_image)
  copy_image = copy_image.reshape(original_image.shape[0] * original_image.shape[1])
  image_mask = image_mask.reshape(original_image.shape[0] * original_image.shape[1])
  for i in range(copy_image.shape[0]):
    if image_mask[i] == True:
      copy_image[i] = copy_image[i]
    else:
      copy_image[i] = 255
  copy_image = copy_image.reshape(original_image.shape[0], original_image.shape[1])
  return copy_image

def prep_meta_img(filename):
    #load metaphase
    framed_img = image.load_img(filename, color_mode='rgb')
    img = image.load_img(filename, color_mode='grayscale')
    img = image.img_to_array(img, dtype='uint8')
    img = img.reshape(img.shape[0], img.shape[1])
    
    
    #find contour - crop - rotate
    ch_img = []
    cropped = []
    origin_cropped = []
    temp_temp = []
    all_contours = []

    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    blur = cv2.GaussianBlur(img,(5,5),10)
    ret,th = cv2.threshold(blur,220,255,cv2.THRESH_BINARY)
    contours = measure.find_contours(th, 200)
    for con in contours:
        w = max(con[:, 1]) - min(con[:, 1])
        h = max(con[:, 0]) - min(con[:, 0])
        if h > 7 and w > 7:
            all_contours.append(con)
            temp_crop = img[int(min(con[:, 0]) - 2):int(max(con[:, 0]) + 2),
                        int(min(con[:, 1]) - 2):int(max(con[:, 1]) + 2)]
            temp_crop = cv2.copyMakeBorder(temp_crop, 5,5,5,5, cv2.BORDER_CONSTANT, value=255)
            blur = cv2.GaussianBlur(temp_crop,(5,5),10)
            ret,th = cv2.threshold(blur,220,255,cv2.THRESH_BINARY)
            cont = measure.find_contours(th, 200)
            max_size = 0
            for con2 in cont:
                w = max(con2[:, 1]) - min(con2[:, 1])
                h = max(con2[:, 0]) - min(con2[:, 0])
                if h > 7 and w > 7 and h*w>max_size:
                  max_size = h*w
                  con3 = con2
            temp_crop = mask_image_by_contour(con3,temp_crop)
            origin_cropped.append(temp_crop[5:int(temp_crop.shape[0])-5, 5:int(temp_crop.shape[1])-5])
            right_rotated = temp_crop
            max_diff = 0
            
            for angle in range(0, 180, 15):  # rotate
                rotated = rotate_bound(temp_crop, angle)
                blur2 = cv2.GaussianBlur(rotated,(5,5),10)
                ret,th2 = cv2.threshold(blur2,230,255,cv2.THRESH_BINARY)
                contours2 = measure.find_contours(th2, 200)
                for con2 in contours2:
                    w = max(con2[:, 1]) - min(con2[:, 1])
                    h = max(con2[:, 0]) - min(con2[:, 0])
                    if h > 7 and w > 7 and h - w > max_diff:
                        max_diff = h - w
                        right_rotated = rotated[int(min(con2[:, 0]) - 2):int(max(con2[:, 0]) + 2),
                                        int(min(con2[:, 1]) - 2):int(max(con2[:, 1]) + 2)]
            cropped.append(right_rotated)
            
   
    #sort width crop image and contour 
    index_sorted = sorted(range(len(cropped)), key=lambda k: cropped[k].shape[1],reverse=True)
    for i in index_sorted:
        all_contours.append(all_contours[i])
        origin_cropped.append(origin_cropped[i])
        cropped.append(cropped[i])
    del all_contours[0:int(len(all_contours)/2)]
    del origin_cropped[0:int(len(origin_cropped)/2)]
    del cropped[0:int(len(cropped)/2)]
    
    
    temp_crop = []
    index_del = []
    num_cropped = len(origin_cropped)
    for i,ch in enumerate(origin_cropped):
        ch = cv2.copyMakeBorder(ch, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
        blur = cv2.GaussianBlur(ch,(3,3),10)
        for value in range(210,180,-10): #210 200 190
          ret,th = cv2.threshold(blur,value,255,cv2.THRESH_BINARY)
          contours = measure.find_contours(th, 200)
          temp_con = []
          for con in contours:
            w = max(con[:, 1]) - min(con[:, 1])
            h = max(con[:, 0]) - min(con[:, 0])
            if h > 7 and w > 7:
                temp_con.append(con)
          if len(temp_con) > 1 :
            num_cropped += len(temp_con)-1
            
            for con2 in temp_con:
              masked_image = mask_image_by_contour(con2,ch)
              temp_crop2 = masked_image[int(min(con2[:, 0]) -2):int(max(con2[:, 0])+2 ),
                                          int(min(con2[:, 1])-2 ):int(max(con2[:, 1])+2 )]
              temp_crop3 = cv2.copyMakeBorder(temp_crop2, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
              right_rotated = temp_crop3
              max_h = 0
              for angle in range(0, 180, 15):  # rotate
                  rotated = rotate_bound(temp_crop3, angle)
                  blur2 = cv2.GaussianBlur(rotated,(5,5),10)
                  ret,th2 = cv2.threshold(blur2,value,255,cv2.THRESH_BINARY)
                  contours2 = measure.find_contours(rotated, 200)
                  for con3 in contours2:
                      w = max(con3[:, 1]) - min(con3[:, 1])
                      h = max(con3[:, 0]) - min(con3[:, 0])
                      if h > 7 and w > 7 and h > max_h:
                        max_h = h
                        right_rotated = rotated[int(min(con3[:, 0]) - 2):int(max(con3[:, 0]) + 2),
                                            int(min(con3[:, 1]) - 2):int(max(con3[:, 1]) + 2)]
                        temp_temp.append(right_rotated)
              
              min_h = int(min(all_contours[i][:, 0]))
              min_w = int(min(all_contours[i][:, 1])) 
              temp_con2 = []
              for value2 in con2:
                  temp_con2.append([value2[0] +min_h-7, value2[1] + min_w-7])
              temp_crop.append([right_rotated,temp_con2,temp_crop2])
            index_del.append(i)
            break

        if num_cropped >=46 or i>15:
          for index in reversed(index_del):
              cropped.pop(index)
              all_contours.pop(index)
              origin_cropped.pop(index)
          for crop in temp_crop:
              cropped.append(crop[0])
              all_contours.append(np.array(crop[1]))
              origin_cropped.append(crop[2])
          break
   
    temp_temp = []
    #sort width crop image and contour 2
    if len(cropped) < 46:
      index_sorted = sorted(range(len(cropped)), key=lambda k: cropped[k].shape[1],reverse=True)
      for i in index_sorted:
        all_contours.append(all_contours[i])
        origin_cropped.append(origin_cropped[i])
        cropped.append(cropped[i])
      del all_contours[0:int(len(all_contours)/2)]
      del origin_cropped[0:int(len(origin_cropped)/2)]
      del cropped[0:int(len(cropped)/2)]

      temp_crop = []
      index_del = []
      num_cropped = len(origin_cropped)
      for i,ch in enumerate(origin_cropped):
          ch = cv2.copyMakeBorder(ch, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
          #blur = cv2.GaussianBlur(ch,(1,1),10)
          for value in range(190,160,-10): #190 180 170
            ret,th = cv2.threshold(ch,value,255,cv2.THRESH_BINARY)
            contours = measure.find_contours(th, 200)
            temp_con = []
            for con in contours:
              w = max(con[:, 1]) - min(con[:, 1])
              h = max(con[:, 0]) - min(con[:, 0])
              if h > 7 and w > 7:
                  temp_con.append(con)
            if len(temp_con) > 1 :
              num_cropped += len(temp_con)-1

              for con2 in temp_con:
                masked_image = mask_image_by_contour(con2,ch)
                temp_crop2 = masked_image[int(min(con2[:, 0]) - 2):int(max(con2[:, 0]) + 2),
                                            int(min(con2[:, 1]) - 2):int(max(con2[:, 1]) + 2)]
                temp_crop3 = cv2.copyMakeBorder(temp_crop2, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
                right_rotated = temp_crop3
                max_h = 0
                for angle in range(0, 180, 15):  # rotate
                    rotated = rotate_bound(temp_crop3, angle)
                    blur2 = cv2.GaussianBlur(rotated,(1,1),10)
                    ret,th2 = cv2.threshold(rotated,value,255,cv2.THRESH_BINARY)
                    contours2 = measure.find_contours(th2, 200)
                    for con3 in contours2:
                        w = max(con3[:, 1]) - min(con3[:, 1])
                        h = max(con3[:, 0]) - min(con3[:, 0])
                        if h > 7 and w > 7 and h > max_h:
                          max_h = h
                          right_rotated = rotated[int(min(con3[:, 0]) - 2):int(max(con3[:, 0]) + 2),
                                              int(min(con3[:, 1]) - 2):int(max(con3[:, 1]) + 2)]
                min_h = int(min(all_contours[i][:, 0]))
                min_w = int(min(all_contours[i][:, 1])) 
                temp_con2 = []
                for value2 in con2:
                  temp_con2.append([value2[0] +min_h-7, value2[1] + min_w-7])
                temp_crop.append([right_rotated,temp_con2]) 
              index_del.append(i)
              break

          if num_cropped >=46 or i>5:
            for index in reversed(index_del):
              cropped.pop(index)
              all_contours.pop(index)
            for crop in temp_crop:
              cropped.append(crop[0])
              all_contours.append(np.array(crop[1]))
            break

    #sort high crop image and contour 
    index_sorted = sorted(range(len(cropped)), key=lambda k: cropped[k].shape[0])
    for i in index_sorted:
        all_contours.append(all_contours[i])
        cropped.append(cropped[i])
    del all_contours[0:int(len(all_contours)/2)]
    del cropped[0:int(len(cropped)/2)]
        
    ch_img = resize_chr(cropped,len(cropped))
    ch_img_s = resize_chr(cropped,9)

    return ch_img, ch_img_s, framed_img, all_contours

# Resize cropped chromosomes to same size
def resize_chr(chr, amount):
  fixed_h, fixed_w = 100, 72
  ch_img = []
  for img in chr[:amount]:

      if (amount!= len(chr)):
          img = cv2.resize(img, dsize=(16,int(16*(img.shape[0]/img.shape[1]))), interpolation=cv2.INTER_CUBIC)
          
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

def framing(no, chro, no_result, framed_img, all_contours):
    text = 'ABCDEF'
    font = ImageFont.truetype('Roboto-Bold.ttf', size=10)
    min_h = int(min(all_contours[no][:, 0])) - 8 # Top border
    max_h = int(max(all_contours[no][:, 0])) - 2 # Bottom border
    min_w = int(min(all_contours[no][:, 1])) - 8 # Left border
    max_w = int(max(all_contours[no][:, 1])) - 2 # Right border
    shape = [min_w,min_h,max_w,max_h]
    if chro == 9:
      color = 'red'
    elif chro == 22:
      color = 'blue'    
    
    draw = ImageDraw.Draw(framed_img)
    draw.rectangle(shape, outline =color) 
    draw.text((min_w , min_h -13),str(chro)+text[no_result], fill=color, font=font)
    
    return framed_img


def predict_22(ch_img, ch_img_s, model_find, model_classify, framed_img, all_contours):
    chromosome = 22
    n = len(ch_img_s)
    
    find = model_find.predict_classes(ch_img_s[:n])
    classify = model_classify.predict_classes(ch_img[:n])
    prob = model_classify.predict(ch_img[:n])

    index_n, index_p, result_img, result_prob, result_pred = [], [], [], [], []
    for j in range(len(ch_img)):
        if j < n and len(result_img) < 4:
            img = array_to_img(ch_img[j])

            if find[j] == 1:  # chr 22
                framed_img = framing(j, chromosome, len(result_img), framed_img, all_contours)
                if classify[j] == 1:  # phila 22
                    result_img.append(img)
                    result_prob.append(prob[j][1])
                    result_pred.append(True)
                    index_p.append(j + 1)
                else:  # normal 22
                    result_img.append(img)
                    result_prob.append(prob[j][0])
                    result_pred.append(False)
                    index_n.append(j + 1)
        else:
            break

    if np.any(index_n) and not np.any(index_p):
        #print('Not found abnormal chromosome %d' % chromosome)
        result = 0
    elif np.any(index_p) and np.any(index_n):
        #print('Found abnormal chromosome %d' % chromosome)
        result = 1
    else:
        #print('cannot predict this metaphase')
        result = None

    return result_img, result_prob, result_pred, result, framed_img


def predict_9(ch_img, model_n, model_p, framed_img, all_contours):
    chromosome = 9
    n = 36

    predicted_N = model_n.predict_classes(ch_img[:n])
    prob_n = model_n.predict(ch_img[:n])
    predicted_P = model_p.predict_classes(ch_img[:n])
    prob_p = model_p.predict(ch_img[:n])
    index_n, index_p, result_img, result_prob, result_pred = [], [], [], [], []
    for j in range(len(ch_img)):
        if j < n and len(result_img) < 4:
            img = array_to_img(ch_img[j])

            if predicted_N[j] == 1 or predicted_P[j] == 1:
                framed_img = framing(j, chromosome,len(result_img),framed_img,all_contours)
            if predicted_N[j] == 1 and predicted_P[j] == 1:
                result_img.append(img)
                index_p.append(j + 1)
                result_prob.append(prob_p[j][1])
                result_pred.append(True)
            elif predicted_N[j] == 1:
                result_img.append(img)
                result_prob.append(prob_n[j][1])
                result_pred.append(False)
                index_n.append(j + 1)
            elif predicted_P[j] == 1:
                result_img.append(img)
                result_prob.append(prob_p[j][1])
                result_pred.append(True)
                index_p.append(j + 1)
        else:
            break

    if np.any(index_n) and not np.any(index_p):
        result = 0
    elif np.any(index_p):
        result = 1
    else:
        result = None

    return result_img, result_prob, result_pred, result, framed_img

def nine_22(meta_filename):
    model_9n = load_922_model('models/9N')
    model_9p = load_922_model('models/9P')
    model_22f = load_922_model('models/find22_2104')
    model_22c = load_922_model('models/classify22_0804')

    prediction = {}

    for i, filename in enumerate(meta_filename, 1):
        pred = {}

        # preprocess metaphase image
        ch_img, ch_img_s, framed_img, all_contours = prep_meta_img(filename)
        # ch_img, ch_img_s, framed_img, all_contours = prep_meta_img_van(filename)

        # predict 9,22 and framing
        img_9t, prob_9t, pred_9t, result_9t, framed_img = predict_9(ch_img, model_9n, model_9p, framed_img, all_contours)
        img_22t, prob_22t, pred_22t, result_22t, framed_img = predict_22(ch_img, ch_img_s, model_22f, model_22c, framed_img, all_contours)

        # interpret image result
        if result_9t is not None and result_22t is not None:
            if result_9t + result_22t == 0:
                # print(">>> Negative")
                pred['result'] = 0
            else:
                # print(">>> Positive")
                pred['result'] = 1
        elif result_9t is not None:
            if result_9t == 0:
                # print(">>> Negative")
                pred['result'] = 0
            else:
                # print(">>> Positive")
                pred['result'] = 1
        elif result_22t is not None:
            if result_22t == 0:
                # print(">>> Negative")
                pred['result'] = 0
            else:
                # print(">>> Positive")
                pred['result'] = 1
        else:
            # print(">>> Cannot detect")
            pred['result'] = None

        # save output
        pred['img_9'] = img_9t
        pred['prob_9'] = prob_9t
        pred['pred_9'] = pred_9t

        pred['img_22'] = img_22t
        pred['prob_22'] = prob_22t
        pred['pred_22'] = pred_22t

        pred['framed'] = framed_img

        # return all cropped chromosome images
        # pred['ch'] = ch_img

        prediction[i] = pred
        print(i, "/", len(meta_filename))
    
    # sort result prediction (ph img first)
    prediction_sorted = {}
    index_sorted = sorted(prediction, key=lambda k: prediction[k]['result'], reverse=True)
    for i,index in enumerate(index_sorted):
      prediction_sorted[i+1] = prediction[index]

    return prediction_sorted