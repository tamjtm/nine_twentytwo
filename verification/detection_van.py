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

from skimage import exposure

def fill_bins_hist(hist_arr,bins_arr,bin_max=255):
  last_val = bins_arr[-1]
  first_val = bins_arr[0]
  front_fill = np.array([0]*first_val)
  last_fill = np.array([0]*(bin_max-last_val))
  filled_hist = np.hstack((front_fill,hist_arr))
  filled_hist = np.hstack((filled_hist,last_fill))
  return filled_hist.astype(int)

def get_pdf(img,bin_max=255):
  hist_img, bins_img = exposure.histogram(img)
  pdf_img = fill_bins_hist(hist_img,bins_img,bin_max)
  return pdf_img

def apply_weighted_pdf(pdf,weight_pdf=0.5):
  pdf_max = np.max(pdf)
  pdf_min = np.min(pdf)
  weighted_pdf = pdf_max * np.power((pdf - pdf_min)/(pdf_max-pdf_min),weight_pdf)
  return weighted_pdf

def get_cdf(pdf):
  cdf = pdf.cumsum()
  cdf = cdf / float(cdf[-1])
  return cdf

def apply_gamma_correction(img,weight_pdf=0.5):
  pdf_img = get_pdf(img)
  weighted_pdf = apply_weighted_pdf(pdf_img,weight_pdf)
  weighted_cdf = get_cdf(weighted_pdf)
  gamma_correction = 255 * np.power(img.ravel() / 255, 1 - weighted_cdf[img.ravel()])
  gamma_correction = gamma_correction.reshape(img.shape)
  return gamma_correction

###

import scipy.ndimage as ndimage 
from skimage.util import invert

def get_one_ellipse_ratio(region):
  if region.minor_axis_length == 0:
    return 0
  else:
    return (region.major_axis_length / region.minor_axis_length) 

def crop_off_border(img,border_len):
  x = img.shape[0]
  y = img.shape[1]
  return img[border_len:x-border_len,border_len:y-border_len]
def overwrite_region_in_original_thresholded(original_thresholded,cropped_thresholded,region):
  minr, minc, maxr, maxc = region.bbox
  x_original,y_original = np.ogrid[minr:maxr,minc:maxc]
  x_cropped,y_cropped = np.ogrid[0:cropped_thresholded.shape[0],0:cropped_thresholded.shape[1]]
  original_thresholded[x_original,y_original] = (cropped_thresholded[x_cropped,y_cropped] != original_thresholded[x_original,y_original]).astype(int)
  return original_thresholded
def extract_cropped_masked_gray_from_region(original_image,bbox,thresh_image,background_value=255):
  cropped_original = original_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
  cropped_original = cropped_original.ravel()
  blank_canvas = np.ones_like(thresh_image) * background_value
  blank_canvas = blank_canvas.ravel()
  blank_canvas[np.where(thresh_image.ravel())] = cropped_original[np.where(thresh_image.ravel())]
  return blank_canvas.reshape(thresh_image.shape)

###

from skimage.filters import threshold_isodata, threshold_mean, threshold_triangle
from skimage.morphology import binary_closing,binary_opening,thin
from skimage.morphology import square,disk
from skimage.util import invert


#normal invert work well for binary 
#but for grayscale sometime it turn value to negative area
def invert_grayscale(img,maxVal=255):
  return (maxVal - img)

def extract_foreground_from_noisy_background(img, safety_margin=1.15, closing_kernel_size=3, noise_area=44, noise_minor_axis_len=5):
  thresh_point = threshold_isodata(img)
  isodata_thresholded = img > (thresh_point * safety_margin)
  closed_isodata_thresholded = binary_closing(invert(isodata_thresholded), square(closing_kernel_size))
  labeled_image = measure.label(closed_isodata_thresholded)
  region_props = measure.regionprops(labeled_image)
  for region in region_props:
    if (region.area < noise_area or region.minor_axis_length < noise_minor_axis_len):
      labeled_image[np.where(labeled_image == region.label)] = 0
  labeled_image[np.where(labeled_image > 0)] = 1
  masked_img = invert_grayscale(img) * labeled_image
  masked_img = invert_grayscale(masked_img)
  return masked_img,thresh_point

def rough_threshold(img):
  thresh_point = threshold_mean(img)
  threshold_image = img < thresh_point
  return threshold_image,thresh_point

def de_border_image(img,border_constant):
  return img[border_constant:img.shape[0]-border_constant,border_constant:img.shape[1]-border_constant]

###

def get_complement_image(more_image,less_image):
  return more_image.astype(int) - less_image.astype(int)

def opening_with_border(img,selem,border_size,border_constant):
  int_img = img.astype(int)
  framed_image = cv2.copyMakeBorder(int_img,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=border_constant)
  opened_image = binary_opening(framed_image,selem)
  no_border_image = de_border_image(opened_image,border_size)
  return no_border_image

def thin_with_border(img,max_iter,border_size,border_constant):
  int_img = img.astype(int)
  framed_image = cv2.copyMakeBorder(int_img,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=border_constant)
  opened_image = thin(framed_image,max_iter)
  no_border_image = de_border_image(opened_image,border_size)
  return no_border_image

def dilate_with_border(img,selem,border_size,border_constant):
  int_img = img.astype(int)
  framed_image = cv2.copyMakeBorder(int_img,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=border_constant)
  dilated_image = binary_dilation(framed_image,selem)
  no_border_image = de_border_image(dilated_image,border_size)
  return no_border_image

def closing_with_border(img,selem,border_size,border_constant):
  int_img = img.astype(int)
  framed_image = cv2.copyMakeBorder(int_img,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=border_constant)
  closed_image = binary_closing(framed_image,selem)
  no_border_image = de_border_image(closed_image,border_size)
  return no_border_image

###

def one_region_opening_with_rollback(binary_img,selem,noise_size=44):
  opened_img = opening_with_border(binary_img,selem,5,0)
  labeled_img = measure.label(opened_img,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_img)
  valid_region_count = 0
  for region in region_props:
    if region.area < noise_size:
      continue
    else:
      valid_region_count = valid_region_count + 1
  if valid_region_count > 1:
    return opened_img
  else:
    return binary_img
def opening_with_rollback(binary_img,selem,noise_size=44):
  labeled_img = measure.label(binary_img,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_img)
  final_img = np.zeros_like(binary_img)
  for region in region_props:
    opened_img = one_region_opening_with_rollback(region.image,selem,noise_size)
    final_img = overwrite_region_in_original_thresholded(final_img,opened_img,region)
  return final_img
def closing_each_region(binary_img,close_size=3):
  labeled_img = measure.label(binary_img,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_img)
  final_img = np.zeros_like(binary_img)
  for region in region_props:
    closed_img = closing_with_border(region.image,close_size,5,0)
    final_img = overwrite_region_in_original_thresholded(final_img,closed_img,region)
  return final_img
def should_break_this_region(braek_image,noise_size=35):
  labeled_img = measure.label(braek_image,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_img)
  valid_region_count = 0
  for region in region_props:
    if region.area < noise_size:
      continue
    else:
      valid_region_count = valid_region_count + 1
  if valid_region_count > 1:
    return True
  else:
    return False

###

def get_first_region(binary_image):
  labeled_image = measure.label(binary_image,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_image)
  return region_props[0]

###

def get_complement_of_thresholded(input_gray_image,moderate_threshold_point,severe_threshold_point):
  moderate_thresholded = input_gray_image < moderate_threshold_point
  severe_thresholded = input_gray_image < severe_threshold_point
  complement_image = get_complement_image(moderate_thresholded,severe_thresholded)
  return complement_image
def trim_connected_region_by_area(thresholded_image,area_size=4,connectivity=1):
  labeled_image = measure.label(thresholded_image,background=0,connectivity=connectivity)
  if (len(np.unique(labeled_image)) > 1):
    region_props = measure.regionprops_table(labeled_image,properties=('label','area'))
    for i in range(len(region_props['label'])):
      if region_props['area'][i] < area_size:
        labeled_image[np.where(labeled_image == region_props['label'][i])] = 0
    labeled_image[np.where(labeled_image > 0)] = 1
  return labeled_image
def add_binary_image(binary_1,binary_2):
  sum_image = binary_1.astype(int) + binary_2.astype(int)
  sum_image[np.where(sum_image > 1)] = 1
  return sum_image

###

from scipy.signal import find_peaks

def get_mask_array_of_intensity(mask_arr,big_image):
  return big_image[np.where(mask_arr == 1)]
def get_mask_filter_peaks(peaks,bins_edge,low_bound=191,high_bound=200):
  return np.fromiter(map(lambda peak: np.logical_and(bins_edge[peak] > low_bound, bins_edge[peak] < high_bound),peaks),bool)
def plot_histogram_and_peaks(hist_list,bins_edge,peaks):
  plt.plot(bins_edge,hist_list)
  plt.plot(bins_edge[peaks],hist_list[peaks],"x")
def normalized_arr(number_arr,low=0,high=1):
  return np.interp(number_arr,(np.min(number_arr),np.max(number_arr)),(low,high))
def get_maximize_high_right_index(peak_location,height_list,peak_low_high=(0,0.9),height_low_high=(0.1,1)):
  normalized_peak_location = normalized_arr(peak_location,low=peak_low_high[0],high=peak_low_high[1])
  normalized_height_list = normalized_arr(height_list,low=height_low_high[0],high=height_low_high[1])
  cost_arr = normalized_peak_location * normalized_height_list
  return np.argmax(cost_arr)

def find_pivot_index(size_of_arr):
  half_size = size_of_arr / 2
  if half_size.is_integer():
    return int(half_size - 1)
  else:
    return np.floor(half_size).astype(int)
def get_middle_distance(location_of_peaks,middle_index):
  left_end_index = np.floor(middle_index).astype(int)
  right_start_index = np.ceil(middle_index).astype(int)
  if left_end_index == right_start_index:
    return location_of_peaks[left_end_index]
  else:
    left_location = location_of_peaks[left_end_index]
    right_location = location_of_peaks[right_start_index]
    return np.interp(middle_index,(left_end_index,right_start_index),(left_location,right_location))
def moment_of_peak_from_pivot(location_of_peaks,magnitude_of_peaks,reference_distance,offset_dist=0):
  return np.sum( (np.abs(location_of_peaks - reference_distance) + offset_dist) * magnitude_of_peaks)

def get_threshold_point_intensity_moment(masked_gray_image,thresholded_mask):
  masked_1D_array = get_mask_array_of_intensity(thresholded_mask,masked_gray_image)
  masked_1D_array = masked_1D_array[np.where(masked_1D_array > 180)]
  if len(masked_1D_array) == 0:
    return 180
  my_hist, bins_edge = exposure.histogram(masked_1D_array)
  peaks, peak_prop = find_peaks(my_hist, height=0)
  if len(peaks) > 2:
    height_list = peak_prop['peak_heights']
    
    location_of_peaks = bins_edge[peaks]
    index_middle_peak = find_pivot_index(len(location_of_peaks))
    left_end_index = index_middle_peak
    right_start_index = index_middle_peak + 1
    middle_distance = get_middle_distance(location_of_peaks,index_middle_peak)
    left_moment = moment_of_peak_from_pivot(location_of_peaks[0:left_end_index+1],height_list[0:left_end_index+1],middle_distance,offset_dist=3)
    right_moment = moment_of_peak_from_pivot(location_of_peaks[right_start_index:],height_list[right_start_index:],middle_distance,offset_dist=3)
    ratio_moment = left_moment / right_moment
    num_peaks = len(location_of_peaks)
    if ratio_moment < 1:
      target_peak_index = num_peaks - 1
    elif ratio_moment < 1.1:
      target_peak_index = np.ceil(10/12 * num_peaks - 1).astype(int)
    elif ratio_moment < 1.3:
      target_peak_index = np.ceil(9.4/12 * num_peaks - 1).astype(int)
    elif ratio_moment < 2: 
      target_peak_index = np.ceil(8/12 * num_peaks - 1).astype(int)
    elif ratio_moment < 3: 
      target_peak_index = np.ceil(6.3/12 * num_peaks - 1).astype(int)
    elif ratio_moment < 4: 
      target_peak_index = np.ceil(6.2/12 * num_peaks - 1).astype(int)
    else: 
      target_peak_index = np.ceil(5/12 * num_peaks - 1).astype(int)
    return bins_edge[peaks[target_peak_index]]
  else:
    return np.mean(masked_1D_array)

###

from scipy.ndimage.morphology import binary_fill_holes
def fill_hole_each_region(binary_img):
  labeled_img = measure.label(binary_img,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_img)
  final_img = np.zeros_like(binary_img)
  for region in region_props:
    filled_img = binary_fill_holes(region.image)
    final_img = overwrite_region_in_original_thresholded(final_img,filled_img,region)
  return final_img
def fill_non_bg_hole(binary_image,gray_image,bg_threshold=255):
  filled_hole_image = fill_hole_each_region(binary_image)
  hole_locations = get_complement_image(filled_hole_image,binary_image)
  light_area = gray_image < bg_threshold
  light_hole_locations = np.logical_and(hole_locations,light_area)
  filled_light_hole_image = add_binary_image(light_hole_locations,binary_image)
  return filled_light_hole_image

###

def get_high_outlier(number_list):
  q1 = np.quantile(number_list,0.25)
  q3 = np.quantile(number_list,0.75)
  iqr = q3 - q1
  high_outlier = q3 + 1.5 * iqr
  return high_outlier
def get_low_outlier(number_list):
  q1 = np.quantile(number_list,0.25)
  q3 = np.quantile(number_list,0.75)
  iqr = q3 - q1
  low_outlier = q1 - 1.5 * iqr
  return low_outlier
def get_properties_of_region_props(region_props):
  solidity_list = []
  ellipse_list = []
  major_length_list = []
  minor_length_list = []
  area_list = []
  for region in region_props:
    solidity_list.append(region.solidity)
    ellipse_list.append(get_one_ellipse_ratio(region))
    major_length_list.append(region.major_axis_length)
    minor_length_list.append(region.minor_axis_length)
    area_list.append(region.area)
  solidity_list = np.array(solidity_list)
  ellipse_list = np.array(ellipse_list)
  major_length_list = np.array(major_length_list)
  minor_length_list = np.array(minor_length_list)
  area_list = np.array(area_list)
  return solidity_list,ellipse_list,major_length_list,minor_length_list,area_list

def compare_ellipse_and_axis(region,major_length_thresh,minor_length_thresh):
  if region.major_axis_length < major_length_thresh:
      if region.minor_axis_length < minor_length_thresh:
          return True
  return False

def classify_region(region_props,solidity_list,ellipse_list,major_length_list,minor_length_list,area_list):

  ellipse_q1 = np.quantile(ellipse_list,0.25)
  ellipse_q2 = np.quantile(ellipse_list,0.5)
  ellipse_q3 = np.quantile(ellipse_list,0.75)
  ellipse_percent_70 = np.quantile(ellipse_list,0.70)
  ellipse_percent_75 = np.quantile(ellipse_list,0.75)
  ellipse_percent_95 = np.quantile(ellipse_list,0.95)

  major_length_q1 = np.quantile(major_length_list,0.25)
  major_length_q2 = np.quantile(major_length_list,0.5)
  major_length_q3 = np.quantile(major_length_list,0.75)
  major_length_percent_70 = np.quantile(major_length_list,0.70)

  minor_length_q1 = np.quantile(minor_length_list,0.25)
  minor_length_q2 = np.quantile(minor_length_list,0.5)
  minor_length_q3 = np.quantile(minor_length_list,0.75)
  minor_length_percent_70 = np.quantile(minor_length_list,0.70)

  solidity_low_outlier = get_low_outlier(solidity_list)
  minor_high_bound = get_high_outlier(minor_length_list)
  triangle_solidity = threshold_triangle(solidity_list,nbins=(10))

  good_solidity_region = []
  small_chromosome_region = []
  solidity_outlier_region = []
  minor_outlier_region = []
  medium_chromosome_region = []
  irregular_ellipse_region = []
  bad_else_region = []
  for region in region_props:
    if region.area > 44:
      if compare_ellipse_and_axis(region,major_length_q2,minor_length_q2):
        small_chromosome_region.append(region)
      elif region.solidity < solidity_low_outlier:
        solidity_outlier_region.append(region)
      elif region.minor_axis_length > minor_high_bound:
        minor_outlier_region.append(region)
      elif region.minor_axis_length < minor_length_percent_70:
        medium_chromosome_region.append(region)
      elif get_one_ellipse_ratio(region) < ellipse_percent_75 or get_one_ellipse_ratio(region) > ellipse_percent_95:
        irregular_ellipse_region.append(region)
      elif region.solidity >= triangle_solidity:
        good_solidity_region.append(region)
      else:
        bad_else_region.append(region)
  good_region = good_solidity_region + small_chromosome_region + medium_chromosome_region
  bad_region = []
  bad_outlier_region = solidity_outlier_region + minor_outlier_region + irregular_ellipse_region +bad_else_region
  return good_region, bad_outlier_region

###

from skimage.morphology import medial_axis

def get_contour_image(thresholded_image):
  thinned_image = thin(thresholded_image,max_iter=1)
  contour_image = get_complement_image(thresholded_image,thinned_image)
  return contour_image
def remove_narrow_part(thresholded_image,narrow_constant=2):
  skel, distance = medial_axis(thresholded_image, return_distance=True)
  dist_on_skel = distance * skel

  contour_image = get_contour_image(thresholded_image)
  distance_no_border = get_complement_image(skel,contour_image)
  distance_no_border[np.where(distance_no_border < 0)] = 0
  distance_no_border[np.where(distance > narrow_constant)] = 0

  distance_labeled_image = measure.label(distance_no_border,connectivity=2)
  if len(np.unique(distance_labeled_image)) > 1:
    distance_region_props = measure.regionprops_table(distance_labeled_image,properties=('label','area','image'))
    result_image = np.zeros_like(distance_labeled_image)
    for i in range(len(distance_region_props['label'])):
      if distance_region_props['area'][i] < 2:
        distance_labeled_image[np.where(distance_labeled_image == distance_region_props['label'][i])] = 0
    result_image[np.where(distance_labeled_image > 0)] = 1
    dilated_image = binary_dilation(result_image,square(narrow_constant+1))

    removed_narrow_part_image = get_complement_image(thresholded_image,dilated_image)
    removed_narrow_part_image[np.where(removed_narrow_part_image < 0)] = 0
    return removed_narrow_part_image
  else:
    return thresholded_image

###

from skimage.morphology import binary_dilation
def fine_threshold_operation(big_gray_image,region_props,good_region,bad_outlier_region):
  bad_image = []
  bad_modified_image = []
  final_thresholded = np.zeros_like(big_gray_image)
  for region in good_region:
    masked_image = extract_cropped_masked_gray_from_region(big_gray_image,region.bbox,region.image)
    final_thresholded = overwrite_region_in_original_thresholded(final_thresholded, region.image,region)

  for region in bad_outlier_region:
    masked_image = extract_cropped_masked_gray_from_region(big_gray_image,region.bbox,region.image)
    bad_image.append(masked_image)

    moment_threshold_point = get_threshold_point_intensity_moment(masked_image,region.image)
    moment_image = masked_image <= moment_threshold_point

    filled_image = fill_non_bg_hole(moment_image,masked_image)

    thin_image = thin(filled_image,max_iter=1)
    removed_narrow_part_image = remove_narrow_part(thin_image)
    square_opened_image = opening_with_rollback(removed_narrow_part_image,square(3))
    circle_opened_image = opening_with_rollback(square_opened_image,disk(2))
    bg_image = (~circle_opened_image.astype(bool)).astype(int)
    thin_bg = thin_with_border(bg_image,1,5,1)
    foreground_thresholded = invert(thin_bg)
    if should_break_this_region(foreground_thresholded):
      selected_mask = foreground_thresholded
    else:
      selected_mask = region.image
    bad_modified_image.append(selected_mask)
    final_thresholded = overwrite_region_in_original_thresholded(final_thresholded, selected_mask,region)
  return final_thresholded,bad_outlier_region,bad_image,bad_modified_image

###

def rotate_to_upright_by_region(image_to_rotate,region):
  orient_degree = np.degrees(region.orientation)
  upright = rotate_bound(image_to_rotate.astype(np.uint8),orient_degree)
  return upright
def get_approx_two_half(input_number):
  one_half = np.floor(input_number / 2).astype(int)
  other_half = input_number - one_half
  return [one_half,other_half]

def get_offset_border(current_border_size,target_border_size):
  if current_border_size < target_border_size:
    border_offset = get_approx_two_half(target_border_size - current_border_size)
  else:
    border_offset = [0,0]
  return border_offset
def resize_put_center(input_image,v_size_target=100,h_size_target=75):
  int_img = input_image.astype(np.uint8)
  vertical_offset = get_offset_border(int_img.shape[0],v_size_target)
  horizontal_offset = get_offset_border(int_img.shape[1],h_size_target)
  framed_image = cv2.copyMakeBorder(int_img,vertical_offset[0],vertical_offset[1],horizontal_offset[0],horizontal_offset[1],cv2.BORDER_CONSTANT,value=255)
  return framed_image
# Resize cropped chromosomes to same size
def resize_chr(chr, amount):
  fixed_h, fixed_w = 100, 72
  ch_img = []
  for j, img in enumerate(chr[:amount]):

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

def post_processing(thresholded_image,big_gray_image):
  all_contours = []
  final_labeled = measure.label(thresholded_image,background=0,connectivity=1)
  final_props = measure.regionprops(final_labeled)
  final_props = sorted(final_props, key=lambda x: x.area)
  final_collection = []
  for region in final_props:
    if region.area > 44:
      masked_image = extract_cropped_masked_gray_from_region(big_gray_image,region.bbox,region.image)
      upright = rotate_to_upright_by_region(masked_image,region)
      final_collection.append(upright)
      all_contours.append(region.coords)

  ch_img = resize_chr(final_collection,len(final_collection))
  ch_img_s = resize_chr(final_collection,9)
  return ch_img,ch_img_s,all_contours

###

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

###

def prep_meta_img_van(filename):
  framed_img = image.load_img(filename, color_mode='rgb')
  img = image.load_img(filename, color_mode='grayscale')
  img = image.img_to_array(img, dtype='uint8')
  img = img.reshape(img.shape[0], img.shape[1])
  img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

  removed_noise_image, noise_thresh_point = extract_foreground_from_noisy_background(img)
  image_rough_thresholded, rough_thresh_val = rough_threshold(removed_noise_image)
  noise_removed_open = binary_opening(image_rough_thresholded,selem=square(2))
  noise_removed_open = binary_opening(noise_removed_open,selem=disk(1))
  square_opened_img = opening_with_rollback(noise_removed_open,square(3))
  disk_opened_img = opening_with_rollback(square_opened_img,disk(3))
  labeled_image = measure.label(disk_opened_img,background=0,connectivity=1)
  region_props = measure.regionprops(labeled_image)

  solidity_list,ellipse_list,major_length_list,minor_length_list,area_list = get_properties_of_region_props(region_props)
  good_region, bad_outlier_region = classify_region(region_props,solidity_list,ellipse_list,major_length_list,minor_length_list,area_list)
  fine_thresholded,bad_outlier_region,bad_image,bad_modified_image = fine_threshold_operation(removed_noise_image,region_props,good_region,bad_outlier_region)
    
  all_contours = []
  ch_img,ch_img_s,all_contours = post_processing(fine_thresholded,removed_noise_image)
  return ch_img,ch_img_s,framed_img,all_contours
