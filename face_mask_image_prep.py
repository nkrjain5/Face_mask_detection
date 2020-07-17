# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
import numpy as np
import os
import cv2

train_data=[]
train_label=[]
test_data=[]
test_label=[]
# train_data_dir_pos=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Train\\with_mask\\'
# train_data_dir_neg=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Train\\without_mask\\'
# test_data_dir_pos=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Test\\with_mask\\'
# test_data_dir_neg=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Test\\without_mask\\'

train_data_dir_pos=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\train\\with_mask\\'
train_data_dir_neg=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\train\\without_mask\\'
test_data_dir_pos=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Train\\with_mask\\'
test_data_dir_neg=r'J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset\\Train\\without_mask\\'

dirs=[train_data_dir_pos,train_data_dir_neg,test_data_dir_pos,test_data_dir_neg]

def rename_dir(dirs=[]):
	for dirr in dirs:
		# dirr.split('\\')[-3]
		for num,files in enumerate(os.listdir(dirr)):
			os.rename(os.path.join(dirr,files),os.path.join(dirr,str(dirr.split('\\')[-3]+'_'+str(num).zfill(3)+'.png')))
def main():
	# rename_dir(dirs)
	# return
	for i,files in enumerate(os.listdir(train_data_dir_pos)):
		img=cv2.imread(os.path.join(train_data_dir_pos,files))
		resized=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
		train_data.append(resized)
		train_label.append(1)


	for files in os.listdir(train_data_dir_neg):
		img=cv2.imread(os.path.join(train_data_dir_neg,files))
		resized=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
		train_data.append(resized)
		train_label.append(0)

	for files in os.listdir(test_data_dir_pos):
		img=cv2.imread(os.path.join(test_data_dir_pos,files))
		resized=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
		test_data.append(resized)
		test_label.append(1)


	for files in os.listdir(test_data_dir_neg):
		img=cv2.imread(os.path.join(test_data_dir_neg,files))
		resized=cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
		test_data.append(resized)
		test_label.append(0)

	np.savez('J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\face_mask_train_img_new.npz', np.array(train_data))
	np.savez('J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\face_mask_train_lbl_new.npz', np.array(train_label))
	np.savez('J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\face_mask_test_img_new.npz', np.array(test_data))
	np.savez('J:\\Udemy\\DL\\face_mask_detection_pyimagesearch\\Face_mask_detector_nkr\\dataset_new\\face_mask_test_lbl_new.npz', np.array(test_label))


		# cv2.imshow(files,resized)
		# cv2.waitKey(1)
		# cv2.destroyAllWindows()


if __name__=='__main__':
	main()