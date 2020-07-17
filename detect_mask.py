from keras.models import load_model
import cv2
import numpy as np
import os
classifier_model_path=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\haarcascade_frontalface_default.xml'
mask_detection_model_01=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_001.h5'
mask_detection_model_02=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_002_50Epochs.h5'
mask_detection_model_03=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_003_100Epochs.h5'

model_04=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_001_100Epochs.h5'
model_05=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_002_100Epochs.h5'
model_06=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_003_50Epochs.h5'
model_07=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_004_100Epochs.h5'

model_08=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_005_newDS_100Epochs.h5'
model_09=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_006_newDS_100Epochs.h5'
model_10=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_007_newDS_50Epochs.h5'
model_11=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_008_newDS_50Epochs.h5'
model_12=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_009_newDS_50Epochs.h5'
VID_TEST=0

def print_on_image(img,res):
	if res==str(1) or res==str(1.0) or res>0.75:
		pred="MASK ON!"
		cv2.putText(img, str(pred), (10,50) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), 2)
	elif res==str(0) or res==str(0.) or res < 0.75:
		pred="MASK OFF!"
		cv2.putText(img, str(pred), (10,50) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2)
	cv2.putText(img, str(res), (10,75) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2)
	return img

def ger_predictions(image,detect_face):
	resized=cv2.resize(image,(224,224))
	resized = resized.reshape(1,224,224,3) 
	# resized=resized.astype('float32')
	# resized/=255
	# res=str(detect_face.predict_classes(resized,1,verbose=0)[0][0])
	res=detect_face.predict(resized)[0][0]
	print(res)
	return res


def main():
	print("start")
	cap=cv2.VideoCapture(0)
	detect_face=load_model(model_05)
	cv2.namedWindow('live',cv2.WINDOW_NORMAL)

	# # face_classifier=cv2.CascadeClassifier(classifier_model_path)
	if VID_TEST==1:
		while cv2.waitKey(1)!=27:
			ret,img=cap.read()
			if not ret : 
				print("unable to fetch image ") 
				break
			img=cv2.flip(img,1)
			res=ger_predictions(img,detect_face)
			img=print_on_image(img,res)
			cv2.imshow('live',img)
		
	else:
		img=[cv2.imread(os.path.join(r'J:\Udemy\DL\face_mask_detection_pyimagesearch\face-mask-detector\examples',file)) for file in os.listdir(r'J:\Udemy\DL\face_mask_detection_pyimagesearch\face-mask-detector\examples')]
		# img=cv2.imread(r'J:\Udemy\DL\face_mask_detection_pyimagesearch\face-mask-detector\examples\example_07.png')
		for imgs in img:
			res=ger_predictions(imgs,detect_face)
			img=print_on_image(imgs,res)
			cv2.imshow('live',imgs)
			cv2.waitKey(0)

	if VID_TEST ==1:
		cap.release()
	cv2.destroyAllWindows()
	



if __name__=='__main__':
	main()