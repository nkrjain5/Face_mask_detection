import cv2
from keras.models import load_model
import numpy as np
import os

model_01=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_001.h5'
model_02=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_002_50Epochs.h5'
model_03=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\face_mask_model_003_100Epochs.h5'

model_04=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_001_100Epochs.h5'
model_05=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_002_100Epochs.h5'
model_06=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_003_50Epochs.h5'
model_07=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_004_100Epochs.h5'

model_08=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_005_newDS_100Epochs.h5'
model_09=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_006_newDS_100Epochs.h5'
model_10=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_007_newDS_50Epochs.h5'
model_11=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_008_newDS_50Epochs.h5'
model_12=r'J:\Udemy\DL\face_mask_detection_pyimagesearch\Face_mask_detector_nkr\Model\Trained_mode\face_mask_model_009_newDS_50Epochs.h5'

caffe_mode_path=r'J:\Udemy\DL\Face_detection_caffee\Model\res10_300x300_ssd_iter_140000.caffemodel'
caffe_model_proto=r'J:\Udemy\DL\Face_detection_caffee\Model\deploy.prototxt.txt'

video_out=r'J:\RT_mask_temp_monitor\RT_mask_detection.avi'
def main():
	caffemodel=cv2.dnn.readNetFromCaffe(caffe_model_proto,caffe_mode_path)
	detect_face=load_model(model_12)
	
	cap=cv2.VideoCapture(0)

	fourcc = cv2.VideoWriter_fourcc(*'XVID') 
	out = cv2.VideoWriter(video_out, fourcc, 10.0, (640, 480)) 

	while(cv2.waitKey(1))!=27:
		ret,img=cap.read()
		if not ret: break
		img=cv2.flip(img,1)
		(h, w) = img.shape[:2]
		print(h,w)
		blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		predictions=caffemodel.forward(caffemodel.setInput(blob))
		for i in range(0,15):#predictions.shape[2]):  #limiting to 10 detections only
			if predictions[0,0,i,2] > 0.75:
				box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				print(i,(startX+int(startX), startY, endX, endY))
				T=startX-int(startX*0.075)
				L=startY-int(startY*0.075)
				B=endY+int(startY*0.075)
				R=endX+int(startX*0.075)
				cropped=img[L:B,T:R]
				cropped=cv2.resize(cropped,(224,224))
				cv2.imshow('cropped',cropped)
				cropped=cropped.reshape(1,224,224,3)
				cropped=cropped.astype('float32')
				cropped/=255
				# res=str(detect_face.predict_classes(resized,1,verbose=0)[0][0])
				res=detect_face.predict(cropped)[0][0]
				print(res)
				if res==str(1) or res==str(1.0) or res>0.75:
					pred="MASK ON!"
					cv2.putText(img, str(pred), (T,L-15) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), 2)
					cv2.rectangle(img,(T, L),(R,B),(0,255,0),2)

				elif res==str(0) or res==str(0.) or res < 0.75:
					pred="MASK OFF!"
					cv2.putText(img, str(pred), (T,L-15) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2)
					cv2.rectangle(img,(T, L),(R,B),(0,0,255),2)
				# cv2.putText(img, str(res), (T+30,L-15) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,0,255), 2)
		cv2.imshow('live',img)
		out.write(img)

	cv2.destroyAllWindows()
	cap.release()

if __name__=='__main__':
	main()