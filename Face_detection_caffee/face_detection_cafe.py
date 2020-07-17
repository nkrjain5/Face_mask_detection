import cv2
import numpy as np
caffe_mode_path=r'J:\Udemy\DL\Face_detection_caffee\Model\res10_300x300_ssd_iter_140000.caffemodel'
caffe_model_proto=r'J:\Udemy\DL\Face_detection_caffee\Model\deploy.prototxt.txt'

def draw_live(img,predictions,prob=0.75):
	(h, w) = img.shape[:2]
	for i in range(0,predictions.shape[2]):
		if predictions[0,0,i,2] > prob:
			print(predictions[0,0,i,3:7])
			box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			cv2.rectangle(img,(startX, startY),(endX,endY),(0,255,2),2)
			img=img[startY:endY,startX:endX]
			img=cv2.resize(img,(224,224))
	return img

def main():
	caffemodel=cv2.dnn.readNetFromCaffe(caffe_model_proto,caffe_mode_path)
	cap=cv2.VideoCapture(0)
	while(cv2.waitKey(1))!=27:
		ret,img=cap.read()
		if not ret: break
		img=cv2.flip(img,1)
		print(img.shape)

		blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		predictions=caffemodel.forward(caffemodel.setInput(blob))
		print(predictions[0,0,0,2])
		img=draw_live(img,predictions)


		cv2.imshow('Live',img)
		# break

	cv2.destroyAllWindows()
	cap.release()

if __name__=='__main__':
	main()