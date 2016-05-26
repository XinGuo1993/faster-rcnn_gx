import cv2
import exif

img=exif.load_exif_jpg('tr.jpg')
cv2.imwrite('tr1.jpg',img)
