import cv2

# # Basic reading, showing, resizing operation
# # remember to use\\ instead of \ in paths 

# img=cv2.imread("C:\\Users\\Mohit\\Desktop\\IMG_20180604_205904.jpg")

# # normal resizing
# # resized=cv2.resize(img, (200,200))

# #making the image symmetric
# #divind the size by 2(integer division)
# # ie new image shape= old/2
# resized=cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))

# cv2.imshow("Legend",resized)
# cv2.waitKey(0) <- 0 means until any key is pressed, 2000,etc means time in millisecods


# cv2.destroyAllWindows()



#<------------------ Face Detection starts here --------------->
# Create a cascade classifier object
face_cascade=cv2.CascadeClassifier("C:\\Users\\Mohit\\Documents\\GitHub\\fnmd\\haarcascade_frontalface_default.xml")

# reading the image
img=cv2.imread("C:\\Users\\Mohit\\Desktop\\IMG_20180604_205904.jpg")
# reading the image as gray scale
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Search the co-ordinates of image
faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

cv2.imshow("Gray",img)
cv2.waitKey(0)
cv2.destroyAllWindows()