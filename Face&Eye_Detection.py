#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


#Face detection
face_cascade = cv2.CascadeClassifier('C:\\Users\Ajay\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
img=cv2.imread("1.jpg")  #name of the image-->1.jpg
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
print(type(faces))
print(faces)


# In[3]:


#finding coordinates of face
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)


# In[4]:


resized=cv2.resize(img,(600,600))
cv2.imshow("final",resized)
cv2.waitKey(0)  #press any key !!
cv2.destroyAllWindows()


# In[5]:


#Eye detection

eye_cascade = cv2.CascadeClassifier('C:\\Users\Ajay\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml')

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# In[6]:


resized=cv2.resize(img,(600,600))
cv2.imshow("final",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
