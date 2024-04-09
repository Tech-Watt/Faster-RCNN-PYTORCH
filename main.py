import torch
import cvzone
import cv2
from torchvision import transforms
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
model.eval()

classnames = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()

#print(classnames)
image = cv2.imread('cat.png')
image = cv2.resize(image,(640,480))

image_transform = transforms.ToTensor()
img = image_transform(image)
print(type(image))
print(type(img))

with torch.no_grad():
    pred = model([img])
    #print(pred)
    #print(pred[0].keys())

    bbox,scores,labels = pred[0]['boxes'],pred[0]['scores'],pred[0]['labels']
    conf = torch.argwhere(scores > 0.70).shape[0]
    for i in range(conf):
        x,y,w,h = bbox[i].numpy().astype('int')
        classname = labels[i].numpy().astype('int')
        class_detected = classnames[classname]
        cv2.rectangle(image,(x,y),(w,h),(0,0,255),4)
        cvzone.putTextRect(image,class_detected,[x+8,y-12],scale=2,border=2)
        cv2.imwrite('data1.png',image)


cv2.imshow('frame',image)
cv2.waitKey(0)
