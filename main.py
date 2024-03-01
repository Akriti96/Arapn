import cv2
import numpy as np
modelConfiguration='cfg/yolov3.cfg'
modelWeights='yolov3.weights'

# Load the YOLO model detection network
yoloNetwork= cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
# print(yoloNetwork)

# coco files load
labelsPath='coco.names'
labels=open(labelsPath).read().strip().split('\n')
# print(labels)


image= cv2.imread('static/img1.jpg')
image=cv2.resize(image,(750,500))
# print(image.shape)

dimensions=image.shape[:2]
# print(dimensions)

Height=dimensions[0]
Width=dimensions[1]

confidenceThreshold=0.5
NMSThreshold=0.3

blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
# print(blob)

yoloNetwork.setInput(blob)

# layername
layerName=yoloNetwork.getUnconnectedOutLayersNames()
layersOutputs= yoloNetwork.forward(layerName)

print(layerName)
# print(layersOutputs)

boxes=[]
confidences=[]
classIds=[]

for output in layersOutputs:
    for detection in output:
        # get class scores and id of class with highest score
        # print(detection)
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        # print(scores)
        
        if confidence > confidenceThreshold:
            box = detection[0:4] * np.array([Width, Height, Width, Height])
            (centerX, centerY,w,h) = box.astype('int')
            x = int(centerX - (w/2))
            y = int(centerY - (h/2))

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            classIds.append(classId)
       
indexes=cv2.dnn.NMSBoxes(boxes,confidences,confidenceThreshold,NMSThreshold) 
print(boxes)   
 
for i in range(len(boxes)):
    if i in indexes:
        x=boxes[i][0]
        y=boxes[i][1]
        w=boxes[i][2]
        h=boxes[i][3]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        label=labels[classIds[i]]
        
        text='{}:{:2f}'.format(label,confidences[i]*100)
        cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    
    

cv2.imshow("image",image)
cv2.waitKey(0)

# other objects not detected...
