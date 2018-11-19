#! /home/xtiandr/.virtualenvs/CpE_Elec_2/bin/python

from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import os
import numpy as np
import subprocess
baseheight = 280
classesFile = "object_names.xdr";
modelConfiguration = "rcnn.config";
modelWeights = "trained_objects.weights";
image = cv2.imread("sample/sample.jpg")
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
class_ids = []
class_count = [0] * 80
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
class_size = 80
classes = None
strng = "Detect\n"
########################################################################################################################

def OpenFile():
    name = askopenfilename(filetypes =(("All Files","*.*"),
                                       ("Portable Network Graphics File", "*.png"),
                                       ("Portable Network Graphics File", "*.png"),
                                       ("JPEG File", "*.jpg"),
                                       ("Graphic Interchange Format File", "*.gif")),
                           title = "Choose a file.")
    dir_loc.set(name)
    images = Image.open(name)
    hpercent = (baseheight / float(images.size[1]))
    wsize = int((float(images.size[0]) * float(hpercent)))
    images = images.resize((wsize, baseheight), Image.ANTIALIAS)
    root.img = ImageTk.PhotoImage(images)
    lb_img.configure(image=root.img)
    lb_img.update()

def ClassifyImage(location):
    setCountZero()
    classesFile = "object_names.xdr";
    modelConfiguration = "rcnn.config";
    modelWeights = "trained_objects.weights";
    scale = 0.00392
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    classes = None
    image = cv2.imread(location)
    Width = image.shape[1]
    Height = image.shape[0]
    net = cv2.dnn.readNet(modelWeights, modelConfiguration)
    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imwrite("output/output.jpg", image)
    images2 = Image.open("output/output.jpg")
    hpercent = (baseheight / float(images2.size[1]))
    wsize = int((float(images2.size[0]) * float(hpercent)))
    images2 = images2.resize((wsize, baseheight), Image.ANTIALIAS)
    root.img2 = ImageTk.PhotoImage(images2)
    lb_outimg.configure(image=root.img2)
    lb_outimg.update()
    text_output.set(getOverallCountObject())

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    label2 = '%.2f' % confidence
    color = COLORS[class_id]
    class_count[class_id] = class_count[class_id] + 1
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label + ':' + label2, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

########################################################################################################################

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

########################################################################################################################

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(modelWeights, modelConfiguration)
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))

########################################################################################################################

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

########################################################################################################################

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
cv2.imwrite("output/output.jpg",image)

########################################################################################################################

def getOverallCountObject():
    strng = ""
    for i in range(0, 80):
        if class_count[i] != 0:
            strng = strng + str(class_count[i]) + " " + str(classes[i]) + "\n"
    return  strng


def setCountZero():
    for i in range(0, 80):
        class_count[i] = 0


def openFolder():
    os.chdir(os.path.dirname(__file__))
    strs = os.getcwd()
    strs = "dolphin " + strs + "/output"
    proc = subprocess.Popen(strs, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

def openFile():
    os.chdir(os.path.dirname(__file__))
    strs = os.getcwd()
    strs = "gwenview " + strs + "/output/output.jpg"
    proc = subprocess.Popen(strs, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)





########################################################################################################################

root = Tk()
dir_loc = StringVar()
images = Image.open("sample/sample.jpg")
hpercent = (baseheight / float(images.size[1]))
wsize = int((float(images.size[0]) * float(hpercent)))
images = images.resize((wsize, baseheight), Image.ADAPTIVE)
root.img = ImageTk.PhotoImage(images)
text_output = StringVar()
text_output.set(getOverallCountObject())

########################################################################################################################

images2 = Image.open("output/output.jpg")
hpercent = (baseheight / float(images2.size[1]))
wsize = int((float(images2.size[0]) * float(hpercent)))
images2 = images2.resize((wsize, baseheight), Image.ANTIALIAS)
root.img2 = ImageTk.PhotoImage(images2)
root.title("Object Detection and Classification from Image using RCNN and OpenCV")
root.geometry("1080x720")
root.resizable(0,0)
frame = Frame(root)
frame.pack(fill=BOTH, expand=1)


lb_input = LabelFrame(frame)
lb_input.configure(text="Input",height=440, width=525)
lb_input.place(x=10,y=10)

lb_help = LabelFrame(frame)
lb_help.configure(text="Instruction and About",height=260, width=525)
lb_help.place(x=10,y=450)

lab01 = Label(lb_help)
lab01.configure(text="How to use this system:\n1. Select an image by pressing a choose file (...) button.\n2. Press the Classify Image button.\n\nAbout:", justify=LEFT, anchor='nw')
lab01.place(x=5,y=0, width=515)

lab02 = Label(lb_help)
lab02.configure(text="Object Detection and Classification from Image\nusing RCNN and OpenCV\n\nCreated by: Christian D. Remonde\n")
lab02.place(x=5,y=90, width=515)


label01 = Label(lb_input)
label01.configure(text="Choose an image:")
label01.place(x=10, y=10, height=25)

txt_dir = Entry(lb_input)
txt_dir.configure(textvariable=dir_loc)
txt_dir.place(x=130, y=10, height=25, width=300)

bt_choose = Button(lb_input)
bt_choose.configure(text="...", command=OpenFile)
bt_choose.place(x=440, y=10, height=25)

lb_img = Label(lb_input)
lb_img.configure(borderwidth=2, relief=SOLID, image= root.img)
lb_img.place(x=10, y=80, height=280, width=500)

bt_classify = Button(lb_input)
bt_classify.configure(text="Classify the Image", command=lambda:ClassifyImage(txt_dir.get()))
bt_classify.place(x=150, y=375, width=200)

lb_output = LabelFrame(frame)
lb_output.configure(text="Output",height=700, width=525)
lb_output.place(x=545,y=10)

lb_outimg = Label(lb_output)
lb_outimg.configure(borderwidth=2, relief=SOLID, image= root.img2)
lb_outimg.place(x=10, y=50, height=280, width=500)

bt_openfolder = Button(lb_output)
bt_openfolder.configure(text="Open Output Folder", command=openFolder)
bt_openfolder.place(x=10, y=5, width=245)

bt_openfile = Button(lb_output)
bt_openfile.configure(text="Open Full Image", command=openFile)
bt_openfile.place(x=265, y=5, width=245)


txtlab = Label(lb_output)
txtlab.configure(text="================= OBJECT COUNT ================")
txtlab.place(x=10, y=380, width=500)

txtar_output = Label(lb_output)
txtar_output.configure(textvariable=text_output, borderwidth=2, relief=SOLID, justify=LEFT, anchor='nw')
txtar_output.place(x=10, y=400, width=500, height=200)

root.mainloop()
