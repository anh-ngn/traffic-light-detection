import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import torchvision
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchvision.ops import nms
from matplotlib.patches import Rectangle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
N_CLASS = 4
label_to_idx = {'go':1, 'warning':2, 'stop': 3}
idx_to_label = {v:k for k,v in label_to_idx.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_FasterRCNN():
    # Load Faster R-CNN
    mFRCNN = fasterrcnn_resnet50_fpn(pretrained = False)

    INP_FEATURES = mFRCNN.roi_heads.box_predictor.cls_score.in_features
    mFRCNN.roi_heads.box_predictor = FastRCNNPredictor(INP_FEATURES, N_CLASS)
    mFRCNN.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth'))
    mFRCNN.to(device)
    #mFRCNN.eval()
    return mFRCNN

def load_SSD():
    #Load SSD
    mSSD = ssd300_vgg16(weights = "DEFAULT", 
                     nms_thresh = 0.3,
#                      topk_candidates = 400,
#                      detections_per_img = 200,
#                      score_thresh = 0.1
                    )
    mSSD.head.classification_head = ssd300_vgg16(weights=None, num_classes=N_CLASS).head.classification_head
    mSSD.load_state_dict(torch.load('ssd300_vgg16.pth'))
    mSSD.to(device)
    #mSSD.eval()
    return mSSD

def load_RetinaNet():
    #Load RetinaNet
    mRetina = retinanet_resnet50_fpn_v2(weights='DEFAULT',
                                  trainable_backbone_layers=1)

    mRetina.head = retinanet_resnet50_fpn_v2(weights=None, num_classes=N_CLASS).head
    mRetina.load_state_dict(torch.load('retinanet_resnet50_fpn_v2.pth'))
    mRetina.to(device)
    #mRetina.eval()

    return mRetina


def img_preprocess(img):
    processed = img
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,512))
    ])
    processed_img = transform_norm(img).float()
    processed_img = processed_img.unsqueeze_(0)
    processed_img.to(device)
    return processed_img


def filterBoxes(output,nms_th=0.3,score_threshold=0.5):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    # Non Max Supression
    mask = nms(boxes,scores,nms_th)
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels





def predict_image(image_path, model, output_path = "output.png"):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # Store the original size
    
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
    ])
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    #img_normalized = img_normalized[0]
    #st.write(img_normalized.shape)
    
    with torch.no_grad():
        model.eval()
        predictions = model(img_normalized)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        pred_boxes, pred_scores, pred_labels = filterBoxes(predictions[0])
        pred_boxes = torch.tensor(pred_boxes)
        pred_scores = torch.tensor(pred_scores)
        pred_labels = torch.tensor(pred_labels)
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            # Resize the bounding boxes to the original size
            box_resized = [
                box[0] * original_size[0] / 512,
                box[1] * original_size[1] / 512,
                box[2] * original_size[0] / 512,
                box[3] * original_size[1] / 512
            ]
            
            rect = Rectangle(
                xy=(box_resized[0], box_resized[1]),
                width=box_resized[2] - box_resized[0],
                height=box_resized[3] - box_resized[1],
                edgecolor='r',
                fill=False
            )
            ax.add_patch(rect)
            ax.text(box_resized[0], box_resized[1], f'{idx_to_label[label.item()]}: {score:.2f}', color='r')
        
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)


def main():
    #mFasterRCNN, mSSD, mRetina = load_models()
    st.title("Traffic Light Detection")
    st.text("Build with Streamlit and OpenCV")

    choice = st.selectbox('Model', ['Faster RCNN', 'SSD', 'RetinaNet']) 
    
    image_file = st.file_uploader("Upload Image", type = ['jpg', 'png', 'jpeg'])
    
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Origial Image")
        st.image(our_image)

        if choice == 'Faster RCNN':
            st.write('Faster RCNN')
            mFasterRCNN = load_FasterRCNN()
            predict_image(image_file, mFasterRCNN)
            img = Image.open('output.png')
            st.image(img)

        if choice == 'SSD':
            st.write('SSD')
            mSSD = load_SSD()
            predict_image(image_file, mSSD)
            img = Image.open('output.png')
            st.image(img)

        if choice == 'RetinaNet':
            st.write('RetinaNet')
            mRetina = load_RetinaNet()
            predict_image(image_file, mRetina)
            img = Image.open('output.png')
            st.image(img)
        
    else:
        st.text("Could not find the image")


if __name__ == '__main__':
    main()
