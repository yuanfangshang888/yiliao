import aidlite_gpu
import cv2
from cvs import *
import numpy as np
import os
import time






model_path = "/home/gongye/tflite/gongye.tflite"
image_path = "/home/gongye/test"
NUMS_CLASS = 7

confThresh = 0.3
NmsThresh = 0.45



# 输入格式 （8400,11）
def postProcess(pres, confThresh, NmsThresh):
    boxes_out = []
    scores_out = []
    class_out = []
    for pred in pres:

        pred_class = pred[4:]
        box_ = pred[0:4]
        # pred_class=(pred_class-min(pred_class))/(max(pred_class)-min(pred_class))
        class_index = np.argmax(pred_class)
        if pred_class[class_index] > 0.3:
            # box=np.array([round(pred[2]-0.5*pred[0]),round(pred[3]-0.5*pred[1]),round(pred[0]),round(pred[1])])
            box_ = pred[0:4]  # w,h,xc,yc
            box = np.array([round((pred[2] / 2 - pred[0])), round((pred[3] / 2 - pred[1])), round(pred[0] * 2),
                            round(pred[1] * 2)])
            boxes_out.append(box)
            score = pred_class[class_index]
            scores_out.append(score)
            class_out.append(class_index)

    result_boxes = cv2.dnn.NMSBoxes(boxes_out, np.array(scores_out), confThresh, NmsThresh)
    # detections=[]
    boxes = []
    scores = []
    classes = []
    for result_box in result_boxes:
        index = int(result_box)
        box = boxes_out[index]
        score = scores_out[index]
        class_type = class_out[index]
        boxes.append(box)
        scores.append(score)
        classes.append(class_type)
    return boxes, scores, classes


def draw(img, xscale, yscale, boxes, scores, classes):
    width = img.shape[1]
    w1 = 1620
    w2 = 2350
    w3 = width
    S1 = []
    S2 = []
    S3 = []
    S1_res = [False for i in range(NUMS_CLASS)]
    S2_res = [False for i in range(NUMS_CLASS)]
    S3_res = [False for i in range(NUMS_CLASS)]
    S_res = [S1_res, S2_res, S3_res]

    img_ = img.copy()
    # 遍历所有box，按照分割区域将box归类
    for i in range(len(boxes)):
        # boxes=[x1,y1,w,h]
        box = boxes[i]
        score = scores[i]
        class_ = int(classes[i])
        # class_text=label[class_]
        # detect=[round(box[0]*xscale),round(box[1]*yscale),round((box[0]+box[2])*xscale),round((box[1]+box[3])*yscale)]
        detect = [round(box[0] * xscale), round(box[1] * yscale), round(box[0] * xscale + (box[2]) * xscale),
                  round(box[1] * yscale + (box[3]) * yscale)]
        text = "{}:{:.2f}".format(label[class_], float(score))
        img_ = cv2.rectangle(img_, (detect[0], detect[1]), (detect[2], detect[3]), (0, 255, 0), 2)
        cv2.putText(img_, text, (detect[0], detect[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)

        # 分割为三块
        if (detect[0] <= w1):
            p1 = []
            p1.append(detect)
            p1.append(class_)
            p1.append(score)
            S1.append(p1)
        elif (w1 < detect[0] <= w2):
            p2 = []
            p2.append(detect)
            p2.append(class_)
            p2.append(score)
            S2.append(p2)
        elif (w2 < detect[0] <= w3):
            p3 = []
            p3.append(detect)
            p3.append(class_)
            p3.append(score)
            S3.append(p3)

            # 判断每个分割图像中的结果
    index = 0
    for S in [S1, S2, S3]:
        for i in range(len(S)):
            p1 = S[i]
            box_temp = p1[0]
            class_temp = p1[1]
            score_temp = p1[2]
            S_res[index][class_temp] = True
        index += 1

    # 最终分割输出结果true or false
    S_out = [False, False, False]
    index_out = 0
    for s_r in S_res:
        c0 = s_r[0]
        c1 = s_r[1]
        c2 = s_r[2]
        c3 = s_r[3]
        c4 = s_r[4]
        c5 = s_r[5]
        c6 = s_r[6]

        if (c0 & c1 & c2 & c3 & (~c4) & (~c5) & (~c6)):
            S_out[index_out] = True
        elif (c0 & c1 & c2 & (~c3) & (~c4) & c5 & (~c6)):
            S_out[index_out] = True
        index_out += 1

    # 打印分割结果
    cv2.putText(img_, "OK" if S_out[0] == True else "NG", (w1 - 200, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
    cv2.putText(img_, "OK" if S_out[1] == True else "NG", (w2 - 200, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
    cv2.putText(img_, "OK" if S_out[2] == True else "NG", (w3 - 200, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)

    return img_


label = ["rubber stopper", "push rod tail", "needle tail", "mouth", "crooked mouth", "screw mouth", "small rubber plug"]

if __name__ == "__main__":


   


    # 1.初始化aidlite类并创建aidlite对象
    aidlite = aidlite_gpu.aidlite()
    print("ok")

    # 2.加载模型
    value = aidlite.ANNModel(model_path, [640 * 640 * 3 * 4], [8400 * 11 * 4], 4, 0)
    print("gpu:", value)
    # file_names=os.listdir(image_path)
    # root,dirs,files = os.walk(image_path)
    for root, dirs, files in os.walk(image_path):
        num = 0
        for file in files:
            file = os.path.join(root, file)
            frame = cv2.imread(file)
            x_scale = frame.shape[1] / 640
            y_scale = frame.shape[0] / 640

            img = cv2.resize(frame, (640, 640))
            # img_copy=img.co
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = img.astype(dtype=np.float32)
            
            print(img.shape)

            # 3.传入模型输入数据
            aidlite.setInput_Float32(img)

            # 4.执行推理
            start = time.time()

            aidlite.invoke()

            end = time.time()
            timerValue = 1000 * (end - start)
            print("infer time(ms):{0}", timerValue)

            # 5.获取输出
            pred = aidlite.getOutput_Float32(0)
            # print(pred.shape)
            pred = np.array(pred)
            print(pred.shape)
            pred = np.reshape(pred, (8400, 11))
            # pred=np.reshape(pred,(11,8400)).transpose()
            print(pred.shape)  # shape=(8400,11)

            # 6.后处理,解析输出
            boxes, scores, classes = postProcess(pred, confThresh, NmsThresh)

            # 7.绘制保存图像
            ret_img = draw(frame, x_scale, y_scale, boxes, scores, classes)

            ret_img = ret_img[:, :, ::-1]
            num += 1
            image_file_name = "/home/gongye/result/tflite/res" + str(num) + ".jpg"
            # 8.保存图片
            cv2.imwrite(image_file_name, ret_img)

