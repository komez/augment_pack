import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import shutil

os.environ['OPENCV_IO_ENABLE_JASPER']= 'TRUE'
def purificate(xpath, ipath, save_img, save_xml):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)
    count0=0

    for xmlfile in tqdm(glob.glob(xpath+"/*.xml")):
        #xmlの読み込み
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        filename = root.find("filename").text
        element_objs = root.findall("object")
        cellcount=0
        #細胞が存在するか確認
        for element_obj in element_objs:
            class_name = element_obj.find('name').text

            if class_name=='silent' or class_name=='low' or class_name=='high':
                cellcount+=1
        #保存
        if not cellcount==0:
            name = os.path.basename(xmlfile)
            if not os.path.exists(save_xml+"/"+name):
                shutil.copy(xmlfile,save_xml+"/"+name)
            if not os.path.exists(save_img+"/"+filename):
                shutil.copy(ipath + "/" + filename, save_img+"/"+filename)
        
        else:
            count0+=1

    print("細胞数０の画像とXMLの組は{}個ありました。".format(count0))

def rotate1(xpath, ipath, save_img, save_xml):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)

    for xmlfile in tqdm(glob.glob(xpath+"/*.xml")):
        #xmlの読み込み
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        filename = root.find("filename").text
        folder = root.find("folder").text
        size_elements = root.findall("size")
        obj_elements = root.findall("object")

        #imgの読み込み
        image = cv2.imread(ipath+filename)

        #imgの保存
        #90
        new_name = filename.replace(".jp2","") + '-' + 'rotate_90' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        img_transpose = np.transpose(image, (1,0,2))
        img_90 = cv2.flip(img_transpose, 1)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_90)

        #180
        new_name_180 = filename.replace(".jp2","") + '-' + 'rotate_180' + '.jp2'
        new_path_180 = os.path.join(save_img, new_name_180)
        img_180 = cv2.flip(image, -1)
        if not os.path.isfile(new_path_180):
            cv2.imwrite(new_path_180,img_180)

        #270
        new_name_270 = filename.replace(".jp2","") + '-' + 'rotate_270' + '.jp2'
        new_path_270 = os.path.join(save_img, new_name_270)
        img_transpose_270 = np.transpose(image, (1,0,2))
        img_270 = cv2.flip(img_transpose_270, 0)
        if not os.path.isfile(new_path_270):
            cv2.imwrite(new_path_270, img_270)

        pwd_lines=[]
        for obj in obj_elements:
            class_name = obj.find('name').text 
            obj_bbox = obj.find('bndbox')
            x1 = int(obj_bbox.find("xmin").text)
            y1 = int(obj_bbox.find("ymin").text)
            x2 = int(obj_bbox.find("xmax").text)
            y2 = int(obj_bbox.find("ymax").text)
            # for angle 90
            
            h,w = image.shape[:2]
            angle_x1 = h - y2
            angle_y1 = x1
            angle_x2 = h -y1
            angle_y2 = x2
            lines = [new_name, ',', str(angle_x1), ',', str(angle_y1), ',', str(angle_x2), ',', str(angle_y2), ',', class_name, '\n']
            pwd_lines.append(lines)
            
            #for angle 180
            
            ang_x1 = w - x2
            ang_y1 = h - y2
            ang_x2 = w - x1
            ang_y2 = h - y1
            lines_180 = [new_name_180, ',', str(ang_x1), ',', str(ang_y1), ',', str(ang_x2), ',', str(ang_y2), ',', class_name, '\n']
            pwd_lines.append(lines_180)
            
            #for angle 270
            
            an_x1 = y1
            an_y1 = w - x2
            an_x2 = y2
            an_y2 = w - x1
            lines_270 = [new_name_270, ',', str(an_x1), ',', str(an_y1), ',', str(an_x2), ',', str(an_y2), ',', class_name, '\n']
            pwd_lines.append(lines_270)
            
            #
            
        for pwd_line in pwd_lines:
            xml_name = pwd_line[0].replace(".jp2",".xml")
            if not os.path.isfile(save_xml+"/"+xml_name):
                #Elementの完成
                root = ET.Element('annotations')
                ET.SubElement(root, 'filename').text = pwd_line[5]
                ET.SubElement(root, 'folder').text = "images"
                size = ET.SubElement(root, 'size')
                w,h,d = image.shape[:3]
                ET.SubElement(size,"width").text = str(w)
                ET.SubElement(size,"height").text = str(h)
                ET.SubElement(size,"depth").text = str(d)
                for pwd_lines in pwd_lines:
                        obj = ET.SubElement(root, 'object')
                        ET.SubElement(obj, "name").text = pwd_line[5]
                        ET.SubElement(obj, "pose").text = "Unspecified"
                        ET.SubElement(obj, "truncated").text = str(0)
                        ET.SubElement(obj, "difficult").text = str(0)
                        ET.SubElement(obj,"xmin").text = str(pwd_line[1])
                        ET.SubElement(obj,"ymin").text = str(pwd_line[2])
                        ET.SubElement(obj,"xmax").text = str(pwd_line[3])
                        ET.SubElement(obj,"ymax").text = str(pwd_line[4])
                #xml
                tree = ET.ElementTree(root)
                tree.write(save_xml+"/"+xml_name)
def rotate2(xpath, ipath, save_img, save_xml):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)
    for xmlfile in tqdm(glob.glob(xpath+"/*.xml")):
        #xmlの読み込み
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        filename = root.find("filename").text
        folder = root.find("folder").text
        size_elements = root.findall("size")
        obj_elements = root.findall("object")

        #imgの読み込み
        image = cv2.imread(ipath+filename)
        pwd_lines=[]

        #imageの保存
        new_name = filename.replace(".jp2","") + '_rotate_45' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        h,w,d = image.shape
        #45
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1)
        img_45 = cv2.warpAffine(image, mat, (w, h))
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_45)
        #135
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 135, 1)
        img_135 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_135' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_135)
        #225
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 225, 1)
        img_225 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_225' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_225)
        
        #315
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 315, 1)
        img_315 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_315' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_315)

        #15
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)
        img_15 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_15' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_15)
        
        #105
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 105, 1)
        img_105 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_105' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_105)

        #195
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 195, 1)
        img_195 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_195' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_195)
        
        #285
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 285, 1)
        img_285 = cv2.warpAffine(image, mat, (w, h))
        new_name = filename.replace(".jp2","") + '_rotate_285' + '.jp2'
        new_path = os.path.join(save_img, new_name)
        if not os.path.isfile(new_path):
            cv2.imwrite(new_path,img_285)

        for obj in obj_elements:
            class_name = obj.find('name').text 
            obj_bbox = obj.find('bndbox')
            x1 = obj_bbox.find("xmin").text
            y1 = obj_bbox.find("ymin").text
            x2 = obj_bbox.find("xmax").text
            y2 = obj_bbox.find("ymax").text
        # for angle 45
        h,w,d = image.shape
        
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1)
        img_45 = cv2.warpAffine(image, mat, (w, h))

        theta = -45
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_45 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_45 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_45 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_45 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_45 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_45 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_45 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_45 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r45_x1 = np.round(min([x1_45, x2_45, x3_45, x4_45])).astype(np.uint16)
        r45_y1 = np.round(min([y1_45, y2_45, y3_45, y4_45])).astype(np.uint16)
        r45_x2 = np.round(max([x1_45, x2_45, x3_45, x4_45])).astype(np.uint16)
        r45_y2 = np.round(max([y1_45, y2_45, y3_45, y4_45])).astype(np.uint16)

        
        lines = [new_name, ',', str(r45_x1), ',', str(r45_y1), ',', str(r45_x2), ',', str(r45_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        

        
        # for angle 135
        theta = -135
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_135 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_135 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_135 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_135 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_135 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_135 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_135 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_135 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r135_x1 = np.round(min([x1_135, x2_135, x3_135, x4_135])).astype(np.uint16)
        r135_y1 = np.round(min([y1_135, y2_135, y3_135, y4_135])).astype(np.uint16)
        r135_x2 = np.round(max([x1_135, x2_135, x3_135, x4_135])).astype(np.uint16)
        r135_y2 = np.round(max([y1_135, y2_135, y3_135, y4_135])).astype(np.uint16)
        
        lines = [new_name, ',', str(r135_x1), ',', str(r135_y1), ',', str(r135_x2), ',', str(r135_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        

        # for angle 225
        theta = -225
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_225 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_225 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_225 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_225 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_225 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_225 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_225 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_225 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r225_x1 = np.round(min([x1_225, x2_225, x3_225, x4_225])).astype(np.uint16)
        r225_y1 = np.round(min([y1_225, y2_225, y3_225, y4_225])).astype(np.uint16)
        r225_x2 = np.round(max([x1_225, x2_225, x3_225, x4_225])).astype(np.uint16)
        r225_y2 = np.round(max([y1_225, y2_225, y3_225, y4_225])).astype(np.uint16)
        
        lines = [new_name, ',', str(r225_x1), ',', str(r225_y1), ',', str(r225_x2), ',', str(r225_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        
        # for angle 315

        theta = -315
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_315 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_315 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_315 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_315 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_315 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_315 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_315 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_315 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r315_x1 = np.round(min([x1_315, x2_315, x3_315, x4_315])).astype(np.uint16)
        r315_y1 = np.round(min([y1_315, y2_315, y3_315, y4_315])).astype(np.uint16)
        r315_x2 = np.round(max([x1_315, x2_315, x3_315, x4_315])).astype(np.uint16)
        r315_y2 = np.round(max([y1_315, y2_315, y3_315, y4_315])).astype(np.uint16)
        
        lines = [new_name, ',', str(r315_x1), ',', str(r315_y1), ',', str(r315_x2), ',', str(r315_y2), ',', class_name, '\n']
        pwd_lines.append(lines)    

        # for angle 15

        theta = -15
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_15 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_15 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_15 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_15 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_15 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_15 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_15 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_15 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r15_x1 = np.round(min([x1_15, x2_15, x3_15, x4_15])).astype(np.uint16)
        r15_y1 = np.round(min([y1_15, y2_15, y3_15, y4_15])).astype(np.uint16)
        r15_x2 = np.round(max([x1_15, x2_15, x3_15, x4_15])).astype(np.uint16)
        r15_y2 = np.round(max([y1_15, y2_15, y3_15, y4_15])).astype(np.uint16)
        
        lines = [new_name, ',', str(r15_x1), ',', str(r15_y1), ',', str(r15_x2), ',', str(r15_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        
        
        # for angle 105
        theta = -105
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_105 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_105 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_105 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_105 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_105 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_105 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_105 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_105 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r105_x1 = np.round(min([x1_105, x2_105, x3_105, x4_105])).astype(np.uint16)
        r105_y1 = np.round(min([y1_105, y2_105, y3_105, y4_105])).astype(np.uint16)
        r105_x2 = np.round(max([x1_105, x2_105, x3_105, x4_105])).astype(np.uint16)
        r105_y2 = np.round(max([y1_105, y2_105, y3_105, y4_105])).astype(np.uint16)
        
        lines = [new_name, ',', str(r105_x1), ',', str(r105_y1), ',', str(r105_x2), ',', str(r105_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        
        # for angle 195

        theta = -195
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_195 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_195 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_195 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_195 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_195 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_195 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_195 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_195 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r195_x1 = np.round(min([x1_195, x2_195, x3_195, x4_195])).astype(np.uint16)
        r195_y1 = np.round(min([y1_195, y2_195, y3_195, y4_195])).astype(np.uint16)
        r195_x2 = np.round(max([x1_195, x2_195, x3_195, x4_195])).astype(np.uint16)
        r195_y2 = np.round(max([y1_195, y2_195, y3_195, y4_195])).astype(np.uint16)

        lines = [new_name, ',', str(r195_x1), ',', str(r195_y1), ',', str(r195_x2), ',', str(r195_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        
        
        # for angle 285

        theta = -285
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))

        x1_285 = (x1 - w/2)*cos - (y1 - h/2)*sin + w/2
        y1_285 = (x1 - w/2)*sin + (y1 - h/2)*cos + h/2
        x2_285 = (x2 - w/2)*cos - (y2 - h/2)*sin + w/2
        y2_285 = (x2 - w/2)*sin + (y2 - h/2)*cos + h/2
        x3_285 = (x2 - w/2)*cos - (y1 - h/2)*sin + w/2
        y3_285 = (x2 - w/2)*sin + (y1 - h/2)*cos + h/2
        x4_285 = (x1 - w/2)*cos - (y2 - h/2)*sin + w/2
        y4_285 = (x1 - w/2)*sin + (y2 - h/2)*cos + h/2

        r285_x1 = np.round(min([x1_285, x2_285, x3_285, x4_285])).astype(np.uint16)
        r285_y1 = np.round(min([y1_285, y2_285, y3_285, y4_285])).astype(np.uint16)
        r285_x2 = np.round(max([x1_285, x2_285, x3_285, x4_285])).astype(np.uint16)
        r285_y2 = np.round(max([y1_285, y2_285, y3_285, y4_285])).astype(np.uint16)

        lines = [new_name, ',', str(r285_x1), ',', str(r285_y1), ',', str(r285_x2), ',', str(r285_y2), ',', class_name, '\n']
        pwd_lines.append(lines)
        
        
        for pwd_line in pwd_lines:
            xml_name = pwd_line[0].replace(".jp2",".xml")
            if not os.path.isfile(save_xml+"/"+xml_name):
                #Elementの完成
                root = ET.Element('annotations')
                ET.SubElement(root, 'filename').text = pwd_line[5]
                ET.SubElement(root, 'folder').text = "images"
                size = ET.SubElement(root, 'size')
                w,h,d = image.shape[:3]
                ET.SubElement(size,"width").text = str(w)
                ET.SubElement(size,"height").text = str(h)
                ET.SubElement(size,"depth").text = str(d)
                for pwd_line in pwd_lines:
                        obj = ET.SubElement(root, 'object')
                        ET.SubElement(obj, "name").text = pwd_line[5]
                        ET.SubElement(obj, "pose").text = "Unspecified"
                        ET.SubElement(obj, "truncated").text = str(0)
                        ET.SubElement(obj, "difficult").text = str(0)
                        ET.SubElement(obj,"xmin").text = str(pwd_line[1])
                        ET.SubElement(obj,"ymin").text = str(pwd_line[2])
                        ET.SubElement(obj,"xmax").text = str(pwd_line[3])
                        ET.SubElement(obj,"ymax").text = str(pwd_line[4])
                #xml
                tree = ET.ElementTree(root)
                tree.write(save_xml+"/"+xml_name)

def confirm (xml_path, data_path):
    xml_files1=glob.glob(xml_path+"/*.xml")
    print('xmlfileの数は　： '+str(len(xml_files1)))
    image_files1=glob.glob(data_path+"/*.jp2")
    print('image_fileの数は　：'+str(len(image_files1)))
    
    silent=0
    low=0
    high=0
    count=0
    count1=0
    count2=0
    
    assert len(xml_files1) <= len(image_files1)
    
    for xml_file in xml_files1:
        et = ET.parse(xml_file)
        #xml fileを一つの木として認識
        element = et.getroot()
        #getroot()で枝を認識
        element_objs = element.findall('object')

        element_filename = element.find('filename').text
        base_filename = os.path.join(data_path, element_filename)
        if os.path.exists(base_filename)==False:
            #print('この画像ファイルは存在しません : '+str(element_filename))
            count2+=1
            continue

        #element_objが1つのbox、element_objsが1枚の画像に含まれるbox
        cellcount=0
        for element_obj in element_objs:
            class_name = element_obj.find('name').text

            if class_name=='silent':
                silent+=1
                cellcount+=1
            if class_name=='low':
                low+=1
                cellcount+=1
            if class_name=='high':
                high+=1
                cellcount+=1
        if cellcount==0:
            count1+=1
        else:
            count+=1

    print()
    print('xmlがあり、細胞が存在する画像の枚数は　：'+str(count))
    print('xmlがあり、細胞が存在しない画像の枚数は　：'+str(count1))
    print('xmlがあり、対応する画像が存在しないものは　：'+str(count2))
    print('silentの細胞数は　：'+str(silent))
    print('lowの細胞数は　： '+str(low))
    print('highの細胞数は　： '+str(high))
    

def set_copy(xpath, ipath, save_xml, save_img):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)
    for file in tqdm(glob.glob(xpath+"/*.xml")):
        name = os.path.basename(file)
        if not os.path.exists(save_xml+name):
            shutil.copy(file, save_xml+name)
            
    for file in tqdm(glob.glob(ipath+"/*.jp2")):
        name = os.path.basename(file)
        if not os.path.exists(save_img+name):
            shutil.copy(file, save_img+name)

def flip(xpath, ipath, save_xml, save_img):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)
    for xmlfile in tqdm(glob.glob(xpath+"/*.xml")):
        #xmlの読み込み
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        filename = root.find("filename").text
        folder = root.find("folder").text
        size_elements = root.findall("size")
        obj_elements = root.findall("object")
        image = cv2.imread(ipath+filename)
        f_points = [0, 1]
        
        save_objs = []
        for obj in obj_elements:
            class_name = obj.find('name').text 
            obj_bbox = obj.find('bndbox')
            xmin = int(obj_bbox.find("xmin").text)
            ymin = int(obj_bbox.find("ymin").text)
            xmax = int(obj_bbox.find("xmax").text)
            ymax = int(obj_bbox.find("ymax").text)

            for f in f_points:
                f_img = cv2.flip(image, f)
                w,h,d = image.shape[:3]

                if f == 1:
                    f_x1 = w-xmax
                    f_y1 = ymin
                    f_x2 = w-xmin
                    f_y2 = ymax
                    f_str = 'f1'
                elif f == 0:
                    f_x1 = xmin
                    f_y1 = h - ymax
                    f_x2 = xmax
                    f_y2 = h-ymin
                    f_str = 'f0'
                save_objs.append([class_name,f_x1, f_y1, f_x2, f_y2])
            
            new_name = filename.replace(".jp2","") + '-' + f_str + '.jp2'
            save_path = save_img + "/" + new_name
            # jp2の保存
            if not os.path.isfile(save_path):
                cv2.imwrite(save_path, f_img)

            # xml の保存
            xml_name = new_name.replace('.jp2','.xml')
            if not os.path.isfile(save_xml+"/"+xml_name):
                #Elementの完成
                root = ET.Element('annotations')
                ET.SubElement(root, 'filename').text = new_name
                ET.SubElement(root, 'folder').text = "images"
                size = ET.SubElement(root, 'size')
                ET.SubElement(size,"width").text = str(w)
                ET.SubElement(size,"height").text = str(h)
                ET.SubElement(size,"depth").text = str(d)
                for save_obj in save_objs:
                    obj = ET.SubElement(root, 'object')
                    ET.SubElement(obj, "name").text = save_obj[0]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = str(0)
                    ET.SubElement(obj, "difficult").text = str(0)
                    ET.SubElement(obj,"xmin").text = str(save_obj[1])
                    ET.SubElement(obj,"ymin").text = str(save_obj[2])
                    ET.SubElement(obj,"xmax").text = str(save_obj[3])
                    ET.SubElement(obj,"ymax").text = str(save_obj[4])

            #保存
            tree = ET.ElementTree(root)
            tree.write(save_xml+"/"+xml_name)

#https://qiita.com/koshian2/items/78de8ccd09dd2998ddfc
#データの色分布を加味した色の加減ができ、Data Augmentationとしてよく用いられるカラーチャンネルシフトよりも自然な画像が出来上がる
def pca_color_augmentation(xpath, ipath, save_img, save_xml):
    assert os.path.exists(xpath)
    assert os.path.exists(ipath)
    os.makedirs(save_xml, exist_ok = True)
    os.makedirs(save_img, exist_ok = True)
    for xmlfile in tqdm(glob.glob(xpath+"/*.xml")):
        #xmlの読み込み
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        filename = root.find("filename").text
        folder = root.find("folder").text
        size_elements = root.findall("size")
        obj_elements = root.findall("object")

        save_objs = []
        for obj in obj_elements:
            class_name = obj.find('name').text 
            obj_bbox = obj.find('bndbox')
            xmin = obj_bbox.find("xmin").text
            ymin = obj_bbox.find("ymin").text
            xmax = obj_bbox.find("xmax").text
            ymax = obj_bbox.find("ymax").text
            save_objs.append([class_name,xmin, ymin, xmax, ymax])
            
        #imgの読み込み
        image = cv2.imread(ipath+filename)
        
        #color_augmentation (image)
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8
        img = image.reshape(-1, 3).astype(np.float32)
        sf = np.sqrt(3.0/np.sum(np.var(img, axis=0)))
        img = (img - np.mean(img, axis=0))*sf 
        cov = np.cov(img, rowvar=False) # calculate the covariance matrix
        value, p = np.linalg.eig(cov) # calculation of eigen vector and eigen value 
        rand = np.random.randn(3)*0.08
        delta = np.dot(p, rand*value)
        delta = (delta*255.0).astype(np.int32)[np.newaxis, np.newaxis, :]
        img_out = np.clip(image+delta, 0, 255).astype(np.uint8)
        
        #imageの保存
        colorname = filename.replace(".jp2",'-color0'  +'.jp2') 
        
        if not os.path.exists(save_img+"/"+colorname):
            cv2.imwrite(save_img+"/"+colorname, img_out)
        
        xml_name = colorname.replace(".jp2",".xml")
        if not os.path.isfile(save_xml+"/"+xml_name):
            #Elementの完成
            root = ET.Element('annotations')
            ET.SubElement(root, 'filename').text = colorname
            ET.SubElement(root, 'folder').text = "images"
            size = ET.SubElement(root, 'size')
            w,h,d = image.shape[:3]
            ET.SubElement(size,"width").text = str(w)
            ET.SubElement(size,"height").text = str(h)
            ET.SubElement(size,"depth").text = str(d)
            for save_obj in save_objs:
                    obj = ET.SubElement(root, 'object')
                    ET.SubElement(obj, "name").text = save_obj[0]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = str(0)
                    ET.SubElement(obj, "difficult").text = str(0)
                    ET.SubElement(obj,"xmin").text = str(save_obj[1])
                    ET.SubElement(obj,"ymin").text = str(save_obj[2])
                    ET.SubElement(obj,"xmax").text = str(save_obj[3])
                    ET.SubElement(obj,"ymax").text = str(save_obj[4])
            #xml
            tree = ET.ElementTree(root)
            tree.write(save_xml+"/"+xml_name)
        