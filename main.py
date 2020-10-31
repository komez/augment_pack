import os
import shutil
from make_data.augment import purificate
from make_data.augment import set_copy
from make_data.augment import rotate1
from make_data.augment import rotate2
from make_data.augment import confirm
from make_data.augment import flip
from make_data.augment import pca_color_augmentation
from make_data.augment import split_text
from make_data.augment import comp_text
from make_data.augment import deldata



os.environ['OPENCV_IO_ENABLE_JASPER']= 'TRUE'

dir = '/Usersdata/kome/Drshirasaki/DrShirasaki/secretion_prediction/ssd/CytokineRelease_Dataset/CytokineRelease/'
xml_path =dir + 'datasets/rawdata/day4_xml/'
jp2_path = dir + 'datasets/rawdata/day4_split/'

presave_xml = dir + '/datasets/datasets/dataset201016/predata_xml/'
presave_img = dir + '/datasets/datasets/dataset201016/predata_jp2/'

save_xml = dir+ '/datasets/datasets/dataset201016/data_xml/'
save_img = dir + '/datasets/datasets/dataset201016/data_jp2/'

pretrain_txt = dir + '/datasets/datasets/dataset201016/pretrain_txt.txt'
preval_txt = dir + '/datasets/datasets/dataset201016/preval_txt.txt'
train_txt = dir + '/datasets/datasets/dataset201016/train_txt.txt'
val_txt = dir + '/datasets/datasets/dataset201016/preval_txt.txt'


txt = dir + '/datasets/datasets/dataset201016/data_xml.txt'

if __name__ == "__main__":
    #print("<purificate前>")
    purificate(xml_path, jp2_path, save_xml, save_img)
    split_text(save_xml, save_img, pretrain_txt, val_txt)
    deldata (save_xml, save_img, pretrain_txt, val_txt)
    #
    flip(save_xml, save_img, presave_xml, presave_img)
    rotate1(save_xml, save_img, presave_xml, presave_img)
    rotate2(save_xml, save_img, presave_xml, presave_img)    
    #ここから
    pca_color_augmentation(save_xml, save_img, save_xml, save_img, 3)
    pca_color_augmentation(presave_xml, presave_img, save_xml, save_img, 3)
    shutil.rmtree(presave_xml)
    shutil.rmtree(presave_img)

    purificate(save_xml, save_img, save_xml, save_img)
    comp_text (save_xml,pretrain_txt , train_txt)
    deldata (save_xml, save_img, train_txt, val_txt)
    shutil.rmtree(presave_xml)
    shutil.rmtree(presave_img)
    confirm (save_xml, save_img)
    #
    #comp_text (save_xml,pretrain_txt , train_txt)
    #split_text(save_xml, save_img, train_txt, val_txt)

    
    #confirm (presave_xml, presave_img)