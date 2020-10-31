# augment_pack

実行ファイルでパスを指定する。
```bash
- main.py
dir = '/dataset'
xml_path =dir + '/xmldata/'
jp2_path = dir + '/jp2data/'
train_txt = dir + '/text/train_txt.txt'
val_txt = dir + '/text/val_txt.txt'
```
メソッドファイル
```
- augment.py
#使用可能データの精製とフォルダの作成
purificate(xml_path, jp2_path, save_xml, save_img)

#クラスごとのボックス数の算出やその他のデータセットの情報をxmlファイルから調べる
confirm (save_xml, save_img)

#不均衡データを対策するためのダウンサンプリング
split_text(save_xml, save_img, pretrain_txt, val_txt)

#使用しないデータの削除
deldata (save_xml, save_img, pretrain_txt, val_txt)

#augmentation
flip(save_xml, save_img, presave_xml, presave_img)
rotate1(save_xml, save_img, presave_xml, presave_img)
rotate2(save_xml, save_img, presave_xml, presave_img)    
pca_color_augmentation(save_xml, save_img, save_xml, save_img, 3)
cutout(save_xml, save_img, save_xml, save_img)

#textファイルの完成
comp_text (save_xml,pretrain_txt , train_txt)
deldata (save_xml, save_img, train_txt, val_txt)
```
