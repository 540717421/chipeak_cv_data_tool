##1、根据标签类别筛选labelme数据集

input_dir = r'W:/disk2/my_work/' # 输入数据集父路径
handle_dict1 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
                    labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme') # 单个数据集路径字典，包含图片文件与对应的labelme标注文件
handle_dict2 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
                    labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme') # 单个数据集路径字典，包含图片文件与对应的labelme标注文件
datasets = [handle_dict1, handle_dict2] # 多个数据集转成列表
labelme = cdt.BaseLabelme(datasets, only_labelme=False) # 调用BaseLabelme基类进行数据集处理
output_dir = 'W:/disk2/my_work/self/' # 输出类别筛选数据集路径
labelme(True, name_classes=['run', 'crane'], shapes_type=['rectangle']) # 类别条件筛选设置
labelme.save_labelme(output_dir, replaces={input_dir: ''}) # 保存处理输出路径参数设置
print('当前筛选类别文件总数:%r' % len(labelme.data_infos))

##2、labelme数据集抠图
```
input_dir = r'W:/disk2/my_work/'
handle_dict1 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
                    labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme')
handle_dict2 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
                    labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme')
datasets = [handle_dict1, handle_dict2]
labelme = cdt.BaseLabelme(datasets, only_labelme=False)
output_dir = 'W:/disk2/my_work/self/'
labelme(True, name_classes=['run', 'crane'], shapes_type=['rectangle'])
labelme.save_labelme(output_dir, replaces={input_dir: ''}, min_pixel=512, )
print('当前抠图总数:%r' % labelme.num_crop_images) 
```
##3、coco数据集转labelme数据集
```
transform_save_dir = r'W:\disk2\02.标注数据集\01.目标检测-xjh\004.明火\01.公开数据'
input_images_dir = r'W:/disk2/my_work/'
input_coco_file = r'Z:/1.model_train/train.json'
replaces = input_images_dir
customize_images_dir = '00.images'
customize_labelme_dir = '01.labelme'
output_images_dir = None
data_infors = None
coco_dir = None
coco = cdt.Coco(False, data_infors, coco_dir, replaces, input_coco_file, transform_save_dir, customize_images_dir,customize_labelme_dir, input_images_dir, output_images_dir, )
coco.self2labelme()
```
##4、labelme数据集转coco数据集
```
input_dir = r'W:/disk2/my_work/'
output_coco_dir = 'W:/disk2/my_work/self/'
handle_dict1 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
              labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme')
handle_dict2 = dict(images_dir=input_dir + '01-crane_run-lxl-20211007-handwork/00.images',
              labelme_dir=input_dir + '01-crane_run-lxl-20211007-handwork/01.labelme')
datasets = [handle_dict,handle_dict2]
labelme = cdt.BaseLabelme(datasets, only_labelme=False)
replaces = ''
coco = cdt.Coco(False, labelme, output_coco_dir, replaces)
```
