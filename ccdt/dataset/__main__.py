import argparse
import sys
import ast
import cpdt.dataset as cpdt



# from .coco.coco import Coco
# python __main__.py --input_datasets="[{'fromat': 'labelme','images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir': 'Z:/4.my_work/9.zy/00/01.labelme'},]"
# --input_datasets="[{'images_dir': 'Z:/4.my_work/9.zy/00/00.images', 'labelme_dir':
# [
#     {'fromat': 'labelme', 'images_dir': '', 'labelme_dir': ''},
#     {'fromat': 'coco', 'images_dir': '', 'annotation_file': ''},
#     {'fromat': 'labelme', 'images_dir': '', 'labelme_dir': ''},
# ]
def process_labelme():
    input_dir = r'Z:/4.my_work/9.zy/'
    # handle_dict1 = dict(images_dir=input_dir + '00/00.images',
    #                     labelme_dir=input_dir + '00/01.labelme')
    # handle_dict2 = dict(images_dir=input_dir + '11/00.images',
    #                     labelme_dir=input_dir + '11/01.labelme')
    # datasets = [handle_dict1, handle_dict2]
    images_dir = r'Z:/4.my_work/9.zy/00/00.images'
    labelme_dir = r'Z:/4.my_work/9.zy/00/01.labelme'
    # labelme = cdttest.BaseLabelme(datasets, only_labelme=False)
    labelme = cpdt.BaseLabelme(labelme_dir, images_dir, only_annt=False)
    output_dir = 'Z:/4.my_work/9.zy/final'
    # 替换路径，把公共路径写到字典中进行传入，min_pixel=512设置像素大小参数，小于512像素不进行截取
    labelme.crop_objs(output_dir, replaces={input_dir: ''}, min_pixel=512, )
    print('当前抠图总数:%r' % labelme.num_crop_images)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_datasets', type=ast.literal_eval, help="输入labelme数据集列表字典路径")
    parser.add_argument('--output_dir', type=str, help="输入抠图保存路径")
    parser.add_argument('--input_dir', type=str, help="输入替换路径")
    parser.add_argument('--function', type=str, help="输入操作功能参数:convert,filter,matting,rename,visualize，只能输入单个")
    parser.add_argument('--only_annotation', action="store_true",
                        help="当不输入--only_annotation的时候，默认为False；输入--only_annotation的时候，才会触发True值。False处理labelme和图片，True只处理labelme")
    parser.add_argument('--name_classes', type=ast.literal_eval, help="输入类别筛选参数，单个与多个都可以输入")
    parser.add_argument('--type_shapes', type=ast.literal_eval, help="输入形状筛选参数，单个与多个都可以输入")
    args = parser.parse_args()
    return args


def load_datasets(datasets_info):
    datasets = []
    for dataset_info in datasets_info:
        # print(dataset)
        # if dataset['format'] == 'coco':
        #     dataset = Coco(dataset)
        if dataset_info['format'] == 'labelme':
            # 把整个对象对象追加到列表中
            dataset = cpdt.BaseLabelme(dataset_info['labelme_dir'], dataset_info['images_dir'], args.only_annotation)
            datasets.append(dataset)
    return datasets


def main(args):
    print(args)
    # 1.加载数据集
    datasets = load_datasets(args.input_datasets)

    # 2.对数据进行处理
    if args.function == 'matting':
        for dataset in datasets:
            print(dataset)
            dataset.crop_objs(args.output_dir, replaces={args.input_dir: ''}, min_pixel=512)
            print('当前抠图总数:%r' % dataset.num_crop_images)
    elif args.function == 'merge':
        pass
    elif args.function == 'convert':
        pass
    elif args.function == 'rename':
        pass
    elif args.function == 'visualize':
        pass
    elif args.function == 'filter':
        for dataset in datasets:
            print(dataset)
            dataset(True, name_classes=args.name_classes, type_shapes=args.type_shapes)
            dataset.save_labelme(args.output_dir, replaces={args.input_dir: ''})
        pass

    # 3.输出


if __name__ == '__main__':
    # process_labelme()
    args = parser_args()
    print(args)
    main(args)
