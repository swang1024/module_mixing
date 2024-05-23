import numpy as np


def load_ds_info(dataset_name):
    global task_list, tasks, data_path
    if dataset_name == 'cifar100_mix':
        ### multi class scenario ###
        task1 = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']  # fish
        task2 = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']  # flowers
        task3 = ['beaver', 'dolphin', 'otter', 'seal', 'whale']  # aquatic mammals
        task4 = ['bottle', 'bowl', 'can', 'cup', 'plate']  # food containers
        task5 = ['bear', 'leopard', 'lion', 'tiger', 'wolf']  # large carnivores
        task6 = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']  # fruit and vegetables
        task7 = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']  # small mammals
        task8 = ['baby', 'boy', 'girl', 'man', 'woman']  # people
        task9 = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']  # trees
        task10 = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train']  # vehicles 1
        task11 = ['clock', 'keyboard', 'lamp', 'telephone', 'television']  # household electrical devices
        task12 = ['bed', 'chair', 'couch', 'table', 'wardrobe']  # household furniture
        task13 = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach']  # insects
        task14 = ['bridge', 'castle', 'house', 'road', 'skyscraper']  # large man-made outdoor things
        task15 = ['cloud', 'forest', 'mountain', 'plain', 'sea']  # large natural outdoor scenes
        task16 = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']  # large omnivores and herbivores
        task17 = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']  # medium-sized mammals
        task18 = ['crab', 'lobster', 'snail', 'spider', 'worm']  # non-insect invertebrates
        task19 = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']  # reptiles
        task20 = ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']  # vehicles 2

        task21 = ['hamster', 'cattle', 'maple_tree', 'squirrel', 'chimpanzee']
        task22 = ['house', 'bridge', 'bicycle', 'baby', 'lamp']
        task23 = ['shark', 'oak_tree', 'shrew', 'beaver', 'plate']
        task24 = ['willow_tree', 'crocodile', 'tiger', 'otter', 'telephone']
        task25 = ['bus', 'aquarium_fish', 'camel', 'skunk', 'apple']
        task26 = ['bee', 'forest', 'tank', 'sweet_pepper', 'fox']
        task27 = ['snail', 'whale', 'clock', 'lion', 'cockroach']
        task28 = ['rabbit', 'castle', 'pine_tree', 'cloud', 'boy']
        task29 = ['butterfly', 'road', 'rocket', 'skyscraper', 'wardrobe']
        task30 = ['spider', 'crab', 'tractor', 'mouse', 'seal']
        task31 = ['bed', 'elephant', 'beetle', 'keyboard', 'train']
        task32 = ['plain', 'table', 'ray', 'worm', 'sea']
        task33 = ['trout', 'bear', 'kangaroo', 'caterpillar', 'turtle']
        task34 = ['motorcycle', 'bottle', 'orchid', 'chair', 'leopard']
        task35 = ['dolphin', 'can', 'porcupine', 'cup', 'pickup_truck']
        task36 = ['poppy', 'wolf', 'pear', 'bowl', 'man']
        task37 = ['snake', 'tulip', 'streetcar', 'palm_tree', 'girl']
        task38 = ['lawn_mower', 'television', 'mushroom', 'lizard', 'raccoon']
        task39 = ['orange', 'dinosaur', 'possum', 'lobster', 'flatfish']
        task40 = ['mountain', 'couch', 'rose', 'woman', 'sunflower']

        task_list = {'task1': task1, 'task2': task2, 'task3': task3, 'task4': task4, 'task5': task5,
                     'task6': task6, 'task7': task7, 'task8': task8, 'task9': task9, 'task10': task10,
                     'task11': task11, 'task12': task12, 'task13': task13, 'task14': task14, 'task15': task15,
                     'task16': task16, 'task17': task17, 'task18': task18, 'task19': task19, 'task20': task20,
                     'task21': task21, 'task22': task22, 'task23': task23, 'task24': task24, 'task25': task25,
                     'task26': task26, 'task27': task27, 'task28': task28, 'task29': task29, 'task30': task30,
                     'task31': task31, 'task32': task32, 'task33': task33, 'task34': task34, 'task35': task35,
                     'task36': task36, 'task37': task37, 'task38': task38, 'task39': task39, 'task40': task40
                     }
        tasks = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9',
                 'task10', 'task11', 'task12', 'task13', 'task14', 'task15', 'task16', 'task17', 'task18', 'task19',
                 'task20', 'task21', 'task22', 'task23', 'task24', 'task25', 'task26', 'task27', 'task28',
                 'task29', 'task30', 'task31', 'task32', 'task33', 'task34', 'task35', 'task36',
                 'task37', 'task38', 'task39', 'task40']
        data_path = "./data/cifar100/"
    elif dataset_name == 'tiny_imagenet_50_overlap':
        task1 = ['n02124075', 'n04067472', 'n04540053', 'n04099969', 'n07749582',
                 'n01641577', 'n02802426', 'n09246464', 'n07920052', 'n03970156',
                 'n03891332', 'n02106662', 'n03201208', 'n02279972', 'n02132136', 
                 'n04146614', 'n07873807', 'n02364673', 'n04507155', 'n03854065', 
                 'n03838899', 'n03733131', 'n01443537', 'n07875152', 'n03544143', 
                 'n09428293', 'n03085013', 'n02437312', 'n07614500', 'n03804744', 
                 'n04265275', 'n02963159', 'n02486410', 'n01944390', 'n09256479', 
                 'n02058221', 'n04275548', 'n02321529', 'n02769748', 'n02099712', 
                 'n07695742', 'n02056570', 'n02281406', 'n01774750', 'n02509815', 
                 'n03983396', 'n07753592', 'n04254777', 'n02233338', 'n04008634']
        task2 = ['n09428293', 'n03085013', 'n02437312', 'n07614500', 'n03804744', 
                 'n04265275', 'n02963159', 'n02486410', 'n01944390', 'n09256479', 
                 'n02058221', 'n04275548', 'n02321529', 'n02769748', 'n02099712', 
                 'n07695742', 'n02056570', 'n02281406', 'n01774750', 'n02509815', 
                 'n03983396', 'n07753592', 'n04254777', 'n02233338', 'n04008634', 
                 'n02823428', 'n02236044', 'n03393912', 'n07583066', 'n04074963', 
                 'n01629819', 'n09332890', 'n02481823', 'n03902125', 'n03404251', 
                 'n09193705', 'n03637318', 'n04456115', 'n02666196', 'n03796401', 
                 'n02795169', 'n02123045', 'n01855672', 'n01882714', 'n02917067', 
                 'n02988304', 'n04398044', 'n02843684', 'n02423022', 'n02669723']
        task3 = ['n02823428', 'n02236044', 'n03393912', 'n07583066', 'n04074963',
                 'n01629819', 'n09332890', 'n02481823', 'n03902125', 'n03404251', 
                 'n09193705', 'n03637318', 'n04456115', 'n02666196', 'n03796401', 
                 'n02795169', 'n02123045', 'n01855672', 'n01882714', 'n02917067', 
                 'n02988304', 'n04398044', 'n02843684', 'n02423022', 'n02669723', 
                 'n04465501', 'n02165456', 'n03770439', 'n02099601', 'n04486054', 
                 'n02950826', 'n03814639', 'n04259630', 'n03424325', 'n02948072', 
                 'n03179701', 'n03400231', 'n02206856', 'n03160309', 'n01984695', 
                 'n03977966', 'n03584254', 'n04023962', 'n02814860', 'n01910747', 
                 'n04596742', 'n03992509', 'n04133789', 'n03937543', 'n02927161']
        task4 = ['n04465501', 'n02165456', 'n03770439', 'n02099601', 'n04486054', 
                 'n02950826', 'n03814639', 'n04259630', 'n03424325', 'n02948072', 
                 'n03179701', 'n03400231', 'n02206856', 'n03160309', 'n01984695', 
                 'n03977966', 'n03584254', 'n04023962', 'n02814860', 'n01910747', 
                 'n04596742', 'n03992509', 'n04133789', 'n03937543', 'n02927161', 
                 'n01945685', 'n02395406', 'n02125311', 'n03126707', 'n04532106', 
                 'n02268443', 'n02977058', 'n07734744', 'n03599486', 'n04562935', 
                 'n03014705', 'n04251144', 'n04356056', 'n02190166', 'n03670208', 
                 'n02002724', 'n02074367', 'n04285008', 'n04560804', 'n04366367', 
                 'n02403003', 'n07615774', 'n04501370', 'n03026506', 'n02906734']
        task5 = ['n01945685', 'n02395406', 'n02125311', 'n03126707', 'n04532106', 
                 'n02268443', 'n02977058', 'n07734744', 'n03599486', 'n04562935', 
                 'n03014705', 'n04251144', 'n04356056', 'n02190166', 'n03670208', 
                 'n02002724', 'n02074367', 'n04285008', 'n04560804', 'n04366367', 
                 'n02403003', 'n07615774', 'n04501370', 'n03026506', 'n02906734',
                 'n01770393', 'n04597913', 'n03930313', 'n04118538', 'n04179913', 
                 'n04311004', 'n02123394', 'n04070727', 'n02793495', 'n02730930', 
                 'n02094433', 'n04371430', 'n04328186', 'n03649909', 'n04417672', 
                 'n03388043', 'n01774384', 'n02837789', 'n07579787', 'n04399382', 
                 'n02791270', 'n03089624', 'n02814533', 'n04149813', 'n07747607']
        task6 = ['n03355925', 'n01983481', 'n04487081', 'n03250847', 'n03255030', 
                 'n02892201', 'n02883205', 'n03100240', 'n02415577', 'n02480495', 
                 'n01698640', 'n01784675', 'n04376876', 'n03444034', 'n01917289', 
                 'n01950731', 'n03042490', 'n07711569', 'n04532670', 'n03763968', 
                 'n07768694', 'n02999410', 'n03617480', 'n06596364', 'n01768244', 
                 'n02410509', 'n03976657', 'n01742172', 'n03980874', 'n02808440', 
                 'n02226429', 'n02231487', 'n02085620', 'n01644900', 'n02129165', 
                 'n02699494', 'n03837869', 'n02815834', 'n07720875', 'n02788148', 
                 'n02909870', 'n03706229', 'n07871810', 'n03447447', 'n02113799', 
                 'n12267677', 'n03662601', 'n02841315', 'n07715103', 'n02504458']
        
        task_list = {'task1': task1, 'task2': task2, 'task3': task3, 'task4': task4, 'task5': task5, 'task6': task6}

        tasks = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6']
        data_path = "./data/tiny-imagenet-200/"

    else:
        assert "task name not implemented"

    return task_list, tasks, data_path
