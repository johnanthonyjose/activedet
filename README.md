# Active Learning in Object Detection

This repository explores on how to perform active learning on Object Detection

## Requirements
- Anaconda/Miniconda 
- CUDA Toolkit 10.2
- NVIDIA GPU 

## Installations

1. Install Anaconda

2. Clone the repository.
```bash
$ git clone https://github.com/johnanthonyjose/activedet.git
```

3. Navigate to the project directory `activedet`

4. Create your conda environment. For example, `activedet`
```bash
$ conda create -n activedet python=3.7
```

5. Update your environment using the provided yaml file:
```bash
$ conda env update --name activedet --file environment.yml
$ conda activate activedet
```

6. Install the project using pip (editable). Adding -e makes it compatible for actively developed project. You should be able to see it installed
```bash
(activedet) $ pip install -e .
```

## Getting Started: Train RetinaNet using Active Learning Workflow with Random Acquisition Function
Detectron2 creates a highly configurable toolbox for experimentation with the help of a config file.

This file describes the different components you want to knit together.
For this intro, we would use a random acquistion function
We'll be training RetinaNet on PASCAL-VOC dataset.

1. Make sure to be in the project directory `activedet`

2. Follow the [Detectron2 dataset structure for Pascal-VOC dataset](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html#expected-dataset-structure-for-pascal-voc). Let $VOC2012 be the full path of VOC2012 directory. Let $VOC2007 be the full path of VOC2007 directory. 

3. Symlink the PASCAL_VOC into the datasets directory
```bash
$ ln -s $VOC2012 datasets/VOC2012
$ ln -s $VOC2007 datasets/VOC2007
```

4. Train RetinaNet using an active learning workflow with random acquisition function

```bash
(activedet) $ python tools/active_lazy_train_net.py --num-gpus 1 --config-file configs/PascalVOC/random_torch_retinanet.py 
```
There are other acquisition functions available that was used in the paper such as:
- Entropy
- CoreSet
- Learning Loss

## Reproduce Active Learning Workflow with Confidence-Binned Entropy as acquisition function

### Pascal-VOC
1. Similar to previous section, follow the [Detectron2 dataset structure for Pascal-VOC dataset](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html#expected-dataset-structure-for-pascal-voc).

2. Train on 5 different seeds using the ff:  
```bash
(activedet) $ ./run_experiments.sh configs/PascalVOC/ConfBinEnt_retinanet.py output/PascalVOC/ConfBinEnt_retinanet
```

3. The result is written on log.txt in the `output/` directory

### COCO

1. follow the [Detectron2 dataset structure for COCO dataset](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html#expected-dataset-structure-for-coco-instance-keypoint-detection)

2. Train on 5 different seeds using the ff:  
```bash
(activedet) $ ./run_experiments.sh configs/COCO/ConfBinEnt_retinanet.py output/COCO/ConfBinEnt_retinanet
```

3. The result is written on log.txt in the `output/` directory
