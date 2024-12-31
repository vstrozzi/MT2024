# Explaining CLIP's Visual Encoders Heads using Text-Embeddings on Principal Components
Pythorch implementation, adapted and extended from https://github.com/yossigandelsman/clip_text_span.


![Teaser](images/teaser.png)

## General
Information on how to run and play with our approach.
### Setup dependencies
Use the provided [`environment.yml`](environment.yml) file to create a Conda environment with all the dependenctios:

```bash
conda env create -f environment.yml
conda activate MT
```
#### Download Dataset(s)
Please download the Imagenet dataset from [here](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat):

```bash
cd datasets
mkdir imagenet
cd imagenet
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

To download the Waterbirds datasets, run:
```bash
cd datasets
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xf  waterbird_complete95_forest2water2.tar.gz
```

### Test
First run the Notebook prepare_data.ipynb to setup all the necessary data for the experiments.
Then use the Notebook playground.ipynb to play around with the different approaches.

## Specific
Explanations on the individual functions used on the Notebooks under General.

### Preprocessing
To obtain the projected residual stream components for the ImageNet validation set, including the contributions from multi-head attentions and MLPs, please run one of the following instructions:

```bash
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k --data_path <PATH>
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path <PATH>
python -m utils.scripts.compute_activation_values --dataset imagenet --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k --data_path <PATH>
```

To obtain the precomputed text representations of the ImageNet classes, please run:
```bash
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k
python -m utils.scripts.compute_classes_embeddings  --dataset imagenet --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k
```


### Convert text labels to representation 
To convert the text labels to CLIP text representations, please run:

```bash
python -m utils.scripts.compute_text_explanations --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path utils/text_descriptions/google_3498_english.txt
python -m utils.scripts.compute_text_explanations --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path utils/text_descriptions/top_1500_nouns_5_sentences_imagenet_bias_clean.txt
```

### ImageNet segmentation
To get the evaluation results, please run:

```bash
python compute_segmentations.py --device cuda:0 --model ViT-H-14 --pretrained laion2b_s32b_b79k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
python compute_segmentations.py --device cuda:0 --model ViT-L-14 --pretrained laion2b_s32b_b82k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
python compute_segmentations.py --device cuda:0 --model ViT-B-16 --pretrained laion2b_s34b_b88k --data_path imagenet_seg/gtsegs_ijcv.mat --save_img
```
Save the results with the `--save_img` flag.

### Spectral Decomposition

Explain the internal components of the CLIP-embeddings ViT-Encoder's for images using with text (ours: svd_data_approx, their: text_span)
```bash
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-H-14 --algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-L-14 --algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
!python -m utils.scripts.compute_text_explanations --device cpu --model ViT-B-16--algorithm svd_data_approx --seed 12 --num_of_last_layers 4 --text_descriptions top_1500_nouns_5_sentences_imagenet_bias_clean
```

### Spatial decomposition
Please see a demo for the spatial decomposition of CLIP in `demo.ipynb`. 


### Nearest neighbors search
Please see the nearest neighbors search demo in `nns.ipynb`.
