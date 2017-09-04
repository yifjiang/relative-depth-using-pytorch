# relative-depth-using-pytorch
The pytorch implementation of the NIPS paper:

Single-Image Depth Perception in the Wild,
Neural Information Processing Systems (NIPS).

# Setup

1.Install pyTorch as described in http://pytorch.org.

2. Clone this repo.

        git clone https://github.com/yifjiang/relative-depth-using-pytorch.git

3. Download and extract the DIW dataset from the [project site](http://www-personal.umich.edu/~wfchen/depth-in-the-wild/). Download and extract `DIW_test.tar.gz` and `DIW_train_val.tar.gz` into 2 folders. Run the following command to download and extract `DIW_Annotations.tar.gz`. Then modify the filepath to images in `DIW_test.csv`, `DIW_train.csv` and `DIW_val.csv` to be the absolute file path where you extracted `DIW_test.tar.gz` and `DIW_train_val.tar.gz`. 

        cd relative_depth
        mkdir data
        cd data
        wget https://vl-lab.eecs.umich.edu/data/nips2016/DIW_Annotations_splitted.tar.gz
        tar -xzf DIW_Annotations_splitted.tar.gz
        rm DIW_Annotations_splitted.tar.gz


# Training and evaluating the networks

## Testing on pre-trained models 

Please first run the following commands to download the test data from our processed NYU dataset and the pre-trained models:

    cd relative_depth
    wget https://vl-lab.eecs.umich.edu/data/nips2016/data.tar.gz
    tar -xzf data.tar.gz
    rm data.tar.gz
    cd data
    python convert_csv_2_h5.py -i 750_train_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv
    python convert_csv_2_h5.py -i 45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv

    cd ../src
    mkdir results
    cd results
    wget https://vl-lab.eecs.umich.edu/data/nips2016/hourglass3.tar.gz
    tar -xzf hourglass3.tar.gz
    rm hourglass3.tar.gz

Then change directory into `/relative_depth/src/experiment`.

1. To evaluate the pre-trained model ***Ours***(model trained on the NYU labeled training subset) on the NYU dataset, run the following command:

        python test_model_on_NYU.py -num_iter 1000 -prev_model_file ../results/Best_model_period1.pt -mode test -crop 8

2. To evaluate the pre-trained model ***Ours_DIW***(our network trained from scratch on DIW) on the DIW dataset, run the following script:

        python test_model_on_DIW.py -num_iter 100 -prev_model_file ../results/Best_model_period1.pt -vis True

3. To test on a single image, we provide a handy script `test_on_one_image.lua`:

        python test_on_one_image.py -prev_model_file ../results/Best_model_period3.pt -input_image ../../data/singleImages/4.png -output_image ../../data/singleImages/4-out.png

## Training 

Please first change directory into `/relative_depth/src/experiment`.

To train the model ***Ours***(model trained on the NYU labeled training subset), please run the following command:

    python main.py -lr 0.001 -bs 4 -it 100000 -t_depth_file 750_train_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv -v_depth_file 45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv -rundir ./results
