# rice-image-classification
Different types of Rice, image classification

Dataset: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

5 categories: Arborio, Basmati, Ipsala, Jasmine and Karacadag rice varieties.
Total dataset contains: 15k images per class (75k images in total).

__Methodology (resnet_classifier.ipynb)__:
1. Create a basic CustomDataset class  which loads  the images from the folder. The class also takes in an optional transformer function which allows to modify images on the fly.
2. Create a torchvision pipeline which resizes the images, converts them to a tensor and normalizes the 3 channels of the RGB image. Given the size (and distribution) of our dataset, no major data augmentation is required.
3. Initialize the dataloader with a batch-size & parallelize using multiple worker threads
4. Use scikit learn's train_test_split to train, test & validation sets. Ensure that you are splitting the data by using stratification to create properly balanced dataset.
5. Load torchvision's pre-trained models (resnet18, in this case).
6. In the model, the last layer is replaced with a new one corresponding to a vector of size 5 (one-hot encoded vector associated  with the various categories of classification)
7. The loss criterion is defined as the Cross Entropy Loss.
8. Run the training loop for 10 epochs with each batch consisting  of 128 images. At the end of each loop, check the classification performance against the validation data set.
9. Evaluate performance against the test dataset. In this particular scenario, only 15 of the 15,000 test images had an incorrect label prediction.

