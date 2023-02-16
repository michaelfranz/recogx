#!/bin/zsh
# Copy mfc images 1-4999 for both males and females to training folder
cp mfc_dataset/female_?.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_??.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_???.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_1???.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_2???.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_3???.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/female_4???.jpg mfc_dataset_train_test/train/female
cp mfc_dataset/male_?.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_??.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_???.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_1???.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_2???.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_3???.jpg mfc_dataset_train_test/train/male
cp mfc_dataset/male_4???.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_6???.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_7???.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_8???.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_9???.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_1????.jpg mfc_dataset_train_test/train/male
# cp mfc_dataset/male_2????.jpg mfc_dataset_train_test/train/male

# Copy mfc images 5000-5999 for both males and females to test folder
cp mfc_dataset/female_5???.jpg mfc_dataset_train_test/test/female
cp mfc_dataset/male_5???.jpg mfc_dataset_train_test/test/male
