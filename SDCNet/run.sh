#!/bin/bash
source deactivate
source activate torch_py3.6

#corpora=(news)
#algorithms=(tina)

#for bpw in {3..4} 
#	do
#	for corpous in ${corpora[*]}
#		do
#		for al in ${algorithms[*]}
#			do
#			python main.py -save-dir snapshot1/${al}/${corpous}_${bpw}bpw/ -train-cover-dir ../../data/${al}/${corpous}/train_data/train_cover.txt -train-stego-dir ../../data/${al}/${corpous}/train_data/${corpous}_${bpw}bpw.txt -epoch 20 

#			python main.py -save-dir snapshot1/${al}/${corpous}_${bpw}bpw/ -train-cover-dir ../../data/${al}/${corpous}/train_data/train_cover.txt -train-stego-dir ../../data/${al}/${corpous}/train_data/${corpous}_${bpw}bpw.txt  -test-cover-dir ../../data/${al}/${corpous}/test_data/test_cover.txt -test-stego-dir ../../data/${al}/${corpous}/test_data/${corpous}_${bpw}bpw.txt -test True
#			done
#		done
#	done

python main.py -save-dir snapshot/SDC/coco/BLOCK-2bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_2bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_2bpw.txt
python main.py -save-dir snapshot/SDC/coco/BLOCK-2bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_2bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_2bpw.txt -test True
python main.py -save-dir snapshot/SDC/coco/BLOCK-4bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_4bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_4bpw.txt
python main.py -save-dir snapshot/SDC/coco/BLOCK-4bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_4bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_4bpw.txt -test True

python main.py -save-dir snapshot/SDC/movie/BLOCK-2bpw-ATT -train-cover-dir ../../data/BLOCK/movie/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/movie/train_data/movie_1bpw.txt -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/movie_2bpw.txt
python main.py -save-dir snapshot/SDC/movie/BLOCK-2bpw-ATT -train-cover-dir ../../data/BLOCK/movie/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/movie/train_data/movie_1bpw.txt -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/movie_2bpw.txt -test True
python main.py -save-dir snapshot/SDC/movie/BLOCK-4bpw-ATT -train-cover-dir ../../data/BLOCK/movie/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/movie/train_data/movie_3bpw.txt -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/movie_4bpw.txt
python main.py -save-dir snapshot/SDC/movie/BLOCK-4bpw-ATT -train-cover-dir ../../data/BLOCK/movie/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/movie/train_data/movie_3bpw.txt -test-cover-dir ../../data/BLOCK/movie/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/movie/test_data/movie_4bpw.txt -test True

#python main.py -save-dir snapshot/SDC/coco/BLOCK-1bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_1bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_1bpw.txt
#python main.py -save-dir snapshot/SDC/coco/BLOCK-1bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_1bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_1bpw.txt -test True
#python main.py -save-dir snapshot/SDC/coco/BLOCK-3bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_3bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_3bpw.txt
#python main.py -save-dir snapshot/SDC/coco/BLOCK-3bpw-ATT -train-cover-dir ../../data/BLOCK/coco/train_data/train_cover.txt -train-stego-dir ../../data/BLOCK/coco/train_data/coco_3bpw.txt -test-cover-dir ../../data/BLOCK/coco/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/coco/test_data/coco_3bpw.txt -test True



