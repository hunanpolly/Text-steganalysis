
#---------------------------------------------BLOCK------------------------------------------

#n+t-t
python main_adam.py -save-dir snapshot30/BLOCK/news2twitter/ -source1-cover-dir ../../data/BLOCK/news/train_data/train_cover.txt -source1-stego-dir ../../data/BLOCK/news/train_data/stego_3words_bit.txt  -target-cover-dir ../../data/BLOCK/twitter/train_data/train_cover.txt -target-stego-dir ../../data/BLOCK/twitter/train_data/stego_3words_bit.txt -test-cover-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -epoch 100 -log-interval 100 -test-interval 500  
#python main_adam.py -save-dir snapshot30/BLOCK/news2twitter/ -source1-cover-dir ../../data/BLOCK/news/train_data/train_cover.txt -source1-stego-dir ../../data/BLOCK/news/train_data/stego_3words_bit.txt  -target-cover-dir ../../data/BLOCK/twitter/train_data/train_cover.txt -target-stego-dir ../../data/BLOCK/twitter/train_data/stego_3words_bit.txt -test-cover-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -epoch 100 -log-interval 100 -test-interval 500   -test True



#python main_adam.py -save-dir snapshot30/BLOCK/news2twitter/  -source1-cover-dir ../../data/BLOCK/news/train_data/train_cover.txt -source1-stego-dir ../../data/BLOCK/news/train_data/stego_3words_bit.txt  -target-cover-dir ../../data/BLOCK/twitter/train_data/train_cover.txt -target-stego-dir ../../data/BLOCK/twitter/train_data/stego_3words_bit.txt  -test-cover-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/test_cover.txt -epoch 100 -log-interval 100 -test-interval 500   -test True
#python main_adam.py -save-dir snapshot30/BLOCK/news2twitter/  -source1-cover-dir ../../data/BLOCK/news/train_data/train_cover.txt -source1-stego-dir ../../data/BLOCK/news/train_data/stego_3words_bit.txt  -target-cover-dir ../../data/BLOCK/twitter/train_data/train_cover.txt -target-stego-dir ../../data/BLOCK/twitter/train_data/stego_3words_bit.txt  -test-cover-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -test-stego-dir ../../data/BLOCK/twitter/test_data/stego_3words_bit.txt -epoch 100 -log-interval 100 -test-interval 500   -test True