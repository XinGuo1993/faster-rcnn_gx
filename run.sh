python tools/train_rpn_net.py --solver models/RPN_Net/ZF/solver_60k80k.prototxt --testdef models/RPN_Net/ZF/test.prototxt --weights data/imagenet_models/ZF.caffemodel --cfg experiments/cfgs/mix_overlap_09.yml --imdb plate_gz_10w_trainval 
#>output/out.txt 2>output/err.txt
