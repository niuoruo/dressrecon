seqname=$1
logname=$2

python lab4d/diffgs/train.py --seqname $seqname --logname diffgs-$logname --pixels_per_image -1 --imgs_per_gpu 16 --eval_res 256 --learning_rate 5e-3 --fg_motion bob --use_init_cam --lab4d_path logdir/${seqname}-${logname}/opts.log --rgb_wt 0.3 --feature_wt 0 ${@:3}
