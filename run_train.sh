python train.py --train_dataset 300W-LP --val_dataset 300W-LP --rot_type rot_mat --do_smooth\
                --num_pts 600 --soft_loss_weight 5.0 --shape_regular_weight 0.0 --ortho_loss_weight 0.0 \
                --device cuda:0 --prefix training_runs/aflw2000_14 \
                --img_size 224 --learning_rate 1e-4