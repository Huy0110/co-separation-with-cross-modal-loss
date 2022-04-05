CUDA_VISIBLE_DEVICES=0,1,2 python3 train.py \
  --name audioVisual \
  --hdf5_path /home/manhnguyen/new/co-separation/dataset/sample_hdf5 \
  --batchSize 8 \
  --nThreads 32 \
  --display_freq 10  \
  --save_latest_freq 500 \
  --niter 2 \
  --validation_freq 200  \
  --validation_batches 20 \
  --num_batch 500000 \
  --lr_steps 15000000 \
  --checkpoints_dir checkpoint_new_huy_44 \
  --classifier_loss_weight 0.05 \
  --coseparation_loss_weight 1 \
  --crossmodal_loss_weight 0.05 \
  --unet_num_layers 7 \
  --lr_visual 0.00001 \
  --lr_unet 0.0001 \
  --lr_classifier 0.0001 \
  --lr_vocal_attributes 0.00001 \
  --lr_facial_attributes 0.00001 \
  --triplet_loss_type triplet \
  --weighted_loss \
  --visual_pool conv1x1 \
  --audio_pool conv1x1 \
  --optimizer adam \
  --log_freq True \
  --tensorboard True \
  --continue_train \
  --identity_feature_dim 512 \
  --margin 0.5 \
  --validation_visualization True \
  --enable_data_augmentation True \
  --gpu_ids 2
  #--weights_visual /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_37/audioVisual/visual_best.pth \
  #--weights_unet /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_37/audioVisual/unet_best.pth \
  #--weights_classifier /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_37/audioVisual/classifier_best.pth  \
  #--weights_facial /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_37/audioVisual/facial_best.pth \
  #--weights_vocal /home/manhnguyen/new/co-separation_copy_huy_copy/checkpoint_new_huy_37/audioVisual/vocal_best.pth
  # |& tee -a log_huy_chay_test.txt 
  #  --weights_visual /home/manhnguyen/new/co-separation/checkpoint_for_train/audioVisual/visual_best.pth \
  # --weights_unet /home/manhnguyen/new/co-separation/checkpoint_for_train/audioVisual/unet_best.pth \
  # --weights_classifier /home/manhnguyen/new/co-separation/checkpoint_for_train/audioVisual/classifier_best.pth  \
  # --gpu_ids 2,3 \
  #  --weights_facial facial.pth \
  #--weights_vocal vocal.pth \
 
  # --with_additional_scene_image \
  #weight for continue train
  