#!/bin/bash

if [[ $MODEL_TYPE == "cnn" ]]; then
      echo "##### $MODEL_TYPE models will be used. #####"
      echo "##### Copying onnx-files to DS docker volume for perserving PVCs... #####"
      cp /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/models/*.onnx /opt/storage/
      cp /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/models/*.etlt /opt/storage/
      ./deepstream-fewshot-learning-app -c configs/mtmc_config.txt -m 1 -t 0 -l 5 --message-rate 1 --tracker-reid 1 --reid-store-age 1

elif  [[ $MODEL_TYPE == "transformer" ]]; then
         echo "##### $MODEL_TYPE models will be used. #####"
         echo "##### Copying onnx-files to DS docker volume for perserving PVCs... #####"
         cp /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-fewshot-learning-app/models/*.onnx /opt/storage/   
         cd srcs
         CUDA_VER=12.8 make   
         mv ./deepstream-fewshot-learning-app ..
         cd .. 
         ./deepstream-fewshot-learning-app -c ./configs/mtmc_config.txt -m 1 -t 1 -l 5 --message-rate 1 --tracker-reid 1 --reid-store-age 1
else
    echo "##### Invalid value $MODEL_TYPE for MODEL_TYPE variable. Valid values are: 'cnn' or 'transformer'. #####"
fi;