INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='/home/ubuntu/infer_models/faster_rcnn_resnet101_sdc.config'
TRAINED_CKPT_PREFIX='/home/ubuntu/infer_models/model.ckpt-69443'
EXPORT_DIR='/home/ubuntu/infer_models/frozen'
python object_detection/export_inference_graph.py \
	--input_type=${INPUT_TYPE} \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
	--output_directory=${EXPORT_DIR}
