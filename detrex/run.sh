#/bin/bash

# BACKBONE="r50_4scale"
BACKBONE="vitdet_base_4scale"
# BACKBONE="swin_base_384_4scale"
# BACKBONE="internimage_base_4scale"
# BACKBONE="dit_base"

# METHOD="DRGG"
# METHOD="DRGGBBoxEmb"
# METHOD="DRGGTextEmb"
# METHOD="DRGGBBoxEmbTextEmb"
# METHOD="DRGGTFEnc"
METHOD="DRGGBBoxEmbTFEnc"
# METHOD="DRGGTextEmbTFEnc"
# METHOD="DRGGBBoxEmbTextEmbTFEnc"

if [ "$BACKBONE" = "r50_4scale" ]; then
    CONFIG_FILE=projects/relation/configs/gtbbox_r50_4scale_24ep_scipostlayouttree.py
elif [ "$BACKBONE" = "vitdet_base_4scale" ]; then
    CONFIG_FILE=projects/relation/configs/gtbbox_vitdet_base_4scale_24ep_scipostlayouttree.py
elif [ "$BACKBONE" = "swin_base_384_4scale" ]; then
    CONFIG_FILE=projects/relation/configs/gtbbox_swin_base_384_4scale_24ep_scipostlayouttree.py
elif [ "$BACKBONE" = "internimage_base_4scale" ]; then
    CONFIG_FILE=projects/relation/configs/gtbbox_internimage_base_4scale_24ep_scipostlayouttree.py
elif [ "$BACKBONE" = "dit_base" ]; then
    CONFIG_FILE=projects/relation/configs/gtbbox_dit_base_24ep_scipostlayouttree.py
else
    echo "Unknown BACKBONE: $BACKBONE"
    exit 1
fi

USE_BBOX_EMB=False
USE_TEXT_EMB=False
USE_TF_ENC=False

case "$METHOD" in
    DRGG)
    ;;
    DRGGBBoxEmb)
    USE_BBOX_EMB=True
    ;;
    DRGGTextEmb)
    USE_TEXT_EMB=True
    ;;
    DRGGBBoxEmbTextEmb)
    USE_BBOX_EMB=True
    USE_TEXT_EMB=True
    ;;
    DRGGTFEnc)
    USE_TF_ENC=True
    ;;
    DRGGBBoxEmbTFEnc)
    USE_BBOX_EMB=True
    USE_TF_ENC=True
    ;;
    DRGGTextEmbTFEnc)
    USE_TEXT_EMB=True
    USE_TF_ENC=True
    ;;
    DRGGBBoxEmbTextEmbTFEnc)
    USE_BBOX_EMB=True
    USE_TEXT_EMB=True
    USE_TF_ENC=True
    ;;
    *)
    echo "Unknown METHOD: $METHOD"
    exit 1
    ;;
esac

OUTPUT_DIR=output/gtbbox_${BACKBONE}_${METHOD}

echo "==============================="
echo "METHOD: $METHOD"
echo "DETECTOR: GTBBox"
echo "BACKBONE: $BACKBONE"
echo "==============================="

python tools/train_net.py --config-file $CONFIG_FILE --num-gpus 8 train.output_dir=$OUTPUT_DIR model.use_bbox_emb=$USE_BBOX_EMB model.use_text_emb=$USE_TEXT_EMB model.use_tf_enc=$USE_TF_ENC

rm -rf $OUTPUT_DIR/model_0*.pth

BEAM_WIDTH=20

python tools/train_net.py --config-file $CONFIG_FILE --num-gpus 8 --eval-only train.output_dir=$OUTPUT_DIR model.use_bbox_emb=$USE_BBOX_EMB model.use_text_emb=$USE_TEXT_EMB model.use_tf_enc=$USE_TF_ENC train.init_checkpoint=$OUTPUT_DIR/model_final.pth model.beam_width=$BEAM_WIDTH

# python tools/visualize_json_results.py --input $OUTPUT_DIR/coco_instances_results.json --output $OUTPUT_DIR/images --dataset scipostlayout_val
