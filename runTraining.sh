#activate your conda environment beforehand

##################
#
# Medical patch net with 8x8 patches
# -patchSize 64
# -imgSize 512 
#
# EfficientNet-B0 => image is one patch so the backbone gets the whole image and mean is computed over one vector => same as EfficentNet-B0
# -patchSize 512
# -imgSize 512 
#
##################

#Train MedicalPatchNet
python3 trainClassification.py \
    -lr 1e-4 \
    -bs 16 \
    -ep 20 \
    -wb \
    -wb_name MedicalPatchNet \
    -patchSize 64 \
    -imgSize 512 \
    -savePath MedicalPatchNet.pt \

echo "DONE TRAINING MedicalPatchNet"

#Train EfficientNet-B0
python3 trainClassification.py \
    -lr 1e-4 \
    -bs 16 \
    -ep 20 \
    -wb \
    -wb_name EfficientNetB0 \
    -patchSize 512 \
    -imgSize 512 \
    -savePath EfficientNetB0.pt \

echo "DONE TRAINING EfficientNet-B0"
EfficientNetB0_weights.pt