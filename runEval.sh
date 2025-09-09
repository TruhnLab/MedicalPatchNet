echo "Eval MedicalPatchNet_weights.pt"

python3 trainClassification.py \
    -patchSize 64 \
    -imgSize 512 \
    -wb \
    -wb_name MedicalPatchNetEvalOnly \
    -evalOnly \
    -loadPath "savedModels/MedicalPatchNet_weights.pt" \


echo " "
echo " "
echo " "
echo " "
echo " "

echo "Eval EfficientNetB0_weights.pt"

python3 trainClassification.py \
    -patchSize 512 \
    -imgSize 512 \
    -wb \
    -wb_name MedicalPatchNetEvalOnly \
    -evalOnly \
    -loadPath "savedModels/EfficientNetB0_weights.pt" \
