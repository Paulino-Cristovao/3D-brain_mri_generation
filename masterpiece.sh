#
# Run all, eagle runner ...

#
LR=0.0002
BATCH=4

LATENT_LIST="32"
NUMBER_EPOCHS="1001"

for LATENT in $LATENT_LIST ; do
for EPOCHS in $NUMBER_EPOCHS ; do

#
#python3 2D_VAEGAN.py --features $LATENT --lr $LR --num_epochs $EPOCHS #> logs.txt
#python3 Multimodal_2DVAEGAN.py --features $LATENT --lr $LR --num_epochs $EPOCHS #> logs.txt
#python3 2D_DCGAN.py --features $LATENT --lr $LR --num_epochs $EPOCHS #> logs.txt


#python3 Multimodal_3D_VAEGAN.py --features $LATENT --lr $LR --batch_size $BATCH --num_epochs $EPOCHS #> logs.txt
#python3 3D_VAEGAN.py --features $LATENT --lr $LR --batch_size $BATCH --num_epochs $EPOCHS #> logs.txt
#python3 3D_DCGAN.py --features $LATENT --lr $LR --batch_size $BATCH --num_epochs $EPOCHS #> logs.txt


done;
done;

