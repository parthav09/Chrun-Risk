#!/bin/bash

for i in {1..1}; 
do 
    python clean.py
    echo "Cleaning done..."
    python create_folds.py
    echo "Folds created..."
    echo ""
    echo "Training XGBClassifier..."
    python -W ignore train.py --model "xgboost"
    # echo "Generating submission..."
    # python predict.py
    echo "Training CatBoostClassifier..."
    python train.py --model "catboost"
done 

echo "Press any key to continue"
while [ true ] ; do
read -t 3 -n 1
if [ $? = 0 ] ; then
exit ;
else
echo "waiting for the keypress"
fi
done