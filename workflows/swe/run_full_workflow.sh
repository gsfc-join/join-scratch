LIS_PATH=s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc
AMSR2_DIR=s3://airborne-smce-prod-user-bucket/JOIN/AMSR2/
CEDA_DIR=s3://airborne-smce-prod-user-bucket/JOIN/CEDA/
VIIRS_DIR=s3://airborne-smce-prod-user-bucket/JOIN/VIIRS/
ICESAT2_PARQUET=_data/icesat2/
WEIGHTS_DIR=_data/weights/

python3 full-workflow.py --lis-path $LIS_PATH --amsr2-dir $AMSR2_DIR --ceda-dir $CEDA_DIR --viirs-dir $VIIRS_DIR --icesat2-parquet $ICESAT2_PARQUET \
                        --weights-dir $WEIGHTS_DIR 
