#prerequisites
# fetch_atl06 already run on the start-date end-date interval
#python fetch_atl06.py --t0 2019-01-01T00:00:00Z --t1 2019-01-07T23:59:59Z
# get_lis_dem already run on the start-date end-date interval
#python get_lis_dem.py --lis-path s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc

LIS_PATH=s3://airborne-smce-prod-user-bucket/JOIN/lis_input_NMP_1000m_missouri.nc
#AMSR2_DIR=s3://airborne-smce-prod-user-bucket/JOIN/AMSR2/
#CEDA_DIR=s3://airborne-smce-prod-user-bucket/JOIN/CEDA/
VIIRS_DIR=s3://airborne-smce-prod-user-bucket/JOIN/VIIRS/
ICESAT2_PARQUET=_data/icesat2/
WEIGHTS_DIR=_data/weights/

START_DATE=2019/01/01
END_DATE=2019/01/02

python3 full-workflow_virtual.py --start-date $START_DATE --end-date $END_DATE --lis-path $LIS_PATH --viirs-dir $VIIRS_DIR \
			       	--icesat2-parquet $ICESAT2_PARQUET --weights-dir $WEIGHTS_DIR
