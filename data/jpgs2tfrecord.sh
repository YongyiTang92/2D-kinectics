export CUDA_VISIBLE_DEVICES='7'
export HADOOP_OPTS="-XX:-UseGCOverheadLimit -Xmx16384m"
export HADOOP_CLIENT_OPTS="-XX:-UseGCOverheadLimit -Xmx16384m"

python3.6 jpgs2tfrecord.py \
    --hdfs_dir='hdfs://100.77.5.27:9000/user/VideoAI/rextang/kinetics-400' \
    --json_path='/data1/rextang/kinetics_400/v1-0/train.csv' \
    --video_source='/data1/rextang/kinetics_400/v1-0/train' \
    --destination='./output_tmp/videos' \
    --jpg_path='./images_tmp' \
    --FPS=24 \
    --batch_start=0 \
    --batch_end=10 \
    $@