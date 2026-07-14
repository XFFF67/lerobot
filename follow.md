conda activate lerobot310
cd lerobot
# 开启 CAN 接口
<!-- sudo ip link set can0 down 
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 down 
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up

cansend can0 008#FFFFFFFFFFFFFFFC
candump can0 -->

# 设置 CAN 接口
# 把该 CAN 接口的 发送队列长度 设置为 1000
for can_port in can0 can1 can2 can3; do
    sudo ip link set $can_port txqueuelen 1000
done

lerobot-setup-can --mode=setup --interfaces=can0,can1 --use_fd=false
lerobot-setup-can --mode=setup --interfaces=can2,can3 --use_fd=true

# 测试电机通信
lerobot-setup-can --mode=test --interfaces=can0,can1 --use_fd=false
lerobot-setup-can --mode=test --interfaces=can2,can3 --use_fd=true

# 主臂标定
<!-- lerobot-calibrate --config_path ~/lerobot/openarm_partial_leader.yaml


lerobot-calibrate \
    --teleop.type=openarm_leader \
    --teleop.port=can3 \
    --teleop.id=my_openarm_leader
    
    
lerobot-calibrate \
    --teleop.type=hyopenarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_hyopenarm_leader -->

lerobot-calibrate \
--teleop.type=bi_hyopenarm_leader \
--teleop.left_arm_config.port=can1 \
--teleop.right_arm_config.port=can0 \
--teleop.id=my_bimanual_hyleader

# 跟随臂标定
<!-- lerobot-calibrate --config_path ~/lerobot/openarm_partial_follower.yaml


lerobot-calibrate \
    --robot.type=openarm_follower \
    --robot.port=can2 \
    --robot.side=right \
    --robot.id=my_openarm_follower
    
    
lerobot-calibrate \
    --robot.type=hyopenarm_follower \
    --robot.port=can0 \
    --robot.side=right \
    --robot.id=my_hyopenarm_follower


lerobot-calibrate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower  -->

lerobot-calibrate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.use_can_fd=true \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.side=right \
    --robot.right_arm_config.use_can_fd=true \
    --robot.id=my_bimanual_follower

# 跟随模式
<!-- lerobot-teleoperate --config_path ~/lerobot/openarm_partial_teleop.yaml


lerobot-teleoperate \
    --robot.type=openarm_follower \
    --robot.port=can0 \
    --robot.side=right \
    --robot.id=my_openarm_follower \
    --teleop.type=openarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_openarm_leader


lerobot-teleoperate \   .port=can0 \
    --robot.side=right \
    --robot.id=my_hyopenarm_follower \
    --teleop.type=hyopenarm_leader \
    --teleop.port=can1 \
    --teleop.id=my_hyopenarm_leader -->
    
lerobot-teleoperate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --teleop.type=bi_hyopenarm_leader \
    --teleop.left_arm_config.port=can1 \
    --teleop.right_arm_config.port=can0 \
    --teleop.id=my_bimanual_hyleader

# 遥操作 显示摄像头画面

lerobot-teleoperate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.left_arm_config.cameras='{
        wrist_left: {"type": "intelrealsense", "serial_number_or_name": "260322272305", "width": 640, "height": 480, "fps": 30},
        top: {"type": "intelrealsense", "serial_number_or_name": "141322252011", "width": 640, "height": 480, "fps": 30},
        }' \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.cameras='{
        wrist_right: {"type": "intelrealsense", "serial_number_or_name": "352122273643", "width": 640, "height": 480, "fps": 30},
        }' \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --teleop.type=bi_hyopenarm_leader \
    --teleop.left_arm_config.port=can1 \
    --teleop.right_arm_config.port=can0 \
    --teleop.id=my_bimanual_hyleader
    --display_data=true


lerobot-teleoperate \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --teleop.type=bi_hyopenarm_leader \
    --teleop.left_arm_config.port=can1 \
    --teleop.right_arm_config.port=can0 \
    --teleop.id=my_bimanual_hyleader
# hugging face
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
pip install "huggingface_hub[cli]"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
hf auth login  #输入token
hf auth whoami #查看user
export HF_USER=" "

# 采集数据集
lerobot-record \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can3 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.right_arm_config.port=can2 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.id=my_bimanual_follower \
    --robot.left_arm_config.cameras='{wrist: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 60}}' \
    --robot.right_arm_config.cameras='{wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 60}}' \
    --robot.cameras='{top: {"type": "realsense", "width": 640, "height": 480, "fps": 90}}' \
    --dataset.repo_id=${HF_USER}/bi_openarm_test \  #token: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
    --dataset.single_task="  " \ #
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=60 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=true

lerobot-record \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --robot.left_arm_config.cameras='{
        wrist_left: {"type": "intelrealsense", "serial_number_or_name": "260322272305", "width": 640, "height": 480, "fps": 30},
        top: {"type": "intelrealsense", "serial_number_or_name": "141322252011", "width": 640, "height": 480, "fps": 30},
        }' \
    --robot.right_arm_config.cameras='{
        wrist_right: {"type": "intelrealsense", "serial_number_or_name": "352122273643", "width": 640, "height": 480, "fps": 30},
        }' \
    --teleop.type=bi_hyopenarm_leader \
    --teleop.left_arm_config.port=can1 \
    --teleop.right_arm_config.port=can0 \
    --teleop.id=my_bimanual_hyleader \
    --dataset.repo_id=xfff67/bi_openarm_test \
    --dataset.root=/home/mizu/lerobot/data/bi_openarm_test \
    --dataset.push_to_hub=false \
    --dataset.single_task="Left gripper picks the light blue stapler into the open black box, then right gripper places the blue duck into the box." \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=60 \
    --dataset.streaming_encoding=false \
    --dataset.vcodec=h264 \
    --dataset.num_image_writer_processes=1 \
    --dataset.num_image_writer_threads_per_camera=4 \
    --resume=true \
    --display_data=false 



top: {"type": "intelrealsense", "serial_number_or_name": "141322252011", "width": 640, "height": 480, "fps": 30},
wrist_left: {"type": "intelrealsense", "serial_number_or_name": "352122273643", "width": 640, "height": 480, "fps": 30},
wrist_left: {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30}, 

# 右臂d405 id：352122273643
# 左壁d405 id：260322272305
 
 hf upload limboyo/lerobot_openarm /home/mizu/lerobot/data/bi_openarm_test / --repo-type=dataset

lerobot-replay \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --dataset.root=/home/mizu/lerobot/data/bi_openarm_test \
    --dataset.repo_id=xfff67/bi_openarm_test \
    --dataset.episode=2

lerobot-train \
    --dataset.repo_id=limboyo/lerobot_openarm \
    --dataset.root=/home/mizu/lerobot/data/bi_openarm_test \
    --policy.type=pi05 \
    --output_dir=./output_lerobot_train/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=Lerobot_limboyo_Project \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=50000 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=32
# 服务器
lerobot-train \
    --dataset.repo_id=limboyo/lerobot_openarm \
    --dataset.root=/home/yanglinbo/lerobot_limbo/data/bi_openarm_test \
    --dataset.revision=v3.0 \
    --policy.type=pi05 \
    --output_dir=~/output_lerobot_train/shake/pi05_A \
    --job_name=shake_pi05_A \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.project=Lerobot_limboyo_Project \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=50000 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=32

# 往服务器上传数据
ssh -p 43222 yanglinbo@dl.neutdt.cn "mkdir -p /home/yanglinbo/lerobot_limbo/data/bi_openarm_test"

rsync -avh --progress -e "ssh -p 43222" /home/mizu/lerobot/data/bi_openarm_test/ yanglinbo@dl.neutdt.cn:/home/yanglinbo/lerobot_limbo/data/bi_openarm_test/

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
    --dataset.repo_id=limboyo/lerobot_openarm \
    --dataset.root=/home/mizu/lerobot/data/bi_openarm_test \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --output_dir=./output_lerobot_train/pi05_training_test \
    --job_name=pi05_training_test \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.max_action_dim=48 \
    --policy.max_state_dim=48 \
    --steps=1 \
    --save_freq=1 \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --batch_size=1

    进入页面后按这个顺序操作：

  1. 先确认右上角已经登录 Hugging Face 账号
     如果右上角显示 Log In，先点进去登录。

  2. 在模型页面往下找这一段：

  Access PaliGemma on Hugging Face

  页面会提示你需要接受 Google 的使用许可。

  3. 点击页面里的访问/同意按钮
     通常会是类似这些按钮文字：

  Agree and access repository

  或：

  Review conditions and access this model

  4. 勾选/填写页面要求的条款
     一般是确认你同意 Gemma / PaliGemma 的使用条款。按页面要求完成即可。

  5. 提交后等待页面显示你已经有权限
     这个模型页面说明里写的是请求会立即处理。完成后你应该能看到 Files and
     versions 里的文件，不再被挡住。

  然后回到终端重新测试：

  python -c "from transformers import AutoTokenizer;
  AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print('ok')"

  如果还是 403，通常是这两个原因：

  - 浏览器登录的 Hugging Face 账号，和你终端里 hf auth login 的 token 所属账号不
    是同一个。

  - 你接受的是别的 PaliGemma 模型页面，不是这个：
    google/paligemma-3b-pt-224

  可以用这个命令确认终端 token 对应的账号：

  hf auth whoami

# 本地推理

lerobot-rollout \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --robot.left_arm_config.cameras='{
        wrist_left: {"type": "intelrealsense", "serial_number_or_name": "260322272305", "width": 640, "height": 480, "fps": 30},
        top: {"type": "intelrealsense", "serial_number_or_name": "141322252011", "width": 640, "height": 480, "fps": 30},
        }' \
    --robot.right_arm_config.cameras='{
        wrist_right: {"type": "intelrealsense", "serial_number_or_name": "352122273643", "width": 640, "height": 480, "fps": 30},
        }' \
    --dataset.repo_id=limboyo/lerobot_openarm \
    --dataset.single_task="Left gripper picks the light blue stapler into the open black box, then right gripper places the blue duck into the box." \
    --display_data=false \
    --dataset.episode_time_s=1000 \
    --policy.path=/home/mizu/lerobot/ckpt/pretrained_model

lerobot-rollout \
    --robot.type=bi_openarm_follower \
    --robot.left_arm_config.port=can2 \
    --robot.left_arm_config.use_can_fd=true \
    --robot.left_arm_config.side=left \
    --robot.right_arm_config.port=can3 \
    --robot.right_arm_config.use_can_fd=true \
    --robot.right_arm_config.side=right \
    --robot.id=my_bimanual_follower \
    --robot.left_arm_config.cameras='{
        wrist_left: {type: intelrealsense,serial_number_or_name: "260322272305", width: 640, height: 480, fps: 30},
        top: {type: intelrealsense, serial_number_or_name: "141322252011", width: 640, height: 480, fps: 30}
        }' \
    --robot.right_arm_config.cameras='{
        wrist_right: {type: intelrealsense,serial_number_or_name: "352122273643", width: 640, height: 480, fps: 30}
        }' \
    --policy.path=/home/mizu/lerobot/ckpt/pretrained_model \
    --task="Left gripper picks the light blue stapler into the open black box,then right gripper places the blue duck into the box." \
    --display_data=false
