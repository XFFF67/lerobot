# OpenArm 单环境部署（验证阶段）

该方案把双臂 CAN、C 板串口、VR Server、MuJoCo、Pinocchio 和 CasADi 放在同一个
`openarm-runtime` Conda 环境中。现有 `lerobot312`、`hei-rebot-vr` 和原启动脚本保持不变，
在单环境完成实车验证前可随时回退。

Pinocchio 3.9.0 与 CasADi 3.7.0 由 conda-forge 安装。不要将 Pinocchio 改为 PyPI 的
`pin==4.1.0`，该 wheel 不包含本项目 IK 使用的 `pinocchio.casadi` 绑定。

## 创建环境

在仓库根目录执行：

```bash
bash examples/mobile_openarm/setup_openarm_runtime.sh
```

脚本发现同名环境时会拒绝修改，避免覆盖已有环境。也可以指定测试名称：

```bash
bash examples/mobile_openarm/setup_openarm_runtime.sh --name openarm-runtime-test
```

## 无硬件检查

```bash
conda activate openarm-runtime
python examples/mobile_openarm/check_openarm_runtime.py --full
```

## 启动 VR Server

```bash
conda activate openarm-runtime
bash examples/mobile_openarm/run_vr_server_single_env.sh --hostname <NUC局域网IP>
```

## 启动实车

另开终端：

```bash
conda activate openarm-runtime
bash examples/mobile_openarm/run_vr_real_single_env.sh \
  --execute \
  --left-port can0 \
  --right-port can1 \
  --mobile-port /dev/serial/by-id/<C板设备>
```

两个启动脚本只使用当前 `conda activate` 后的 `python`；不会调用固定的 Conda 环境路径。
它们会将动态库路径限定为当前 Conda 环境，避免 ROS 环境中的旧 `libstdc++` 干扰。
默认不加 `--viewer`，因此不打开 MuJoCo 窗口。

完成软件检查后仍需依次进行 CAN 只读、双臂准备、底盘/升降和全身 VR 实车验证。全部通过前，
不要删除原来的两个环境，也不要替换原启动脚本。
