# VR 直控底盘与升降（CH343 / RoboMaster C板）

状态：C板下位机已烧录，底盘/升降独立VR控制和底盘模式切换已经实车验证；当前又完成了与
双臂真机控制的同进程融合。联合整机仍需按本文命令进行首次低速现场验证。

## 设计范围

- 不需要遥控器，不读取 `ch4/ch5`，C板上电后直接等待VR串口命令。
- 底盘继续使用原 `classic.c` 中经过遥控器实车验证的 `MANUAL_MODE`（平移）和 `TURN_MODE`（自转）；串口 `turn_flag` 只负责在两者之间切换，原7字节 `0x55` 字段保持不变。
- 升降继续使用原RobStride PP位置控制，最高点仍为 `0 rad`，范围 `[-1.7, 0] rad`。
- 没有会话号、递增序号、ACK、C板状态回传、CAN反馈超时或双臂状态文件联锁。
- 沿用原NUC离线计数；串口断流后底盘进入原 `SAFE_MODE`，升降保持最后位置目标。

## VR键位

| 控制对象 | 操作 | 功能 |
|---|---|---|
| 左臂／右臂 | 按住对应Grip，移动或旋转手柄 | 原有双臂末端控制，不受移动桥接修改 |
| 夹爪 | Grip按住时按／松Trigger | 原有渐进张开／闭合 |
| 升降 | 左Grip + 左摇杆上／下 | 上升／下降；左Grip按住时左摇杆不控制轮向 |
| 底盘行驶 | 右Grip + 右摇杆上／下 | 沿当前轮向前进／后退 |
| 轮组转向 | 右Grip按住、左Grip松开 + 左摇杆左右 | 在 -90°～90° 内调整轮组转角 |
| 平移／自转模式切换 | 右Grip + 按下右摇杆 | 每按一次只切换一次构型；不产生旋转速度 |
| 自转 | 自转模式下右Grip + 右摇杆左／右 | 摇杆控制逆／顺时针速度；回中停止并保持自转构型 |
| 底盘停止 | 松开右Grip | 速度归零；保持当前轮向和平移／自转构型，未完成的构型切换继续到最后目标 |
| 双臂复位 | 两侧Grip松开时按左X | 双臂缓慢回准备姿态；A不触发复位 |

平移模式下右摇杆左右不会写入轮向；只有切到自转模式后才控制旋转。再次按下右摇杆返回平移时，恢复进入自转前保留的轮向，不自动回正。B/Y不控制底盘。X/A不会被移动桥接当成暂停键。

## 串口协议

USART1：115200、8N1、无流控，C板 `PB7=RX`、`PA9=TX`。原来的前7字节完全保留，只在末尾追加2个升降字节：

```text
55 | 原字段1 | 原字段2 | 原方向字段 | 原字段4 | 原字段5 | 原turn_flag | lift_enable | lift_speed
```

固定9字节：

- Byte 0~6：保持原 `0x55` NUC底盘协议不变，包括原角度、方向、速度和 `turn_flag`。
- Byte 7：`lift_enable`，左Grip按住为1，松开为0。
- Byte 8：`lift_speed`，有符号8位数，单位mrad/s，范围-100~100。

## 安装与编译

```bash
python3 examples/mobile_openarm/stm32_serial/install_firmware.py
python3 examples/mobile_openarm/stm32_serial/install_firmware.py --apply
```

安装器检查原文件哈希并备份，但不会编译或烧录。随后用Keil打开：

```text
~/000-11.4_Nav_follow/MDK-ARM/can.uvprojx
```

## 运行

### 整机联合控制（正常使用）

双臂、夹爪、底盘和升降现在由同一个真机进程控制：

```bash
bash examples/mobile_openarm/run_vr_real.sh --execute --viewer \
  --mobile-port /dev/ttyACM0
```

该进程只订阅一次VR数据，并独占双臂CAN和C板串口。不要再同时运行下面的独立移动桥接。

### 仅底盘/升降诊断

只预览VR转换结果，不打开串口：

```bash
bash examples/mobile_openarm/run_vr_mobile.sh
```

烧录并完成架空方向检查后才执行：

```bash
bash examples/mobile_openarm/run_vr_mobile.sh \
  --port /dev/serial/by-id/实际CH343设备 --execute
```

默认平移上限 `0.05 m/s`、升降 `0.05 rad/s`。方向不一致时分别使用
`--drive-sign -1`、`--steer-sign -1`、`--spin-sign -1`、`--lift-sign -1`。

## 无硬件测试

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest \
  tests/robots/test_mobile_serial.py -q
```
