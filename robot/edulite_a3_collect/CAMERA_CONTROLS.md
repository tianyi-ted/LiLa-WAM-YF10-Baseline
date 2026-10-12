# 固定相机参数

主配置 `camera_controls_file: ./camera_controls.yaml` 指向独立配置，路径相对主配置解析。
采集、预览和当前真机推理入口均加载它；推理使用 `head_camera`，胸部相机仍保存为原始采集数据。

两台相机已按用户提供的设备控制列表启用 `enabled: true`，下次启动时应用。
胸部相机使用手动曝光 100（V4L2 标准单位 100 µs，即约 10 ms）、手动色温 4600、增益 4、50 Hz 防闪烁设置。
之前曝光 60（约 6 ms）拍摄的三张图片中出现移动的横向明暗带；
10 ms 是针对 50 Hz 市电所致 100 Hz 光波动的第一轮实验值。
若画面太暗，优先将 gain 从 0 逐步提高到 4、8、16；若仍有条纹，
光源可能采用其他频率 PWM，需优先调整或更换照明。
曝光和色温在原输出中是 inactive，因此这些数值只是手动初始值，不能当作
自动模式当时实际采用的曝光和色温；切换后必须预览确认亮度和颜色。
头部相机固定亮度 50、对比度/饱和度/色调 128。2026-09-16 的截图显示，
关闭自动白平衡会使白色桌面明显偏绿，所以当前设置为
`white_balance_automatic: 1`，让相机自行校正颜色。它没有暴露曝光、增益、
手动色温或 RGB 增益控制，因此无法仅靠现有 V4L2 参数保证与胸部相机
完全一致，也无法锁定跨次实验的头部白平衡。每次开始采集前保持相同照明，
给自动白平衡留几秒稳定时间，并保存两台相机的对照截图。
上述参数尚未在当前环境连接硬件验证，以启动回读和真实画面为准。
在连接相机的电脑关闭采集程序后查询：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls-menus
v4l2-ctl -d /dev/video8 --list-ctrls-menus
```

按输出修改控制项名称、菜单值和范围，删除不支持的项。
曝光名称可能为 exposure_auto/exposure_absolute，也可能为
auto_exposure/exposure_time_absolute，以设备实际输出为准。
先关闭自动曝光再设置曝光时间；先关闭自动白平衡再设置色温。
如有自动增益或自动对焦，也需按实际控制名称关闭并设置手动值。
快门通常就是曝光时间，不必另设一个“快门”参数。单位须按设备确认。
如要暂停应用配置，将相应相机 enabled 改为 false；它不会恢复设备之前的参数。

每次启动在分辨率设置后依次写入控制项并回读，成功打印 CAMERA CONTROL VERIFIED。
设置失败或读回不一致将报错并释放当前相机；失败前已写入的参数不会回滚。
修改 YAML 后重启预览/采集生效。dry-run 不访问硬件，不能验证硬件参数。

预览命令：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_collect/preview_camera.py --config robot/edulite_a3_collect/config.master_slave.yaml --camera chest_camera
```

按 s 保存截图，q 退出。复制 camera_controls.yaml 并修改主配置引用路径，
即可保存不同实验的参数。锁定参数不能消除光照变化，请同时固定照明与相机位置。
设备插拔可能改变 video 节点，使用前核对设备身份。
