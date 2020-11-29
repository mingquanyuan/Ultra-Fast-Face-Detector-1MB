TNN支持大多数的主流硬件平台和系统，以下文档基于`aarch64`架构linux系统（Raspberry Pi 4 Model B）。

# Build From Source
```
cd Ultra-Fast-Face-Detector-1MB/TNN-Face-Detector
git clone https://github.com/Tencent/TNN.git
cd TNN/scripts && ./build_aarch64_linux.sh
cd -

bash ./build_aarch64_linux_face_detector.sh
```

# Inference
```
cd build 

./demo_arm_linux_facedetector_retinaface ../model/faceDetector_mobile.opt.tnnproto ../model/faceDetector_mobile.opt.tnnmodel 768 1024 ../../imgs/sample_1_1024_768.jpg
```



# Todo
- 加入对`armhf`架构的支持
- 可视化人脸关键点