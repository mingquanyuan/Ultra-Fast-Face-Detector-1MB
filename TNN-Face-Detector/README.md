# Build From Source
```
cd Ultra-Fast-Face-Detector-1MB/TNN-Face-Detector
git clone https://github.com/Tencent/TNN.git
cd TNN
```

TNN支持大多数的主流硬件平台和系统，以下基于`aarch64`架构linux系统的Raspberry Pi 4 Model B为例：
```
cd scripts
./build_aarch64_linux.sh
cd ../../

sudo bash ./build_aarch64_linux_face_detector.sh
```

# Inference
```
cd build
./demo_arm_linux_facedetector_retinaface ../model/faceDetector_mobile.opt.tnnproto ../model/faceDetector_mobile.opt.tnnmodel 768 1024 ../img/sample_1_1024_768.jpg
```