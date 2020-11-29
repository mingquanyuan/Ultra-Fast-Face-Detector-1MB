# pth to onnx
- https://github.com/biubug6/Face-Detector-1MB-with-landmark
- notes: 
    - add additional arg in the script such that we can set both input width and height independetly
        ```
        parser.add_argument('--img_width', default=320, type=int, help='when origin_size is false, img_width is scaled size(320 or 640 for long side)')
        parser.add_argument('--img_height', default=240, type=int, help='when origin_size is false, img_height is scaled size(320 or 640 for long side)')
        ```
    - change the output names which can be used with ease
        ```
        output_names = ["loc", "score", "landms"]
        inputs = torch.randn(1, 3, args.img_height, args.img_width).to(device)
        torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
        ```

# onnx to tnn
- https://github.com/Tencent/TNN/blob/master/doc/cn/user/onnx2tnn.md