# InstantID-swapface-multiple_in_out
Base on InstantID,support multiple inputs and outputs, face fusion, swap face.

## Demo

### multiple inputs and outputs, face fusion

![demo1.png](demo_img%2Fdemo1.png)

You can choose multiple pose images as input, which will generate multiple outputs.

![demo3.png](demo_img%2Fdemo3.png)
If you only choose face images as input, which will generate one face fusion output.

### swap face

![demo2.png](demo_img%2Fdemo2.png)

Only swap face, support multiple outputs.

## Guide

Same as [InstantID](https://github.com/InstantID/InstantID)

In addition, for **swap face**, you need download [face_occluder.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models/face_occluder.onnx).

## Start a local gradio demo

Run the following command:

    python gradio_demo/app-multicontrolnet-multi.py

for swapface, run:
    
    python gradio_demo/app-multicontrolnet-multi-swapface.py
