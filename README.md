# TestSimpleCoreMLCommandLine
Run a simple Core ML classification model without using coremlc generated wrapper 

## files
* MobileNet_EdgeTPU_multi_arrays.mlmodel: converted from MobileNet EdgeTPU model, Both input and output types are MLMultiArray
* grace_hopper.raw: 24-bit rgb raw image, converted with something like
```
convert grace_hopper.bmp -gravity Center  -resize 224x224 -extent 224x224  foo.rgb
```

## run it 
```
make run
```
You should get something like
```
2022-07-10 21:16:06.934 test_cmd[91957:28263153] 653
```
653 means the (653+1)-th class, which is "military uniform". Since this image is Rear Admiral Grace Hopper in military uniform,
this is an expected result.
