# falco-home-security

Home of the falco-home-security project for the Sysdig Hackaton 2021.  
Team: **CATBUSTERS**!

Fear of thieves?  
No more!  
Falco is here to protect your home!

With this plugin, Falco will become your smart home protector;  
Want to know how many people are in your home? How many cats? Here we are!

Basically, the idea is to capture image from IP cameras/webcams around your house, and let Falco manage events coming from those, through a plugin!

## Setup

You obviously need Falco 0.31 (from master) that has plugins support enabled;  
Moreover, a tensorflow model is required:  

    $ wget http://download.tensorflow.org/models/object_blob/ssd_mobilenet_v1_coco_2017_11_17.tar.gz (modelFile)
    $ wget https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt (configFile)

Please be sure to extract that tarball.  

## Build

A makefile is made available to let you build either the Falco plugin or a standalone program.  
To build Falco plugin, issue:

    $ make libhomesecurity.so

To build standalone program:

    $ make main

## Run

To run the plugin, please refer to: https://falco.org/blog/falco-plugins-early-access/#configuring-plugins-in-falco

Instead, the standalone executable can be run with: 

    $ ./plugin "$CAPTURE_DEV" "$PATH_TO_ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb" "PATH_TO_ssd_mobilenet_v1_coco_2017_11_17.pbtxt"

where CAPTURE_DEV is either:
* index of webcam device
* ip address for a network ip camera
* path to video file

## Plugin parameters

### InitConfig
```
{
  "model": "./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb",
  "netConfig": "./models/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
  "backend": "",
  "target": "",
  "minConfidence": 0.75,
  "memoryMinConfidence": 0.50,
  "memoryDecayFactor": 0.98,
  "memoryNearnessThreshold": 0.65,
  "memoryClassSwitchThreshold": 0.15,
  "memoryCollapseMultiple": true
}
```

* model: path to pb model
* netConfig:
* backend: opencv backend, between { halide, openvino, opencv, vulkan, cuda, default }
* target: opencv target, between { cpu, fp32, fp16, vpu, vulkan, fpga, cuda, cudafp16 }
* minConfidence: minimum confidence for new detected blobs
* memoryMinConfidence: at each refresh cycle, blobs are discarded if their confidence goes below this value
* memoryDecayFactor: at each refresh cycle, the confidence of each blob is reduced by this factor
* memoryNearnessThreshold: while searching for near blobs, this is the minimum value required to consider two blob similars
* memoryClassSwitchThreshold: while merging a new blob with a new one, the new blob should surpass the condidence of the known blob by this threshold, in order to override its confidence and class values.
* memoryCollapseMultiple: collapses all the near rectangles in a single one

### OpenParams
```
{
  "videoSource": "0", // /dev/video0
  "showWindow": true,
  "snapshotPath": "./snapshots"
}
```

* videoSource: capture device to be used, see above CAPTURE_DEV
* showWindow: whether to also show a GUI window
* snapshotPath: folder where to store snapshots for each event; if not set, no snapshot will be taken


