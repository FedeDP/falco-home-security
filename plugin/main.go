// What it does:
//
// This example uses a deep neural network to perform object detection.
// It can be used with either the Caffe face tracking or Tensorflow object detection models that are
// included with OpenCV 3.4
//
// To perform face tracking with the Caffe model:
//
// Download the model file from:
// https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
//
// You will also need the prototxt config file:
// https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
//
// To perform object tracking with the Tensorflow model:
//
// Download and extract the model file named "frozen_inference_graph.pb" from:
// http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
//
// You will also need the pbtxt config file:
// https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt
//
// How to run:
//
// 		go run ./cmd/dnn-detection/main.go [videosource] [modelfile] [configfile] [classModel] [classFile]
//

//  To use, first download models an extract them:
//  * wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz (modelFile)
//  * wget https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt (configFile)
//  * wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip (classModel + classFile)

package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"
)

var (
	classNet     gocv.Net
	descriptions []string
)

func main() {
	if len(os.Args) < 6 {
		fmt.Println("How to run:\ndnn-detection [videosource] [modelfile] [configfile] [classModel] [classFile]")
		return
	}

	// parse args
	deviceID := os.Args[1]
	model := os.Args[2]
	config := os.Args[3]
	classModel := os.Args[4]
	classPath := os.Args[5]
	backend := gocv.NetBackendDefault
	target := gocv.NetTargetCPU

	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("DNN Detection")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// open DNN object tracking model
	net := gocv.ReadNet(model, config)
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, config)
		return
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	// open DNN classifier
	classNet = gocv.ReadNet(classModel, "")
	if classNet.Empty() {
		fmt.Printf("Error reading network model : %v\n", classModel)
		return
	}
	defer classNet.Close()
	classNet.SetPreferableBackend(gocv.NetBackendType(backend))
	classNet.SetPreferableTarget(gocv.NetTargetType(target))

	descriptions, err = readDescriptions(classPath)
	if err != nil {
		fmt.Printf("Error reading descriptions file: %v\n", classPath)
		return
	}

	ratio := 1.0 / 127.5
	mean := gocv.NewScalar(127.5, 127.5, 127.5, 0)

	fmt.Printf("Start reading device: %v\n", deviceID)

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// convert image Mat to 300x300 blob that the object detector can analyze
		blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, true, false)

		// feed the blob into the detector
		net.SetInput(blob, "")

		// run a forward pass thru the network
		prob := net.Forward("")

		performDetection(&img, prob)

		prob.Close()
		blob.Close()

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performDetection(frame *gocv.Mat, results gocv.Mat) {
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))

			r := image.Rect(left, top, right, bottom) // take bounding box
			bbox := frame.Region(r)

			// convert image Mat to 224x224 blob that the classifier can analyze
			blob := gocv.BlobFromImage(bbox, 1.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), true, false)

			// feed the blob into the classifier
			classNet.SetInput(blob, "input")

			// run a forward pass thru the network
			prob := classNet.Forward("softmax2")

			// reshape the results into a 1x1000 matrix
			probMat := prob.Reshape(1, 1)

			// determine the most probable classification
			_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)

			// display classification
			desc := "Unknown"
			if maxLoc.X < 1000 {
				desc = descriptions[maxLoc.X]
			}
			if i == 0 {
				status := fmt.Sprintf("description: %v, maxVal: %v\n", desc, maxVal)
				statusColor := color.RGBA{0, 255, 0, 0}
				gocv.PutText(frame, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)
			}
			blob.Close()
			prob.Close()
			probMat.Close()

			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), color.RGBA{0, 255, 0, 0}, 2)
		}
	}
}

// readDescriptions reads the descriptions from a file
// and returns a slice of its lines.
func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}
