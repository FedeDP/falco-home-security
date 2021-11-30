// How to run:
//
// 		go run ./cmd/dnn-detection/main.go [videosource] [modelfile] [configfile]
//
//  To use, first download models an extract them:
//  * wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz (modelFile)
//  * wget https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt (configFile)

package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"os"
)

type Blob struct {
	class ClassID
	confidence float32
}

type ClassID int
const (
	Human ClassID = 1
	Cat = 16
	Dog = 17
)

func (c ClassID) String() string {
	switch c {
	case Human:
		return "Human"
	case Cat:
		return "Cat"
	case Dog:
		return "Dog"
	}
	return ""
}

func (c ClassID) Known() bool {
	switch c {
	case Human, Cat, Dog:
		return true
	}
	return false
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\ndnn-detection [videosource] [modelfile] [configfile]")
		return
	}

	// parse args
	deviceID := os.Args[1]
	model := os.Args[2]
	config := os.Args[3]
	backend := gocv.NetBackendDefault
	if len(os.Args) > 4 {
		backend = gocv.ParseNetBackend(os.Args[4])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 5 {
		target = gocv.ParseNetTarget(os.Args[5])
	}

	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	// Create output window
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
	net.SetPreferableBackend(backend)
	net.SetPreferableTarget(target)

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
func performDetection(frame *gocv.Mat, results gocv.Mat) []Blob {
	var blobs []Blob
	statusColor := color.RGBA{G: 255}
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			classId := int(results.GetFloatAt(0, i+1))

			c := ClassID(classId)
			if c.Known() {
				blobs = append(blobs, Blob{
					class: c,
					confidence: confidence,
				})
				status := fmt.Sprintf("type: %v, confidence: %v\n", c.String(), confidence)
				gocv.PutText(frame, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.0, statusColor, 2)
				gocv.Rectangle(frame, image.Rect(left, top, right, bottom), statusColor, 2)
			}
		}
	}
	return blobs
}