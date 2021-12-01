// How to run:
//
// 		go run ./cmd/dnn-blob/main.go [videosource] [modelfile] [configfile]
//
//  To use, first download models an extract them:
//  * wget http://download.tensorflow.org/models/object_blob/ssd_mobilenet_v1_coco_2017_11_17.tar.gz (modelFile)
//  * wget https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt (configFile)

package main

import (
	"fmt"
	"image"
	"os"
	"strconv"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\ndnn-blob [videosource] [modelfile] [configfile]")
		return
	}

	// parse args
	videosource := os.Args[1]
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

	// open capture device (webcam or file)
	var (
		capture *gocv.VideoCapture
		err     error
	)

	// If it is a number, open a video capture from webcam, else from file
	id, err := strconv.Atoi(videosource)
	if err == nil {
		capture, err = gocv.OpenVideoCapture(id)
	} else {
		capture, err = gocv.VideoCaptureFile(videosource)
	}
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", videosource)
		return
	}
	defer capture.Close()

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

	fmt.Printf("Start reading device: %v\n", videosource)

	var blobList BlobList
	for {
		if ok := capture.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", videosource)
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

		blobs := performBlob(&img, prob)
		if blobList.Update(blobs) {
			println("Blobs changed in the scene!")
		}
		DrawBlobs(&img, blobList.Blobs())

		prob.Close()
		blob.Close()

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

// performBlob analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of blobs, and each blob
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performBlob(frame *gocv.Mat, results gocv.Mat) []Blob {
	var blobs []Blob
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.75 {
			pos := BlobPosition{
				left:   int(results.GetFloatAt(0, i+3) * float32(frame.Cols())),
				top:    int(results.GetFloatAt(0, i+4) * float32(frame.Rows())),
				right:  int(results.GetFloatAt(0, i+5) * float32(frame.Cols())),
				bottom: int(results.GetFloatAt(0, i+6) * float32(frame.Rows())),
			}
			classId := int(results.GetFloatAt(0, i+1))

			c := ClassID(classId)
			if c.Known() {
				blobs = append(blobs, Blob{
					Class:      c,
					Confidence: float64(confidence),
					Position:   pos,
				})
			}
		}
	}
	return blobs
}

func DrawBlobs(frame *gocv.Mat, blobs []Blob) {
	for i, d := range blobs {
		status := fmt.Sprintf("type: %v, confidence: %v", d.Class.String(), d.Confidence)
		gocv.PutText(frame, status, image.Pt(10, 20*(len(blobs)-i)), gocv.FontHersheyPlain, 1.0, d.Color(), 2)
		gocv.Rectangle(frame, image.Rect(d.Position.left, d.Position.top, d.Position.right, d.Position.bottom), d.Color(), 2)
	}
}
