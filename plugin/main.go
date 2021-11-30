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
	"math"
	"os"
	"strconv"
)

type Ring struct {
	vals []float32
	cursor int
}

func (r *Ring) Push(val float32) {
	r.vals[r.cursor] = val
	r.cursor++
	r.cursor %= len(r.vals)
}

func NewRing(size int) *Ring {
	return &Ring{
		vals:   make([]float32, size, size),
		cursor: 0,
	}
}

type ClassHistogram map[ClassID]*Ring

type BlobPosition struct {
	left  int
	top int
	right int
	bottom int
}

type BlobCenter struct {
	x int
	y int
}

func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (bp BlobPosition) Center() BlobCenter {
	x := (bp.right - bp.left) / 2
	y := (bp.bottom - bp.top) / 2
	return BlobCenter{x, y}
}

func (bc BlobCenter) Near(other BlobCenter) int {
	const threshold = 10

	if Abs(bc.x - other.x) < threshold &&
		Abs(bc.y - other.y) < threshold {

		return (bc.x - other.x) + Abs(bc.y - other.y)
	}
	return math.MaxInt
}

type Blob struct {
	position BlobPosition
	cumulativeConfidence ClassHistogram
	nframes int
}

type Blobs []Blob

// See https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
type ClassID int
const (
	Human ClassID = 1
	Cat = 17
	Dog = 18
)

var Classes = []ClassID{ Human, Cat, Dog }

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

func (b Blob) Color() color.RGBA {
	switch b.cumulativeConfidence.Winner() {
	case Human:
		return color.RGBA{B: 255}
	case Cat:
		return color.RGBA{G: 255}
	case Dog:
		return color.RGBA{R: 255}
	}
	return color.RGBA{}
}

func (b Blob) Mean(class ClassID) float32 {
	sum := b.cumulativeConfidence.Sum(class)

	div := b.nframes
	if div > 8 {
		div = 8
	}
	return sum / float32(div)
}

func (ch ClassHistogram) Sum(class ClassID) float32 {
	var sum float32
	for _, val := range ch[class].vals {
		sum += val
	}
	return sum
}

func (ch ClassHistogram) Winner() ClassID {
	var (
		maxVal float32
		maxClass ClassID
	)
	for class := range ch {
		sum := ch.Sum(class)
		if sum > maxVal {
			maxVal = sum
			maxClass = class
		}
	}
	return maxClass
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\ndnn-detection [videosource] [modelfile] [configfile]")
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
		err error
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

	var blobs []Blob
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

		blobs = performDetection(&img, prob, blobs)

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
func performDetection(frame *gocv.Mat, results gocv.Mat, oldBlobs Blobs) Blobs {
	var blobs []Blob
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.5 {
			pos := BlobPosition{
				left: int(results.GetFloatAt(0, i+3) * float32(frame.Cols())),
				top: int(results.GetFloatAt(0, i+4) * float32(frame.Rows())),
				right: int(results.GetFloatAt(0, i+5) * float32(frame.Cols())),
				bottom: int(results.GetFloatAt(0, i+6) * float32(frame.Rows())),
			}
			classId := int(results.GetFloatAt(0, i+1))

			c := ClassID(classId)
			if c.Known() {
				blob := oldBlobs.findNearest(pos.Center())
				if blob == nil {
					fmt.Println("found new blob!")
					// New blob!
					blob = &Blob{
						position: pos,
						cumulativeConfidence: make(map[ClassID]*Ring),
					}
					for _, class := range Classes {
						blob.cumulativeConfidence[class] = NewRing(8)
					}
				} else {
					fmt.Println("found old blob!")
					blob.position = pos
				}
				blob.nframes++
				blob.cumulativeConfidence[c].Push(confidence)
				blobs = append(blobs, *blob)

				if blob.nframes > 3 {
					status := fmt.Sprintf("type: %v, confidence: %v, nframes: %v", c.String(), blob.Mean(blob.cumulativeConfidence.Winner()), blob.nframes)
					gocv.PutText(frame, status, image.Pt(10, 20*len(blobs)), gocv.FontHersheyPlain, 1.0, blob.Color(), 2)
					gocv.Rectangle(frame, image.Rect(pos.left, pos.top, pos.right, pos.bottom), blob.Color(), 2)
				}
			}
		}
	}
	return blobs
}

func (bb Blobs) findNearest(center BlobCenter) *Blob {
	if bb == nil {
		return nil
	}
	var (
		minVal = math.MaxInt
		minValBlob *Blob
	)
	for _, b := range bb {
		val := b.position.Center().Near(center)
		if minVal > val {
			minVal = val
			minValBlob = &b
		}
	}
	return minValBlob
}