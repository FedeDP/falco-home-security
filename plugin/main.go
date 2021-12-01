//  To use, first download models an extract them:
//  * wget http://download.tensorflow.org/models/object_blob/ssd_mobilenet_v1_coco_2017_11_17.tar.gz (modelFile)
//  * wget https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt (configFile)

package main

import (
	"bytes"
	"errors"
	"fmt"
	"github.com/nfnt/resize"
	"image"
	"os"
	"os/signal"
	"reflect"
	"strconv"
	"sync"
	"syscall"
	"time"

	"gocv.io/x/gocv"
)

// VideoEvent represents the event payload to be serialized
type VideoEvent struct {
	VideoSource 		string
	Blobs       		[]Blob
	SnapshotPath 		string
	AsciiImage			string
}

var errDeviceClosed = errors.New("device has been closed")

type RenderChan chan gocv.Mat

type QuitChan chan bool

type DetectionChan chan VideoEvent

type ErrorChan chan error

type DetectionConfig struct {
	Model     string `json:"model"`
	NetConfig string `json:"netConfig"`

	// (optional)
	Backend string `json:"backend"`

	// (optional)
	Target string `json:"target"`

	// (optional) Minimum confidence for new detected blobs.
	MinConfidence float64 `json:"minConfidence"`

	// (optional) At each refresh cycle, blobs are discarded if their confidence goes
	// below this value.
	MemoryMinConfidence float64 `json:"memoryMinConfidence"`

	// (optional) At each refresh cycle, the confidence of each blob is reduced by
	// this factor.
	MemoryDecayFactor float64 `json:"memoryDecayFactor"`

	// (optional) While searching for near blobs, this is the minimum value required
	// to consider two blob similars.
	MemoryNearnessThreshold float64 `json:"memoryNearnessThreshold"`

	// (optional) While merging a new blob with a new one, the new blob should surpass
	// the condidence of the known blob by this threshold, in order to override
	// its confidence and class values.
	MemoryClassSwitchThreshold float64 `json:"memoryClassSwitchThreshold"`

	// (optional) Collapses all the near rectangles in a single one
	MemoryCollapseMultiple bool `json:"memoryCollapseMultiple"`
}

func LaunchVideoDetection(cfg *DetectionConfig, oCfg *OpenConfig, quitc QuitChan, wg *sync.WaitGroup) (DetectionChan, RenderChan, ErrorChan) {
	detectionChan := make(DetectionChan)
	renderChan := make(RenderChan)
	errorChan := make(ErrorChan)
	go func() {
		defer close(detectionChan)
		defer close(renderChan)
		defer close(errorChan)

		var (
			capture *gocv.VideoCapture
			err     error
		)

		// open capture device (webcam or file)
		// If it is a number, open a video capture from webcam, else from file
		id, err := strconv.Atoi(oCfg.VideoSource)
		if err == nil {
			capture, err = gocv.OpenVideoCapture(id)
		} else {
			capture, err = gocv.VideoCaptureFile(oCfg.VideoSource)
		}
		if err != nil {
			errorChan <- fmt.Errorf("error opening video capture device: %v", oCfg.VideoSource)
			return
		}
		defer capture.Close()

		img := gocv.NewMat()
		defer img.Close()

		// open DNN object tracking model
		net := gocv.ReadNet(cfg.Model, cfg.NetConfig)
		if net.Empty() {
			errorChan <- fmt.Errorf("error reading network model from : %v %v", cfg.Model, cfg.NetConfig)
			return
		}
		defer net.Close()

		_ = net.SetPreferableBackend(gocv.ParseNetBackend(cfg.Backend))
		_ = net.SetPreferableTarget(gocv.ParseNetTarget(cfg.Target))

		ratio := 1.0 / 127.5
		mean := gocv.NewScalar(127.5, 127.5, 127.5, 0)

		var blobList BlobList
		for {
			select {
			case <-quitc:
				return
			default:
			}

			if ok := capture.Read(&img); !ok {
				select {
				case <-quitc:
					return
				case errorChan <- errDeviceClosed:
					return
				}
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

			blobs := performBlob(&img, prob, cfg.MinConfidence)
			if blobList.Update(blobs, cfg) {
				var imgPath string
				if len(oCfg.SnapshotPath) > 0 {
					imgPath = oCfg.SnapshotPath + "/" + GetImageFileName()
					gocv.IMWrite(imgPath, img)
				}

				videoEv := VideoEvent{
					VideoSource: oCfg.VideoSource,
					Blobs:       blobList.Blobs(),
					SnapshotPath: imgPath,
				}
				goImg, err := img.ToImageYUV()
				if err == nil {
					videoEv.AsciiImage = string(Convert2Ascii(ScaleImage(goImg, 80)))
				} else {
					fmt.Printf("error: %s", err.Error())
				}

				select {
				case <-quitc:
					return
				case detectionChan <- videoEv:
				}

			}

			prob.Close()
			blob.Close()

			if oCfg.ShowWindow {
				DrawBlobs(&img, blobList.Blobs())
				select {
				case <-quitc:
					return
				case renderChan <- img:
				}
			}
		}
	}()
	return detectionChan, renderChan, errorChan
}

// performBlob analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of blobs, and each blob
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performBlob(frame *gocv.Mat, results gocv.Mat, minConfidence float64) []Blob {
	var blobs []Blob
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if float64(confidence) > minConfidence {
			pos := BlobPosition{
				Left:   int(results.GetFloatAt(0, i+3) * float32(frame.Cols())),
				Top:    int(results.GetFloatAt(0, i+4) * float32(frame.Rows())),
				Right:  int(results.GetFloatAt(0, i+5) * float32(frame.Cols())),
				Bottom: int(results.GetFloatAt(0, i+6) * float32(frame.Rows())),
			}
			classId := int(results.GetFloatAt(0, i+1))

			c := ParseClassID(classId)
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
		gocv.Rectangle(frame, image.Rect(d.Position.Left, d.Position.Top, d.Position.Right, d.Position.Bottom), d.Color(), 2)
	}
}

func GetImageFileName() string {
	const layout = "01-02-2006_15.04.05.000"
	t := time.Now()
	return "Falco-" + t.Format(layout) + ".png"
}

func ScaleImage(img image.Image, w int) (image.Image, int, int) {
	sz := img.Bounds()
	h := (sz.Max.Y * w * 10) / (sz.Max.X * 16)
	img = resize.Resize(uint(w), uint(h), img, resize.Lanczos3)
	return img, w, h
}

func Convert2Ascii(img image.Image, w, h int) []byte {
	var ASCIISTR = "@%#*+=-:. "
	table := []byte(ASCIISTR)
	buf := new(bytes.Buffer)

	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			y := reflect.ValueOf(img.At(j, i)).FieldByName("Y").Uint()
			pos := int(y) * (len(ASCIISTR) - 1) / 255
			_ = buf.WriteByte(table[pos])
		}
		_ = buf.WriteByte('\n')
	}
	return buf.Bytes()
}

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\nplugin [videosource] [modelfile] [configfile]")
		return
	}

	// parse args
	videosource := os.Args[1]
	model := os.Args[2]
	config := os.Args[3]
	var backend string
	if len(os.Args) > 4 {
		backend = os.Args[4]
	}

	var target string
	if len(os.Args) > 5 {
		target = os.Args[5]
	}

	cfg := DetectionConfig{
		Model:                      model,
		NetConfig:                  config,
		Backend:                    backend,
		Target:                     target,
		MinConfidence:              0.75,
		MemoryMinConfidence:        0.5,
		MemoryDecayFactor:          0.98,
		MemoryNearnessThreshold:    0.65,
		MemoryClassSwitchThreshold: 0.15,
		MemoryCollapseMultiple:     true,
	}

	oCfg := OpenConfig{
		VideoSource: videosource,
		ShowWindow:  true,
	}

	var window *gocv.Window
	if oCfg.ShowWindow {
		window = gocv.NewWindow("Falco Home Security")
		defer window.Close()
	}

	var wg sync.WaitGroup
	quitc := make(QuitChan)
	detectionc, renderc, errorc := LaunchVideoDetection(&cfg, &oCfg, quitc, &wg)
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc,
		syscall.SIGINT,
		syscall.SIGTERM,
		syscall.SIGQUIT)

	for {
		select {
		case <-sigc:
			return
		case e := <-errorc:
			fmt.Printf("Exiting: %v\n", e)
			return
		case <-detectionc:
			fmt.Println("Blobs changed")
		case img := <-renderc:
			if oCfg.ShowWindow {
				window.IMShow(img)
				if window.WaitKey(1) >= 0 || window.GetWindowProperty(gocv.WindowPropertyVisible) == 0 {
					println("user quit")
					return
				}
			}
		}
	}
}
