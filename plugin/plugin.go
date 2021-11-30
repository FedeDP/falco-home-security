package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins/extractor"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins/source"
	"gocv.io/x/gocv"
)

type VideoPlugin struct {
	plugins.BasePlugin
	cfg *initConfig
}

type VideoInstance struct {
	source.BaseInstance
	capture *gocv.VideoCapture
	net     *gocv.Net
}

// VideoEvent represents the event payload to be serialized
type VideoEvent struct {
}

type initConfig struct {
	VideoSource string `json:"videoSource"`
	Model       string `json:"model"`
	Backend     string `json:"backend"`
	Target      string `json:"target"`
	NetConfig   string `json:"netConfig"`
}

func init() {
	p := &VideoPlugin{}
	extractor.Register(p)
	source.Register(p)
}

// Info returns a pointer to a plugin.Info struct, containing all the
// general information about this plugin.
// This method is mandatory for source plugins.
func (m *VideoPlugin) Info() *plugins.Info {
	return &plugins.Info{
		ID:                 999,
		Name:               "home-security",
		Description:        "Video recognition plugin",
		Contact:            "github.com/FedeDP/falco-home-security",
		Version:            "0.1.0",
		RequiredAPIVersion: "0.2.0",
		EventSource:        "video",
	}
}

// Init initializes this plugin with a given config string, which is unused
// in this example. This method is mandatory for source plugins.
func (m *VideoPlugin) Init(config string) error {
	cfg := initConfig{}

	err := json.Unmarshal([]byte(config), &cfg)
	if err != nil {
		return err
	}
	m.cfg = &cfg

	return nil
}

// Fields return the list of extractor fields exported by this plugin.
// This method is optional for source plugins, and enables the extraction
// capabilities. If the Fields method is defined, the framework expects
// an Extract method to be specified too.
func (m *VideoPlugin) Fields() []sdk.FieldEntry {
	return []sdk.FieldEntry{
		{Type: "uint64", Name: "entity.count", Display: "Entities count", Desc: "Number of entities in the scene, use entity.count[<type>] to count a specific entity type (eg. cat, dog)"},
	}
}

// This method is optional for source plugins, and enables the extraction
// capabilities. If the Extract method is defined, the framework expects
// an Fields method to be specified too.
func (m *VideoPlugin) Extract(req sdk.ExtractRequest, evt sdk.EventReader) error {
	var payload *VideoEvent
	encoder := gob.NewDecoder(evt.Reader())
	if err := encoder.Decode(payload); err != nil {
		return err
	}

	switch req.FieldID() {
	case 0: // entity.count
		req.SetValue(0) // todo
		return nil
	// case 1:
	// 	return nil
	default:
		return fmt.Errorf("unsupported field: %s", req.Field())
	}
}

// Open opens the plugin source and starts a new capture session (e.g. stream
// of events), creating a new plugin instance.
func (m *VideoPlugin) Open(params string) (source.Instance, error) {
	backend := gocv.ParseNetBackend(m.cfg.Backend)
	target := gocv.ParseNetTarget(m.cfg.Target)

	// open capture device (webcam or file)
	var (
		capture *gocv.VideoCapture
		err     error
	)

	// If it is a number, open a video capture from webcam, else from file
	id, err := strconv.Atoi(m.cfg.VideoSource)
	if err == nil {
		capture, err = gocv.OpenVideoCapture(id)
	} else {
		capture, err = gocv.VideoCaptureFile(m.cfg.VideoSource)
	}
	if err != nil {
		return nil, err
	}

	// open DNN object tracking model
	net := gocv.ReadNet(m.cfg.Model, m.cfg.NetConfig)
	if net.Empty() {
		return nil, fmt.Errorf("Error reading network model from : %v %v", m.cfg.Model, m.cfg.NetConfig)
	}
	net.SetPreferableBackend(backend)
	net.SetPreferableTarget(target)

	return &VideoInstance{
		capture: capture,
		net:     &net,
	}, nil
}

// String produces a string representation of an event data produced by the
// event source of this plugin. This method is mandatory for source plugins.
func (m *VideoPlugin) String(in io.ReadSeeker) (string, error) {
	var payload *VideoEvent
	encoder := gob.NewDecoder(in)
	if err := encoder.Decode(payload); err != nil {
		return "", err
	}
	return fmt.Sprintf("%#v", *payload), nil
}

// NextBatch produces a batch of new events, and is called repeatedly by the
// framework. For source plugins, it's mandatory to specify a NextBatch method.
// The batch has a maximum size that dependes on the size of the underlying
// reusable memory buffer. A batch can be smaller than the maximum size.
func (m *VideoInstance) NextBatch(pState sdk.PluginState, evts sdk.EventWriters) (int, error) {
	var n int
	var evt sdk.EventWriter
	for n = 0; n < evts.Len(); n++ {
		evt = evts.Get(n)
		encoder := gob.NewEncoder(evt.Writer())
		payload := VideoEvent{} // todo(leogr): fill the payload
		if err := encoder.Encode(&payload); err != nil {
			return 0, err
		}
		evt.SetTimestamp(uint64(time.Now().UnixNano()))
	}
	return n, nil
}

// Progress returns a percentage indicator referring to the production progress
// of the event source of this plugin.
// This method is optional for source plugins. If specified, the following
// package needs to be imported to advise the SDK to enable this feature:
// import _ "github.com/falcosecurity/plugin-sdk-go/pkg/sdk/symbols/progress"
// func (m *VideoInstance) Progress(pState sdk.PluginState) (float64, string) {
//
// }

func (m *VideoInstance) Close() {
	m.capture.Close()
	m.net.Close()
}

// Destroy is gets called by the SDK when the plugin gets deinitialized.
// This is useful to release any open resource used by the plugin.
// This method is optional for source plugins.
// func (m *VideoPlugin) Destroy() {
//
// }

// todo(leogr): temp disabled to avoid conflict with main.go
// func main() {}
