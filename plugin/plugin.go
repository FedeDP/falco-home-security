package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins/extractor"
	"github.com/falcosecurity/plugin-sdk-go/pkg/sdk/plugins/source"
	"gocv.io/x/gocv"
)

type OpenConfig struct {
	VideoSource  string `json:"videoSource"`
	ShowWindow   bool   `json:"showWindow"`
	SnapshotPath string `json:"snapshotPath"`
}

type VideoPlugin struct {
	plugins.BasePlugin
	cfg *DetectionConfig
}

type VideoInstance struct {
	source.BaseInstance
	cfg        *OpenConfig
	detectionc DetectionChan
	errorc     ErrorChan
	quitc      QuitChan
	renderc    RenderChan
	window     *gocv.Window
	wg         *sync.WaitGroup
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
		ID:                  999,
		Name:                "homesecurity",
		Description:         "Video recognition plugin",
		Contact:             "github.com/FedeDP/falco-home-security",
		Version:             "0.1.0",
		RequiredAPIVersion:  "0.2.0",
		EventSource:         "homesecurity",
		ExtractEventSources: []string{"homesecurity"},
	}
}

// Init initializes this plugin with a given config string, which is unused
// in this example. This method is mandatory for source plugins.
func (m *VideoPlugin) Init(config string) error {
	cfg := DetectionConfig{
		Model:                      "",
		NetConfig:                  "",
		Backend:                    "",
		Target:                     "",
		MinConfidence:              0.75,
		MemoryMinConfidence:        0.5,
		MemoryDecayFactor:          0.98,
		MemoryNearnessThreshold:    0.65,
		MemoryClassSwitchThreshold: 0.15,
		MemoryCollapseMultiple:     true,
	}

	if len(config) == 0 {
		println("no init")
		return fmt.Errorf("you must specify an init configuration")
	}

	err := json.Unmarshal([]byte(config), &cfg)
	if err != nil {
		println(config)
		println("init: " + err.Error())
		return err
	}

	if len(cfg.Model) == 0 || len(cfg.NetConfig) == 0 {
		println("init mandatory")
		return fmt.Errorf("model and netConfig are mandatory init config parameters")
	}

	m.cfg = &cfg
	return nil
}

// Open opens the plugin source and starts a new capture session (e.g. stream
// of events), creating a new plugin instance.
func (m *VideoPlugin) Open(params string) (source.Instance, error) {
	cfg := OpenConfig{
		VideoSource:  "",
		ShowWindow:   false,
		SnapshotPath: "",
	}

	if len(params) == 0 {
		return nil, fmt.Errorf("you must specify an open configuration")
	}

	err := json.Unmarshal([]byte(params), &cfg)
	if err != nil {
		return nil, err
	}

	if len(cfg.VideoSource) == 0 {
		return nil, fmt.Errorf("videoSource is a mandatory open config parameters")
	}

	var window *gocv.Window
	if cfg.ShowWindow {
		window = gocv.NewWindow("Falco Home Security")
	}

	var wg sync.WaitGroup
	quitc := make(QuitChan, 1)
	detectionc, renderc, errorc := LaunchVideoDetection(m.cfg, &cfg, quitc, &wg)
	instance := &VideoInstance{
		cfg:        &cfg,
		detectionc: detectionc,
		renderc:    renderc,
		errorc:     errorc,
		quitc:      quitc,
		window:     window,
		wg:         &wg,
	}

	// Override event buffer
	events, err := sdk.NewEventWriters(1, int64(sdk.DefaultEvtSize))
	if err != nil {
		return nil, err
	}
	instance.SetEvents(events)

	return instance, err
}

func (m *VideoInstance) Close() {
	m.quitc <- true
	close(m.quitc)
	if m.cfg.ShowWindow {
		m.window.Close()
	}
}

// NextBatch produces a batch of new events, and is called repeatedly by the
// framework. For source plugins, it's mandatory to specify a NextBatch method.
// The batch has a maximum size that dependes on the size of the underlying
// reusable memory buffer. A batch can be smaller than the maximum size.
func (m *VideoInstance) NextBatch(pState sdk.PluginState, evts sdk.EventWriters) (int, error) {
	evt := evts.Get(0)
	writer := evt.Writer()
	timeout := time.After(time.Millisecond * 1000)
	for {
		select {
		case payload := <-m.detectionc:
			encoder := gob.NewEncoder(writer)
			if err := encoder.Encode(&payload); err != nil {
				return 0, err
			}
			evt.SetTimestamp(uint64(time.Now().UnixNano()))
			return 1, nil
		case err := <-m.errorc:
			if err == errDeviceClosed {
				return 0, sdk.ErrEOF
			}
			return 0, err
		case img := <-m.renderc:
			if m.cfg.ShowWindow {
				m.window.IMShow(img)
				if m.window.WaitKey(1) >= 0 || m.window.GetWindowProperty(gocv.WindowPropertyVisible) == 0 {
					return 0, sdk.ErrEOF
				}
			}
		case <-timeout:
			return 0, sdk.ErrTimeout
		}
	}
}

// String produces a string representation of an event data produced by the
// event source of this plugin. This method is mandatory for source plugins.
func (m *VideoPlugin) String(in io.ReadSeeker) (string, error) {
	var payload VideoEvent
	encoder := gob.NewDecoder(in)
	if err := encoder.Decode(&payload); err != nil {
		return "", err
	}
	return payload.AsciiImage, nil
}

// Fields return the list of extractor fields exported by this plugin.
// This method is optional for source plugins, and enables the extraction
// capabilities. If the Fields method is defined, the framework expects
// an Extract method to be specified too.
func (m *VideoPlugin) Fields() []sdk.FieldEntry {
	return []sdk.FieldEntry{
		{
			Type:    "uint64",
			Name:    "video.entities",
			Display: "Count of the entities detected in the scene",
			Desc:    "Number of entities in the scene, use video.entities[<type>] to count a specific entity type between { human, animal }",
		},
		{
			Type:    "string",
			Name:    "video.source",
			Display: "Name of the opened video source",
			Desc:    "Name of the opened video source.",
		},
		{
			Type:    "string",
			Name:    "video.snapshot",
			Display: "Fullpath to last snapshot stored, if any",
			Desc:    "Fullpath to last snapshot stored, if any",
		},
	}
}

// Extract is optional for source plugins, and enables the extraction
// capabilities. If the Extract method is defined, the framework expects
// a Fields method to be specified too.
func (m *VideoPlugin) Extract(req sdk.ExtractRequest, evt sdk.EventReader) error {
	var payload VideoEvent
	encoder := gob.NewDecoder(evt.Reader())
	if err := encoder.Decode(&payload); err != nil {
		return err
	}

	switch req.FieldID() {
	case 0: // video.entities
		count := uint64(len(payload.Blobs))
		if len(req.Arg()) > 0 {
			count = 0
			for _, blob := range payload.Blobs {
				if strings.EqualFold(blob.Category.String(), req.Arg()) {
					count++
				}
			}
		}
		req.SetValue(count)
	case 1: // video.source
		req.SetValue(payload.VideoSource)
	case 2: // video.snapshot
		req.SetValue(payload.SnapshotPath)
	default:
		return fmt.Errorf("unsupported field: %s", req.Field())
	}
	return nil
}
