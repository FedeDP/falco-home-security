package main

import "image/color"

// See https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

type ClassID int
type CategoryID int

const (
	Unknown    CategoryID = iota
	Human      CategoryID = iota
	Vehicle    CategoryID = iota
	Outdoor    CategoryID = iota
	Animal     CategoryID = iota
	Accessory  CategoryID = iota
	Sports     CategoryID = iota
	Kitchen    CategoryID = iota
	Food       CategoryID = iota
	Furniture  CategoryID = iota
	Electronic CategoryID = iota
	Appliance  CategoryID = iota
	Indoor     CategoryID = iota
)

type categoryRange struct {
	start int
	end   int
}

var categoryRanges = map[CategoryID]categoryRange{
	Human:      categoryRange{1, 1},
	Vehicle:    categoryRange{2, 9},
	Outdoor:    categoryRange{10, 15},
	Animal:     categoryRange{16, 25},
	Accessory:  categoryRange{26, 33},
	Sports:     categoryRange{34, 43},
	Kitchen:    categoryRange{44, 51},
	Food:       categoryRange{52, 61},
	Furniture:  categoryRange{62, 71},
	Electronic: categoryRange{72, 77},
	Appliance:  categoryRange{78, 83},
	Indoor:     categoryRange{84, 91},
}

// Categories we want to handle
var Categories = map[CategoryID]string{
	Human:  "Human",
	Animal: "Animal",
}

func (c CategoryID) String() string {
	return Categories[c]
}

func (c CategoryID) Known() bool {
	if _, ok := Categories[c]; ok {
		return true
	}
	return false
}

func ParseClassID(classId int) CategoryID {
	for c, r := range categoryRanges {
		if r.start <= classId && classId <= r.end {
			return c
		}
	}

	return Unknown
}

type BlobPosition struct {
	Left   int
	Top    int
	Right  int
	Bottom int
}

type BlobPoint struct {
	x int
	y int
}

type Blob struct {
	Category   CategoryID
	Confidence float64
	Position   BlobPosition
}

type BlobList struct {
	blobs []Blob
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (b BlobPosition) Center() BlobPoint {
	x := (b.Right - b.Left) / 2
	y := (b.Bottom - b.Top) / 2
	return BlobPoint{x, y}
}

func (b BlobPoint) Near(other BlobPoint) float64 {
	xDiff := float64(minInt(b.x, other.x)) / float64(maxInt(b.x, other.x))
	yDiff := float64(minInt(b.y, other.y)) / float64(maxInt(b.y, other.y))
	return xDiff * yDiff
}

func (b Blob) Color() color.RGBA {
	switch b.Category {
	case Human:
		return color.RGBA{B: 255}
	case Animal:
		return color.RGBA{G: 255}
	}
	return color.RGBA{}
}

// Given a new blob, returns the index of the most similar known blob.
// If no blob is similar enough, -1 is returned.
func (b *BlobList) findNearestIndex(blob Blob, merged map[int]bool, blobFindNearestThreshold float64) int {
	maxNearness := 0.0
	maxIndex := -1
	for i, blob := range b.blobs {
		nearness := blob.Position.Center().Near(blob.Position.Center())
		// The nearess value should be above a certain threshold
		if !merged[i] && nearness > blobFindNearestThreshold && nearness > maxNearness {
			maxNearness = nearness
			maxIndex = i
		}
	}
	return maxIndex
}

// Merges a new blob with a known one
func (b *BlobList) mergeAtIndex(blob Blob, index int, blobMergeConfidenceThreshold float64) bool {
	changed := false
	// If the confidence of the new blob is better than the current
	// one, both the confidence and the class are overridden.
	if blob.Confidence >= b.blobs[index].Confidence+blobMergeConfidenceThreshold {
		changed = b.blobs[index].Category != blob.Category
		b.blobs[index].Confidence = blob.Confidence
		b.blobs[index].Category = blob.Category
	}
	// The position is the mean value of all the coordinates of the two blobs
	b.blobs[index].Position.Top = (b.blobs[index].Position.Top + blob.Position.Top) / 2
	b.blobs[index].Position.Left = (b.blobs[index].Position.Left + blob.Position.Left) / 2
	b.blobs[index].Position.Bottom = (b.blobs[index].Position.Bottom + blob.Position.Bottom) / 2
	b.blobs[index].Position.Right = (b.blobs[index].Position.Right + blob.Position.Right) / 2
	return changed
}

// Decreases the confidence of all the known blobs.
// If the confidence crosses a threshold, the blob is discarded.
func (b *BlobList) refreshConfidence(blobConfidenceRefreshRatio, blobConfidenceRefreshThreshold float64) {
	var newBlobs []Blob
	for _, blob := range b.blobs {
		blob.Confidence = blob.Confidence * blobConfidenceRefreshRatio
		if blob.Confidence > blobConfidenceRefreshThreshold {
			newBlobs = append(newBlobs, blob)
		}
	}
	b.blobs = newBlobs
}

// Adds new blob observations
func (b *BlobList) Update(blobs []Blob, cfg *DetectionConfig) bool {
	changed := false

	merged := make(map[int]bool)
	b.refreshConfidence(cfg.MemoryDecayFactor, cfg.MemoryMinConfidence)
	for _, blob := range blobs {
		nearestIndex := b.findNearestIndex(blob, merged, cfg.MemoryNearnessThreshold)
		if nearestIndex < 0 {
			b.blobs = append(b.blobs, blob)
			changed = true
		} else {
			if b.mergeAtIndex(blob, nearestIndex, cfg.MemoryClassSwitchThreshold) {
				changed = true
			}
			if !cfg.MemoryCollapseMultiple {
				merged[nearestIndex] = true
			}
		}
	}
	return changed
}

// Returns the known blobs
func (b *BlobList) Blobs() []Blob {
	return b.blobs
}
