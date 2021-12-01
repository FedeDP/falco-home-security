package main

import "image/color"

const (
	// At each refresh cycle, blobs are discarded if their confidence goes
	// below this value.
	blobConfidenceRefreshThreshold = 0.5

	// At each refresh cycle, the confidence of each blob is reduced by
	// this factor.
	blobConfidenceRefreshRatio = 0.98

	// While searching for near blobs, this is the minimum value required
	// to consider two blob similars.
	blobFindNearestThreshold = 0.65

	// While merging a new blob with a new one, the new blob should surpass
	// the condidence of the known blob by this threshold, in order to override
	// its confidence and class values.
	blobMergeConfidenceThreshold = 0.15

	// Collapses all the near rectangles in a single one
	blobMergeCloseRectangles = true
)

// See https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
type ClassID int

const (
	Human ClassID = 1
	Cat   ClassID = 17
	Dog   ClassID = 18
)

var Classes = []ClassID{Human, Cat, Dog}

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
	Class      ClassID
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
	switch b.Class {
	case Human:
		return color.RGBA{B: 255}
	case Cat:
		return color.RGBA{G: 255}
	case Dog:
		return color.RGBA{R: 255}
	}
	return color.RGBA{}
}

// Given a new blob, returns the index of the most similar known blob.
// If no blob is similar enough, -1 is returned.
func (b *BlobList) findNearestIndex(blob Blob, merged map[int]bool) int {
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
func (b *BlobList) mergeAtIndex(blob Blob, index int) bool {
	changed := false
	// If the confidence of the new blob is better than the current
	// one, both the confidence and the class are overridden.
	if blob.Confidence >= b.blobs[index].Confidence+blobMergeConfidenceThreshold {
		changed = b.blobs[index].Class != blob.Class
		b.blobs[index].Confidence = blob.Confidence
		b.blobs[index].Class = blob.Class
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
func (b *BlobList) refreshConfidence() {
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
func (b *BlobList) Update(blobs []Blob) bool {
	changed := false
	merged := make(map[int]bool)
	b.refreshConfidence()
	for _, blob := range blobs {
		nearestIndex := b.findNearestIndex(blob, merged)
		if nearestIndex < 0 {
			b.blobs = append(b.blobs, blob)
			changed = true
		} else {
			if b.mergeAtIndex(blob, nearestIndex) {
				changed = true
			}
			if !blobMergeCloseRectangles {
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
