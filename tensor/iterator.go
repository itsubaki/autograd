package tensor

import "slices"

type Layout interface {
	NumDims() int
	Shape() []int
	Stride() []int
}

type Iterator struct {
	shape   []int
	coord   []int
	offsets []int
	strides [][]int
	started bool
	done    bool
}

func NewIterator(layouts ...Layout) *Iterator {
	shape := append([]int{}, layouts[0].Shape()...)
	ndim := layouts[0].NumDims()

	strides := make([][]int, len(layouts))
	for i, layout := range layouts {
		if !SliceEqual(layout.Shape(), shape) {
			panic("layouts have incompatible shapes")
		}

		strides[i] = append([]int{}, layout.Stride()...)
	}

	return &Iterator{
		shape:   shape,
		coord:   make([]int, ndim),
		offsets: make([]int, len(layouts)),
		strides: strides,
		done:    slices.Contains(shape, 0),
	}
}

func (it *Iterator) Next() bool {
	if it.done {
		return false
	}

	if !it.started {
		it.started = true
		return true
	}

	for axis := len(it.shape) - 1; axis >= 0; axis-- {
		it.coord[axis]++
		for i := range it.offsets {
			it.offsets[i] += it.strides[i][axis]
		}

		if it.coord[axis] < it.shape[axis] {
			return true
		}

		// reset
		it.coord[axis] = 0
		for i := range it.offsets {
			it.offsets[i] -= it.shape[axis] * it.strides[i][axis]
		}
	}

	it.done = true
	return false
}

func (it *Iterator) Offset(i int) int {
	return it.offsets[i]
}
