package tensor

import "slices"

type Iterator struct {
	shape   []int
	coord   []int
	offsets []int
	strides [][]int
	started bool
	done    bool
}

func NewIterator[T Number](tensors ...*Tensor[T]) *Iterator {
	shape := append([]int{}, tensors[0].Shape...)
	ndim := tensors[0].NumDims()

	strides := make([][]int, len(tensors))
	for i, t := range tensors {
		strides[i] = append([]int{}, t.Stride...)
	}

	return &Iterator{
		shape:   shape,
		coord:   make([]int, ndim),
		offsets: make([]int, len(tensors)),
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
