package tensor

import (
	"fmt"
	"math"
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/rand"
)

type Number interface {
	~int | ~float64
}

type Tensor[T Number] struct {
	Shape  []int
	Stride []int
	Data   []T
}

// New returns a new tensor with the given shape and data.
func New[T Number](shape []int, data []T) *Tensor[T] {
	return &Tensor[T]{
		Shape:  shape,
		Stride: stride(shape...),
		Data:   data,
	}
}

// Full returns a new tensor with elements that are all the given value.
func Full[T Number](shape []int, value T) *Tensor[T] {
	return F(Zero[T](shape...), func(_ T) T { return value })
}

// Rand returns a new tensor with elements that pseudo-random number in the half-open interval [0.0,1.0).
func Rand(shape []int, s ...randv2.Source) *Tensor[float64] {
	return F(Zero[float64](shape...), func(_ float64) float64 { return rnd(s...).Float64() })
}

// Randn returns a new tensor with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
func Randn(shape []int, s ...randv2.Source) *Tensor[float64] {
	return F(Zero[float64](shape...), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

// Zero returns a new tensor with elements that are all zero.
func Zero[T Number](shape ...int) *Tensor[T] {
	return New(shape, make([]T, size(shape)))
}

// ZeroLike returns a new tensor with the same shape as v and elements that are all zero.
func ZeroLike[T Number](v *Tensor[T]) *Tensor[T] {
	return Zero[T](v.Shape...)
}

// OneLike returns a new tensor with the same shape as v and elements that are all one.
func OneLike[T Number](v *Tensor[T]) *Tensor[T] {
	return F(ZeroLike(v), func(_ T) T { return 1 })
}

// Reshape returns a new tensor with the same data as v with the given shape.
func Reshape[T Number](v *Tensor[T], shape ...int) *Tensor[T] {
	if size(shape) != v.Size() {
		panic("invalid shape")
	}

	return New(shape, v.Data)
}

// Take returns a new tensor with elements selected from the given indices along the specified axis.
func Take[T Number](v *Tensor[T], indices []int, axis int) *Tensor[T] {
	ndim := v.NumDims()
	axis, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	index, err := adjIndices(indices, v.Shape, axis)
	if err != nil {
		panic(err)
	}

	outShape := make([]int, ndim)
	copy(outShape, v.Shape)
	outShape[axis] = len(indices)

	out := &Tensor[T]{
		Shape:  outShape,
		Stride: stride(outShape...),
		Data:   make([]T, size(outShape)),
	}

	for i := range len(out.Data) {
		coords := Unravel(out, i)
		coords[axis] = index[coords[axis]]
		out.Data[i] = v.Data[Ravel(v, coords...)]
	}

	return out
}

// Clone returns a copy of the tensor.
func (v *Tensor[T]) Clone() *Tensor[T] {
	data := make([]T, len(v.Data))
	copy(data, v.Data)
	return New(v.Shape, data)
}

// Float64 returns a new tensor with elements casted to float64.
func (v *Tensor[T]) Float64() *Tensor[float64] {
	data := make([]float64, len(v.Data))
	for i, x := range v.Data {
		data[i] = float64(x)
	}

	return New(v.Shape, data)
}

// NumDims returns the number of dimensions of the tensor.
func (v *Tensor[T]) NumDims() int {
	return len(v.Shape)
}

// Size returns the number of elements in the tensor.
func (v *Tensor[T]) Size() int {
	return len(v.Data)
}

// At returns the element at the given index.
func (v *Tensor[T]) At(coord ...int) T {
	return v.Data[Ravel(v, coord...)]
}

// Set sets the element at the given index to the given value.
func (v *Tensor[T]) Set(coord []int, value T) {
	v.Data[Ravel(v, coord...)] = value
}

// AddAt adds the given value to the element at the given index.
func (v *Tensor[T]) AddAt(coord []int, value T) {
	v.Data[Ravel(v, coord...)] += value
}

// ScatterAdd adds the elements of w to v at the given indices.
func (v *Tensor[T]) ScatterAdd(w *Tensor[T], indices []int, axis int) {
	ndim := v.NumDims()
	axis, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	if w.Shape[axis] != len(indices) {
		panic(fmt.Sprintf("indices length=%v are not equal to shape[%d]=%d", len(indices), axis, w.Shape[axis]))
	}

	index, err := adjIndices(indices, v.Shape, axis)
	if err != nil {
		panic(err)
	}

	for i := range len(w.Data) {
		coord := Unravel(w, i)
		coord[axis] = index[coord[axis]]
		v.Data[Ravel(v, coord...)] += w.Data[i]
	}
}

// Equal returns true if the two tensors are equal.
func Equal(v, w *Tensor[int]) bool {
	if !equal(v.Shape, w.Shape) {
		return false
	}

	for i := range v.Data {
		if v.Data[i] != w.Data[i] {
			return false
		}
	}

	return true
}

// IsClose returns true if the two tensors are close enough.
func IsClose(v, w *Tensor[float64], atol, rtol float64) bool {
	if !equal(v.Shape, w.Shape) {
		return false
	}

	for i := range v.Data {
		a, b := v.Data[i], w.Data[i]
		if math.Abs(a-b) > atol+rtol*math.Abs(b) {
			return false
		}
	}

	return true
}

// AddC applies c + v for each element in v and returns a new tensor.
func AddC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c + a })
}

// SubC applies c - v for each element in v and returns a new tensor.
func SubC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c - a })
}

// MulC applies c * v for each element in v and returns a new tensor.
func MulC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c * a })
}

// Pow applies v**p for each element in v and returns a new tensor.
func Pow(p float64, v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Pow(a, p) })
}

// Exp applies exp(v) for each element in v and returns a new tensor.
func Exp(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

// Log applies log(v) for each element in v and returns a new tensor.
func Log(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Log(a) })
}

// Sin applies sin(v) for each element in v and returns a new tensor.
func Sin(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

// Cos applies cos(v) for each element in v and returns a new tensor.
func Cos(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

// Tanh applies tanh(v) for each element in v and returns a new tensor.
func Tanh(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Tanh(a) })
}

// Add applies v + w for each element in v and returns a new tensor.
func Add[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a + b })
}

// Sub applies v - w for each element in v and returns a new tensor.
func Sub[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a - b })
}

// Mul applies v * w for each element in v and returns a new tensor.
func Mul[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a * b })
}

// Div applies v / w for each element in v and returns a new tensor.
func Div[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a / b })
}

// Sum returns the sum of all elements in v.
// If axes is specified, it reduces along the given axes.
func Sum[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	return Reduce(v, 0, func(acc, x T) T { return acc + x }, axes...)
}

// Max returns the maximum value among all elements in v.
// If axes is specified, it reduces along the given axes.
func Max(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Reduce(v, -math.MaxFloat64, func(acc, x float64) float64 {
		if x > acc {
			return x
		}

		return acc
	}, axes...)
}

// Min returns the minimum value among all elements in v.
// If axes is specified, it reduces along the given axes.
func Min(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Reduce(v, math.MaxFloat64, func(acc, x float64) float64 {
		if x < acc {
			return x
		}

		return acc
	}, axes...)
}

// Mean returns the mean of all elements in v.
// If axes is specified, it reduces along the given axes.
func Mean(v *Tensor[float64], axes ...int) *Tensor[float64] {
	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		return v.Clone()
	}

	if len(axes) == 0 {
		// mean all
		return MulC(1/float64(v.Size()), Sum(v))
	}

	seen := make(map[int]bool, len(axes))
	for _, a := range axes {
		a, err := adjAxis(a, ndim)
		if err != nil {
			panic(err)
		}

		if seen[a] {
			panic(fmt.Sprintf("duplicate axis=%v", a))
		}

		seen[a] = true
	}

	// count
	count := 1
	for a := range seen {
		count = count * v.Shape[a]
	}

	// mean
	return MulC(1/float64(count), Sum(v, axes...))
}

// Argmax returns the indices of the maximum values along the given axis.
func Argmax[T Number](v *Tensor[T], axis int) *Tensor[int] {
	ndim := v.NumDims()
	axis, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	// NOTE: Consider Transpose implementation.
	// NOTE: Consider a view implementation for the Transpose.

	// out tensor
	outShape := make([]int, 0, ndim-1)
	for i, s := range v.Shape {
		if i == axis {
			continue
		}

		outShape = append(outShape, s)
	}
	out := Zero[int](outShape...)

	for i := range out.Data {
		// index -> outCoord
		outCoord := Unravel(out, i)

		// outCoord -> inCoord
		inCoord := make([]int, ndim)
		var idx int
		for j := range ndim {
			if j == axis {
				continue
			}

			inCoord[j] = outCoord[idx]
			idx++
		}

		// find max
		maxVal, maxIdx := v.At(inCoord...), 0
		for j := 1; j < v.Shape[axis]; j++ {
			inCoord[axis] = j
			if val := v.At(inCoord...); val > maxVal {
				maxVal, maxIdx = val, j
			}
		}

		out.Data[i] = maxIdx
	}

	return out
}

// Mask returns a new tensor with elements that 1 if f() is true and 0 otherwise.
func Mask[T Number](v *Tensor[T], f func(x T) bool) *Tensor[T] {
	return F(v, func(x T) T {
		if f(x) {
			return 1
		}

		return 0
	})
}

// Clip returns a new tensor with elements that are clipped to the interval [min, max].
func Clip[T Number](v *Tensor[T], min, max T) *Tensor[T] {
	return F(v, func(x T) T {
		if x < min {
			return min
		}

		if x > max {
			return max
		}

		return x
	})
}

// Transpose returns a new tensor with the axes transposed.
func Transpose[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		return v.Clone()
	}

	transpose := func(perm ...int) *Tensor[T] {
		// new shape
		shape := make([]int, ndim)
		for i, a := range perm {
			shape[i] = v.Shape[a]
		}

		// stride
		newStride := stride(shape...)
		oldStride := make([]int, ndim)
		for i := range ndim {
			oldStride[i] = v.Stride[perm[i]]
		}

		// new data
		data := make([]T, len(v.Data))
		for i := range len(v.Data) {
			k, remain := 0, i
			for j := range ndim {
				k += (remain / newStride[j]) * oldStride[j]
				remain %= newStride[j]
			}

			data[i] = v.Data[k]
		}

		return New(shape, data)
	}

	if len(axes) == 0 {
		// reverse
		perm := make([]int, ndim)
		for i := range ndim {
			perm[i] = (ndim - 1) - i
		}

		return transpose(perm...)
	}

	if len(axes) != ndim {
		panic(fmt.Sprintf("axes length=%v are not equal to ndim=%v", len(axes), ndim))
	}

	seen, perm := make([]bool, len(axes)), make([]int, len(axes))
	for i, a := range axes {
		a, err := adjAxis(a, ndim)
		if err != nil {
			panic(err)
		}

		if seen[a] {
			panic(fmt.Sprintf("duplicate axis=%v", a))
		}

		seen[a], perm[i] = true, a
	}

	return transpose(perm...)
}

// Squeeze returns a new tensor with the given axes removed.
// If axes is empty, all axes with size 1 are removed.
func Squeeze[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	if len(axes) == 0 {
		var shape []int
		for _, s := range v.Shape {
			if s == 1 {
				continue
			}

			shape = append(shape, s)
		}

		return New(shape, v.Data)
	}

	seen, ndim := make(map[int]bool), v.NumDims()
	for _, a := range axes {
		a, err := adjAxis(a, ndim)
		if err != nil {
			panic(err)
		}

		if v.Shape[a] == 1 {
			seen[a] = true
			continue
		}

		panic(fmt.Sprintf("axis=%v is not 1 (shape %v)", a, v.Shape))
	}

	var shape []int
	for i, s := range v.Shape {
		if seen[i] {
			continue
		}

		shape = append(shape, s)
	}

	return New(shape, v.Data)
}

// Expand returns a new tensor with a new axis inserted at the given position.
func Expand[T Number](v *Tensor[T], axis int) *Tensor[T] {
	ndim := v.NumDims()
	axis, err := adjAxis(axis, ndim+1)
	if err != nil {
		panic(err)
	}

	// insert 1 at axis
	shape := make([]int, ndim+1)
	copy(shape[:axis], v.Shape[:axis])
	shape[axis] = 1
	copy(shape[axis+1:], v.Shape[axis:])

	return New(shape, v.Data)
}

// Broadcast returns new tensors by broadcasting v and w to a common shape.
func Broadcast[T Number](v, w *Tensor[T], keepLast ...int) (*Tensor[T], *Tensor[T]) {
	s0, s1, err := broadcast(v.Shape, w.Shape, keepLast...)
	if err != nil {
		panic(err)
	}

	return BroadcastTo(v, s0...), BroadcastTo(w, s1...)
}

// BroadcastTo returns a new tensor with the given shape by broadcasting v to the shape.
func BroadcastTo[T Number](v *Tensor[T], shape ...int) *Tensor[T] {
	out := Zero[T](shape...)
	outNDim := out.NumDims()

	ndim := v.NumDims()
	if outNDim < ndim {
		panic(fmt.Sprintf("shape %v is smaller than tensor shape %v", shape, v.Shape))
	}

	for i := range ndim {
		s0, s1 := v.Shape[ndim-1-i], shape[outNDim-1-i]
		if s0 != s1 && s0 != 1 {
			panic(fmt.Sprintf("shape %v is not compatible with tensor shape %v", shape, v.Shape))
		}
	}

	for i := range len(out.Data) {
		k, remain := 0, i
		for a := range outNDim {
			coord := remain / out.Stride[a]
			remain %= out.Stride[a]

			j := a - (outNDim - ndim)
			if j < 0 {
				// implicit leading dimension (=1) in original; always pick index 0
				continue
			}

			if v.Shape[j] == 1 {
				// broadcast dimension -> always 0
				continue
			}

			k += coord * v.Stride[j]
		}

		out.Data[i] = v.Data[k]
	}

	return out
}

// MatMul returns the matrix multiplication of v and w.
func MatMul[T Number](v, w *Tensor[T]) *Tensor[T] {
	a, b := Broadcast(v, w, 2)
	ndim := a.NumDims()

	arows, acols := a.Shape[ndim-2], a.Shape[ndim-1]
	brows, bcols := b.Shape[ndim-2], b.Shape[ndim-1]
	if acols != brows {
		panic(fmt.Sprintf("shapes %v and %v are not aligned for matmul", v.Shape, w.Shape))
	}

	// offset
	offset := func(batch int, shape, stride []int) int {
		var v int
		for i := len(shape) - 1; i >= 0; i-- {
			idx := batch % shape[i]
			batch /= shape[i]

			v += idx * stride[i]
		}

		return v
	}

	// out tensor
	batch := a.Shape[:ndim-2]
	shape := append(batch, []int{arows, bcols}...)
	out := Zero[T](shape...)

	// batch matmul
	for batchIdx := range size(batch) {
		offseta := offset(batchIdx, batch, a.Stride[:ndim-2])
		offsetb := offset(batchIdx, batch, b.Stride[:ndim-2])
		offseto := offset(batchIdx, batch, out.Stride[:ndim-2])

		// matmul
		for i := range arows {
			for j := range bcols {
				var sum T
				for k := range acols {
					ai := offseta + i*a.Stride[ndim-2] + k*a.Stride[ndim-1]
					bi := offsetb + k*b.Stride[ndim-2] + j*b.Stride[ndim-1]
					sum += a.Data[ai] * b.Data[bi]
				}

				// set
				idx := offseto + i*out.Stride[ndim-2] + j*out.Stride[ndim-1]
				out.Data[idx] = sum
			}
		}
	}

	return out
}

// F applies the function f to each element of the tensor v and returns a new tensor.
func F[T Number](v *Tensor[T], f func(a T) T) *Tensor[T] {
	out := ZeroLike(v)
	for i := range v.Data {
		out.Data[i] = f(v.Data[i])
	}

	return out
}

// F2 applies the function f to each element of the tensors v and w and returns a new tensor.
// v and w are broadcasted to a common shape.
func F2[T Number](v, w *Tensor[T], f func(a, b T) T) *Tensor[T] {
	a, b := Broadcast(v, w)

	out := ZeroLike(a)
	for i := range a.Data {
		out.Data[i] = f(a.Data[i], b.Data[i])
	}

	return out
}

// Reduce reduces the tensor v to a tensor with fewer dimensions by applying the function f along the given axes.
func Reduce[T Number](v *Tensor[T], acc T, f func(a, b T) T, axes ...int) *Tensor[T] {
	if len(axes) == 0 {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []T{acc})
	}

	seen, ndim := make(map[int]bool, len(axes)), v.NumDims()
	for _, a := range axes {
		a, err := adjAxis(a, ndim)
		if err != nil {
			panic(err)
		}

		if seen[a] {
			panic(fmt.Sprintf("duplicate axis=%v", a))
		}

		seen[a] = true
	}

	if len(seen) == ndim {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []T{acc})
	}

	// reduced layout
	outShape := make([]int, 0, ndim-len(seen))
	outNDim := make([]int, ndim)

	var pos int
	for i := range ndim {
		if seen[i] {
			outNDim[i] = -1
			continue
		}

		outShape = append(outShape, v.Shape[i])
		outNDim[i] = pos
		pos++
	}
	outStride := stride(outShape...)

	// output
	out := Full(outShape, acc)
	for x := range len(v.Data) {
		i, remain := 0, x
		for j := range ndim {
			coord := remain / v.Stride[j]
			remain = remain % v.Stride[j]

			idx := outNDim[j]
			if idx < 0 {
				// reduced axis
				continue
			}

			i += coord * outStride[idx]
		}

		// set
		out.Data[i] = f(out.Data[i], v.Data[x])
	}

	return out
}

// Ravel returns the index in the flat data slice for the given multi-dimensional coordinates.
func Ravel[T Number](v *Tensor[T], coord ...int) int {
	if len(coord) != len(v.Shape) {
		panic(fmt.Sprintf("coord length=%v are not equal to ndim=%v", len(coord), len(v.Shape)))
	}

	var idx int
	for i, c := range coord {
		if c < 0 || c >= v.Shape[i] {
			panic(fmt.Sprintf("coord=%v out of range for axis=%v (shape=%v)", c, i, v.Shape))
		}

		idx += c * v.Stride[i]
	}

	return idx
}

// Unravel returns the multi-dimensional coordinates for the given index in the flat data slice.
func Unravel[T Number](v *Tensor[T], index int) []int {
	ndim := v.NumDims()

	coord := make([]int, ndim)
	for i := range ndim {
		coord[i] = index / v.Stride[i]
		index %= v.Stride[i]
	}

	return coord
}

// rnd returns a pseudo-random number generator.
func rnd(s ...randv2.Source) *randv2.Rand {
	if len(s) == 0 || s[0] == nil {
		return randv2.New(rand.NewSource(rand.MustRead()))
	}

	return randv2.New(s[0])
}

// broadcast returns the broadcasted shape of s0 and s1.
func broadcast(s0, s1 []int, keepLast ...int) ([]int, []int, error) {
	pad := func(shape []int, length int) []int {
		diff := length - len(shape)
		out := make([]int, length)
		for i := range diff {
			out[i] = 1
		}

		copy(out[diff:], shape)
		return out
	}

	// keep last n dimensions
	var n int
	if len(keepLast) > 0 {
		n = keepLast[0]
	}

	if len(s0) < n || len(s1) < n {
		return nil, nil, fmt.Errorf("shapes %v and %v are not compatible", s0, s1)
	}

	s0, tail0 := s0[:len(s0)-n], s0[len(s0)-n:]
	s1, tail1 := s1[:len(s1)-n], s1[len(s1)-n:]

	// pad to the same length
	s0len, s1len := len(s0), len(s1)
	maxLen := max(s0len, s1len)
	s0, s1 = pad(s0, maxLen), pad(s1, maxLen)

	shape := make([]int, maxLen)
	for i := range maxLen {
		d1, d2 := s0[i], s1[i]
		switch {
		case d1 == d2, d2 == 1:
			shape[i] = d1
		case d1 == 1:
			shape[i] = d2
		default:
			return nil, nil, fmt.Errorf("shapes %v and %v are not compatible", s0, s1)
		}
	}

	// append the tail back
	return append(shape, tail0...), append(shape, tail1...), nil
}

// stride returns the stride for the given shape.
func stride(shape ...int) []int {
	n := len(shape)
	if n == 0 {
		return nil
	}

	s := make([]int, n)
	s[n-1] = 1
	for i := n - 2; i >= 0; i-- {
		s[i] = s[i+1] * shape[i+1]
	}

	return s
}

// equal returns true if the two shapes are equal.
func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// size returns the number of elements in the given shape.
func size(shape []int) int {
	size := 1
	for _, s := range shape {
		size *= s
	}

	return size
}

// adjAxis adjusts negative axis and checks the range.
func adjAxis(axis, ndim int) (int, error) {
	if axis < 0 {
		axis += ndim
	}

	if axis < 0 || axis >= ndim {
		return -1, fmt.Errorf("axis=%d out of range for ndim=%d", axis, ndim)
	}

	return axis, nil
}

// adjIndices adjusts negative indices and checks the range.
func adjIndices(indices, shape []int, axis int) ([]int, error) {
	adj := make([]int, len(indices))
	for i, idx := range indices {
		if idx < 0 {
			idx += shape[axis]
		}

		if idx < 0 || idx >= shape[axis] {
			return nil, fmt.Errorf("index %d out of range for axis=%d (shape=%v)", idx, axis, shape)
		}

		adj[i] = idx
	}

	return adj, nil
}
