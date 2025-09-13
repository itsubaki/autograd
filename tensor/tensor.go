package tensor

import (
	"fmt"
	"math"
	randv2 "math/rand/v2"
	"slices"

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
	if axis < 0 {
		axis += ndim
	}

	if axis < 0 || axis >= ndim {
		panic(fmt.Sprintf("invalid axis %d for shape %v", axis, v.Shape))
	}

	indicesAdj := make([]int, len(indices))
	for i, idx := range indices {
		if idx < 0 {
			idx += v.Shape[axis]
		}

		if idx < 0 || idx >= v.Shape[axis] {
			panic(fmt.Sprintf("index %d out of range for axis=%d (shape=%v)", idx, axis, v.Shape))
		}

		indicesAdj[i] = idx
	}

	newShape := make([]int, ndim)
	copy(newShape, v.Shape)
	newShape[axis] = len(indices)

	out := &Tensor[T]{
		Shape:  newShape,
		Stride: stride(newShape...),
		Data:   make([]T, size(newShape)),
	}

	ondim := out.NumDims()
	coords := make([]int, ondim)
	for i := range len(out.Data) {
		remain := i
		for j := range ondim {
			coords[j] = remain / out.Stride[j]
			remain = remain % out.Stride[j]
		}

		coord := make([]int, len(coords))
		copy(coord, coords)
		coord[axis] = indicesAdj[coords[axis]]

		src := Ravel(v, coord...)
		dst := Ravel(out, coords...)
		out.Data[dst] = v.Data[src]
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
func Max[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	return Reduce(v, slices.Min(v.Data), func(acc, x T) T {
		if x > acc {
			return x
		}

		return acc
	}, axes...)
}

// Min returns the minimum value among all elements in v.
// If axes is specified, it reduces along the given axes.
func Min[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	return Reduce(v, slices.Max(v.Data), func(acc, x T) T {
		if x < acc {
			return x
		}

		return acc
	}, axes...)
}

// Mean returns the mean of all elements in v.
// If axes is specified, it reduces along the given axes.
func Mean(v *Tensor[float64], axes ...int) *Tensor[float64] {
	if len(axes) == 0 {
		return MulC(1/float64(v.Size()), Sum(v))
	}

	// validate
	validated, _, err := validate(v, axes...)
	if err != nil {
		panic(err)
	}

	// count
	count := 1
	for a := range validated {
		count = count * v.Shape[a]
	}

	// mean
	return MulC(1/float64(count), Sum(v, axes...))
}

// Argmax returns the indices of the maximum values along the given axis.
func Argmax[T Number](v *Tensor[T], axis int) *Tensor[int] {
	ndim := v.NumDims()
	if axis < 0 {
		axis += ndim
	}

	if axis < 0 || axis >= ndim {
		panic(fmt.Sprintf("axis %d is out of bounds for tensor with %d dimensions", axis, ndim))
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

	outCoord, inCoord := make([]int, out.NumDims()), make([]int, v.NumDims())
	for i := range out.Data {
		// index -> outCoord
		remain := i
		for j := range out.NumDims() {
			outCoord[j] = remain / out.Stride[j]
			remain %= out.Stride[j]
		}

		// outCoord -> inCoord
		var idx int
		for j := range ndim {
			if j == axis {
				continue
			}

			inCoord[j] = outCoord[idx]
			idx++
		}

		// find max
		inCoord[axis] = 0
		maxVal, maxIdx := v.At(inCoord...), 0

		// loop along the axis
		for j := 1; j < v.Shape[axis]; j++ {
			inCoord[axis] = j
			currVal := v.At(inCoord...)
			if currVal > maxVal {
				maxVal, maxIdx = currVal, j
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
		return v
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

	perm, seen := make([]int, len(axes)), make([]bool, ndim)
	for i, a := range axes {
		if a < 0 {
			a += ndim
		}

		if a < 0 || a >= ndim {
			panic("axis out of range")
		}

		if seen[a] {
			panic("duplicate axis")
		}

		perm[i], seen[a] = a, true
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

	ndim, seen := v.NumDims(), make(map[int]bool)
	for _, a := range axes {
		if a < 0 {
			a += ndim
		}

		if a < 0 || a >= ndim {
			panic("axis out of range")
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
	if axis < 0 {
		axis += ndim + 1
	}

	if axis < 0 || axis > ndim {
		panic(fmt.Sprintf("axis %v out of range for tensor with %v dims", axis, ndim))
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

	ndim, vndim := out.NumDims(), v.NumDims()
	if ndim < vndim {
		panic(fmt.Sprintf("shape %v is smaller than tensor shape %v", shape, v.Shape))
	}

	for i := range vndim {
		s0, s1 := v.Shape[vndim-1-i], shape[ndim-1-i]
		if s0 != s1 && s0 != 1 {
			panic(fmt.Sprintf("shape %v is not compatible with tensor shape %v", shape, v.Shape))
		}
	}

	for i := range len(out.Data) {
		k, remain := 0, i
		for a := range ndim {
			coord := remain / out.Stride[a]
			remain %= out.Stride[a]

			j := a - (ndim - vndim)
			if j < 0 {
				// implicit leading dimension (=1) in original; always pick index 0
				continue
			}

			if s := v.Shape[j]; s == 1 {
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
		panic("incompatible matrix shapes")
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

	// validate
	validated, ndim, err := validate(v, axes...)
	if err != nil {
		panic(err)
	}

	if len(validated) == ndim {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []T{acc})
	}

	// reduced layout
	outShape := make([]int, 0, ndim-len(validated))
	outNdim := make([]int, ndim)

	var pos int
	for i := range ndim {
		if validated[i] {
			outNdim[i] = -1
			continue
		}

		outShape = append(outShape, v.Shape[i])
		outNdim[i] = pos
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

			idx := outNdim[j]
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
		panic("invalid number of coord")
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

// validate validates the given axes and returns a map of the axes to be reduced.
func validate[T Number](v *Tensor[T], axis ...int) (map[int]bool, int, error) {
	ndim := v.NumDims()
	if ndim == 0 {
		return nil, 0, fmt.Errorf("axis out of range (ndim=%v)", ndim)
	}

	seen := make(map[int]bool, len(axis))
	for _, a := range axis {
		if a < 0 || a >= ndim {
			return nil, ndim, fmt.Errorf("axis %v out of range (ndim=%v)", a, ndim)
		}

		if seen[a] {
			return nil, ndim, fmt.Errorf("duplicate axis %v", a)
		}

		seen[a] = true
	}

	return seen, ndim, nil
}

// broadcast returns the broadcasted shape of s0 and s1.
func broadcast(s0, s1 []int, keepLast ...int) ([]int, []int, error) {
	pad := func(shape []int, length int) []int {
		diff := length - len(shape)
		newShape := make([]int, length)
		for i := range diff {
			newShape[i] = 1
		}

		copy(newShape[diff:], shape)
		return newShape
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
