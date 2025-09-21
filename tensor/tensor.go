package tensor

import (
	"fmt"
	"iter"
	"math"
	randv2 "math/rand/v2"
	"runtime"
	"sync"

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
	return F(Zeros[T](shape...), func(_ T) T { return value })
}

// Rand returns a new tensor with elements that pseudo-random number in the half-open interval [0.0,1.0).
func Rand(shape []int, s ...randv2.Source) *Tensor[float64] {
	return F(Zeros[float64](shape...), func(_ float64) float64 { return rnd(s...).Float64() })
}

// Randn returns a new tensor with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
func Randn(shape []int, s ...randv2.Source) *Tensor[float64] {
	return F(Zeros[float64](shape...), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

// Zeros returns a new tensor with elements that are all zero.
func Zeros[T Number](shape ...int) *Tensor[T] {
	return New(shape, make([]T, size(shape)))
}

// Ones returns a new tensor with elements that are all one.
func Ones[T Number](shape ...int) *Tensor[T] {
	return F(Zeros[T](shape...), func(_ T) T { return 1 })
}

// ZeroLike returns a new tensor with the same shape as v and elements that are all zero.
func ZeroLike[T Number](v *Tensor[T]) *Tensor[T] {
	return Zeros[T](v.Shape...)
}

// OneLike returns a new tensor with the same shape as v and elements that are all one.
func OneLike[T Number](v *Tensor[T]) *Tensor[T] {
	return F(ZeroLike(v), func(_ T) T { return 1 })
}

// Arange returns a new tensor with evenly spaced values within a given interval.
func Arange[T Number](start, stop T, step ...T) *Tensor[T] {
	var s T = 1
	if len(step) != 0 {
		s = step[0]
	}

	if s > 0 {
		var data []T
		for i := start; i < stop; i += s {
			data = append(data, i)
		}

		return New([]int{len(data)}, data)
	}

	var data []T
	for i := start; i > stop; i += s {
		data = append(data, i)
	}

	return New([]int{len(data)}, data)
}

// Linspace returns a new tensor with n evenly spaced samples, calculated over the interval [start, stop].
func Linspace(start, stop float64, n int) *Tensor[float64] {
	if n < 2 {
		panic("n is less than 2")
	}

	step := (stop - start) / float64(n-1)
	data := make([]float64, n)
	for i := range n {
		data[i] = start + float64(i)*step
	}

	return New([]int{n}, data)
}

// Identity returns a new tensor with ones on the diagonal and zeros elsewhere.
func Identity[T Number](rows, cols int) *Tensor[T] {
	data := make([]T, rows*cols)
	for i := 0; i < rows && i < cols; i++ {
		data[i*cols+i] = 1
	}

	return New([]int{rows, cols}, data)
}

// Eye returns a new tensor with ones on the diagonal and zeros elsewhere.
func Eye[T Number](n int) *Tensor[T] {
	return Identity[T](n, n)
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

// Seq2 returns a sequence of rows.
func (v *Tensor[T]) Seq2() iter.Seq2[int, []T] {
	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		return func(yield func(int, []T) bool) {
			yield(0, v.Data)
		}
	}

	size := v.Shape[ndim-1]
	total := len(v.Data) / size

	return func(yield func(int, []T) bool) {
		for i := range total {
			start := i * size
			if !yield(i, v.Data[start:start+size]) {
				return
			}
		}
	}
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

// Sqrt applies sqrt(v) for each element in v and returns a new tensor.
func Sqrt(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Sqrt(a) })
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
		return Clone(v)
	}

	if len(axes) == 0 {
		// mean all
		return MulC(1/float64(v.Size()), Sum(v))
	}

	ax, _, err := adjAxes(ndim, axes...)
	if err != nil {
		panic(err)
	}

	// count
	count := 1
	for _, a := range ax {
		count = count * v.Shape[a]
	}

	// mean
	return MulC(1/float64(count), Sum(v, ax...))
}

// Variance returns a new tensor with the variance of elements in v.
func Variance(v *Tensor[float64], axes ...int) *Tensor[float64] {
	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		return New(nil, []float64{0})
	}

	// insert 1 at the given axes
	shape := make([]int, ndim)
	copy(shape, v.Shape)
	for _, ax := range axes {
		ax, err := adjAxis(ax, ndim)
		if err != nil {
			panic(err)
		}

		shape[ax] = 1
	}

	mu := Mean(v, axes...)
	xc := Sub(v, Reshape(mu, shape...))
	return Mean(Mul(xc, xc), axes...)
}

// Std returns a new tensor with the standard deviation of elements in v.
func Std(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Sqrt(Variance(v, axes...))
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

// Mask returns a new tensor with elements that 1 if f() is true and 0 otherwise.
func Mask[T Number](v *Tensor[T], f func(x T) bool) *Tensor[T] {
	return F(v, func(x T) T {
		if f(x) {
			return 1
		}

		return 0
	})
}

// Equal returns a new tensor with elements that are 1 if v and w are equal and 0 otherwise.
func Equal(v, w *Tensor[int]) *Tensor[int] {
	return F2(v, w, func(a, b int) int {
		if a == b {
			return 1
		}

		return 0
	})
}

// IsClose returns a new tensor with elements that are 1 if v and w are close enough and 0 otherwise.
func IsClose(v, w *Tensor[float64], atol, rtol float64) *Tensor[int] {
	return F2(v, w, func(a, b float64) int {
		if isClose(a, b, atol, rtol) {
			return 1
		}

		return 0
	})
}

// Argmax returns the indices of the maximum values along the given axis.
func Argmax[T Number](v *Tensor[T], axis int) *Tensor[int] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	// NOTE: Consider Transpose implementation.
	// NOTE: Consider a view implementation for the Transpose.

	// out tensor
	shape := make([]int, 0, ndim-1)
	for i, s := range v.Shape {
		if i == ax {
			continue
		}

		shape = append(shape, s)
	}
	out := Zeros[int](shape...)

	for i := range out.Data {
		coord := Unravel(out, i)

		vcoord := make([]int, ndim)
		var idx int
		for j := range ndim {
			if j == ax {
				continue
			}

			vcoord[j] = coord[idx]
			idx++
		}

		// find max
		maxVal, maxIdx := v.At(vcoord...), 0
		for j := 1; j < v.Shape[ax]; j++ {
			vcoord[ax] = j
			if val := v.At(vcoord...); val > maxVal {
				maxVal, maxIdx = val, j
			}
		}

		out.Data[i] = maxIdx
	}

	return out
}

// Clone returns a copy of the tensor.
func Clone[T Number](v *Tensor[T]) *Tensor[T] {
	data := make([]T, len(v.Data))
	copy(data, v.Data)
	return New(v.Shape, data)
}

// Float64 returns a new tensor with elements casted to float64.
func Float64(v *Tensor[int]) *Tensor[float64] {
	data := make([]float64, len(v.Data))
	for i, x := range v.Data {
		data[i] = float64(x)
	}

	return New(v.Shape, data)
}

// Reshape returns a new tensor with the same data as v with the given shape.
func Reshape[T Number](v *Tensor[T], shape ...int) *Tensor[T] {
	if size(shape) != v.Size() {
		panic("invalid shape")
	}

	return New(shape, v.Data)
}

// Flatten returns a new tensor with the same data as v with shape (v.Size(),).
func Flatten[T Number](v *Tensor[T]) *Tensor[T] {
	return Reshape(v, v.Size())
}

// Take returns a new tensor with elements selected from the given indices along the specified axis.
func Take[T Number](v *Tensor[T], indices []int, axis int) *Tensor[T] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	idx, err := adjIndices(indices, v.Shape, ax)
	if err != nil {
		panic(err)
	}

	// out tensor
	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[ax] = len(indices)
	out := Zeros[T](shape...)

	// take
	for i := range out.Data {
		coords := Unravel(out, i)
		coords[ax] = idx[coords[ax]]
		out.Data[i] = v.Data[Ravel(v, coords...)]
	}

	return out
}

// ScatterAdd return a new tensor with elements added from w at the given indices along the specified axis.
func ScatterAdd[T Number](v, w *Tensor[T], indices []int, axis int) *Tensor[T] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	if w.Shape[ax] != len(indices) {
		panic(fmt.Sprintf("indices length=%v are not equal to shape[%d]=%d", len(indices), ax, w.Shape[ax]))
	}

	idx, err := adjIndices(indices, v.Shape, ax)
	if err != nil {
		panic(err)
	}

	out := Clone(v)
	for i := range w.Data {
		coord := Unravel(w, i)
		coord[ax] = idx[coord[ax]]
		out.Data[Ravel(out, coord...)] += w.Data[i]
	}

	return out
}

// Transpose returns a new tensor with the axes transposed.
func Transpose[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		return Clone(v)
	}

	transpose := func(perm ...int) *Tensor[T] {
		old := make([]int, ndim)
		for i := range ndim {
			old[i] = v.Stride[perm[i]]
		}

		// out tensor
		shape := make([]int, ndim)
		for i, a := range perm {
			shape[i] = v.Shape[a]
		}
		out := Zeros[T](shape...)

		// transpose
		for i := range v.Data {
			k, remain := 0, i
			for j := range ndim {
				k += (remain / out.Stride[j]) * old[j]
				remain %= out.Stride[j]
			}

			out.Data[i] = v.Data[k]
		}

		return out
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

	perm, _, err := adjAxes(ndim, axes...)
	if err != nil {
		panic(err)
	}

	return transpose(perm...)
}

// Flip returns a new tensor with the elements reversed along the given axes.
func Flip[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	ndim := v.NumDims()
	if len(axes) == 0 {
		axes = make([]int, ndim)
		for i := range ndim {
			axes[i] = i
		}
	}

	ax, _, err := adjAxes(ndim, axes...)
	if err != nil {
		panic(err)
	}

	out := ZeroLike(v)
	for i := range v.Data {
		coord := Unravel(v, i)
		for _, a := range ax {
			coord[a] = v.Shape[a] - 1 - coord[a]
		}

		out.Data[i] = v.Data[Ravel(v, coord...)]
	}

	return out
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
		ax, err := adjAxis(a, ndim)
		if err != nil {
			panic(err)
		}

		if v.Shape[ax] == 1 {
			seen[ax] = true
			continue
		}

		panic(fmt.Sprintf("axis=%v is not 1 (shape %v)", ax, v.Shape))
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
	ax, err := adjAxis(axis, ndim+1)
	if err != nil {
		panic(err)
	}

	// insert 1 at axis
	shape := make([]int, ndim+1)
	copy(shape[:ax], v.Shape[:ax])
	shape[ax] = 1
	copy(shape[ax+1:], v.Shape[ax:])

	// out tensor
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
	out := Zeros[T](shape...)
	ndim := out.NumDims()

	vndim := v.NumDims()
	if ndim < vndim {
		panic(fmt.Sprintf("shape %v is smaller than tensor shape %v", shape, v.Shape))
	}

	for i := range vndim {
		s0, s1 := v.Shape[vndim-1-i], shape[ndim-1-i]
		if s0 != s1 && s0 != 1 {
			panic(fmt.Sprintf("shape %v is not compatible with tensor shape %v", shape, v.Shape))
		}
	}

	diff := ndim - vndim
	for i := range out.Data {
		k, remain := 0, i
		for a := range ndim {
			coord := remain / out.Stride[a]
			remain %= out.Stride[a]

			j := a - diff
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

// SumTo returns a new tensor with the given shape by summing v to the shape.
func SumTo[N Number](v *Tensor[N], shape ...int) *Tensor[N] {
	axes := func(a, b []int) []int {
		if len(a) < len(b) {
			diff := len(b) - len(a)

			adj := make([]int, len(b))
			for i := range diff {
				adj[i] = 1
			}

			copy(adj[diff:], a)
			a = adj
		}

		var axes []int
		for i := range a {
			if a[i] == 1 && b[i] > 1 {
				axes = append(axes, i)
			}
		}

		return axes
	}

	ax := axes(shape, v.Shape)
	if len(ax) == 0 {
		return Reshape(v, shape...)
	}

	return Reshape(Sum(v, ax...), shape...)
}

// Concat concatenates the tensors along the given axis.
func Concat[T Number](v, w *Tensor[T], axis int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim != w.NumDims() {
		panic("tensors are not the same number of dimensions")
	}

	if ndim == 0 {
		// scalar
		panic("tensor is a scalar")
	}

	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	for i := range ndim {
		if i == ax {
			continue
		}

		if v.Shape[i] != w.Shape[i] {
			panic(fmt.Sprintf("shapes %v and %v are not compatible for concat", v.Shape, w.Shape))
		}
	}

	// out tensor
	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[ax] = v.Shape[ax] + w.Shape[ax]
	out := Zeros[T](shape...)

	// concat
	for i := range v.Data {
		coord := Unravel(v, i)
		out.Data[Ravel(out, coord...)] = v.Data[i]
	}

	for i := range w.Data {
		coord := Unravel(w, i)
		coord[ax] += v.Shape[ax]
		out.Data[Ravel(out, coord...)] = w.Data[i]
	}

	return out
}

// Split returns a new tensors by splitting v into n tensors along the given axis.
func Split[T Number](v *Tensor[T], n, axis int) []*Tensor[T] {
	if n < 1 {
		panic("n is less than 1")
	}

	ndim := v.NumDims()
	if ndim == 0 {
		panic("tensor is a scalar")
	}

	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	if v.Shape[ax]%n != 0 {
		panic(fmt.Sprintf("shape %v is not divisible by n=%d along axis=%d", v.Shape, n, ax))
	}

	// new shape
	shape := make([]int, ndim)
	copy(shape, v.Shape)

	part := v.Shape[ax] / n
	shape[ax] = part

	// out tensors
	out := make([]*Tensor[T], n)
	for i := range n {
		out[i] = Zeros[T](shape...)
	}

	// split
	for i := range v.Data {
		coord := Unravel(v, i)
		idx := coord[ax] / part

		coord[ax] = coord[ax] % part
		j := Ravel(out[idx], coord...)

		// set
		out[idx].Data[j] = v.Data[i]
	}

	return out
}

// Tile returns a new tensor by concatenating v with itself n times along the given axis.
func Tile[T Number](v *Tensor[T], n, axis int) *Tensor[T] {
	if n < 1 {
		panic("n is less than 1")
	}

	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		data := make([]T, n)
		for i := range n {
			data[i] = v.Data[0]
		}

		return New([]int{n}, data)
	}

	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	// out tensor
	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[ax] = v.Shape[ax] * n
	out := Zeros[T](shape...)

	// repeat
	for i := range out.Data {
		coords := Unravel(out, i)
		coords[ax] = coords[ax] % v.Shape[ax]
		out.Data[i] = v.Data[Ravel(v, coords...)]
	}

	return out
}

// Repeat returns a new tensor where each element of v is repeated n times along the given axis.
func Repeat[T Number](v *Tensor[T], n, axis int) *Tensor[T] {
	if n < 1 {
		panic("n is less than 1")
	}

	ndim := v.NumDims()
	if ndim == 0 {
		// scalar
		data := make([]T, n)
		for i := range n {
			data[i] = v.Data[0]
		}

		return New([]int{n}, data)
	}

	axis, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[axis] *= n
	out := Zeros[T](shape...)

	for i := range out.Data {
		coords := Unravel(out, i)

		inCoord := make([]int, ndim)
		copy(inCoord, coords)
		inCoord[axis] = coords[axis] / n

		out.Data[i] = v.Data[Ravel(v, inCoord...)]
	}

	return out
}

// Tril returns a new tensor with the elements below the k-th diagonal zeroed.
func Tril[T Number](v *Tensor[T], k ...int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim < 2 {
		// scalar or 1 dim
		return Clone(v)
	}

	var kk int
	if len(k) != 0 {
		kk = k[0]
	}

	out := ZeroLike(v)
	for i := range v.Data {
		coord := Unravel(v, i)
		if coord[ndim-1] > coord[ndim-2]+kk {
			continue
		}

		out.Data[i] = v.Data[i]
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
	o := Zeros[T](shape...)

	// batch matmul
	workers := runtime.NumCPU()
	chunk := (arows + workers - 1) / workers

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)

		start := w * chunk
		end := min(start+chunk, arows)
		go func(start, end int) {
			defer wg.Done()

			// batch
			for batchIdx := range size(batch) {
				offseta := offset(batchIdx, batch, a.Stride[:ndim-2])
				offsetb := offset(batchIdx, batch, b.Stride[:ndim-2])
				offseto := offset(batchIdx, batch, o.Stride[:ndim-2])

				// matmul
				for i := start; i < end; i++ {
					ai := offseta + i*a.Stride[ndim-2]
					oi := offseto + i*o.Stride[ndim-2]

					for k := range acols {
						aik := a.Data[ai+k*a.Stride[ndim-1]]
						bk := offsetb + k*b.Stride[ndim-2]

						for j := range bcols {
							bkj := b.Data[bk+j*b.Stride[ndim-1]]
							o.Data[oi+j*o.Stride[ndim-1]] += aik * bkj
						}
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
	return o
}

// EqualAll returns true if the two tensors are equal.
func EqualAll(v, w *Tensor[int]) bool {
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

// IsCloseAll returns true if the two tensors are close enough.
func IsCloseAll(v, w *Tensor[float64], atol, rtol float64) bool {
	if !equal(v.Shape, w.Shape) {
		return false
	}

	for i := range v.Data {
		a, b := v.Data[i], w.Data[i]
		if !isClose(a, b, atol, rtol) {
			return false
		}
	}

	return true
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
func F2[T, U Number](v, w *Tensor[T], f func(a, b T) U) *Tensor[U] {
	a, b := Broadcast(v, w)

	out := Zeros[U](a.Shape...)
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

	vndim := v.NumDims()
	_, seen, err := adjAxes(vndim, axes...)
	if err != nil {
		panic(err)
	}

	if len(seen) == vndim {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []T{acc})
	}

	// out tensor
	shape := make([]int, 0, vndim-len(seen))
	ndim := make([]int, vndim)

	var pos int
	for i := range vndim {
		if seen[i] {
			ndim[i] = -1
			continue
		}

		shape = append(shape, v.Shape[i])
		ndim[i] = pos
		pos++
	}

	// output
	out := Full(shape, acc)
	for x := range v.Data {
		i, remain := 0, x
		for j := range vndim {
			coord := remain / v.Stride[j]
			remain = remain % v.Stride[j]

			idx := ndim[j]
			if idx < 0 {
				// reduced axis
				continue
			}

			i += coord * out.Stride[idx]
		}

		// set
		out.Data[i] = f(out.Data[i], v.Data[x])
	}

	return out
}

// Ravel returns the index in the flat data slice for the given multi-dimensional coordinates.
func Ravel[T Number](v *Tensor[T], coord ...int) int {
	if len(coord) == 0 {
		return 0
	}

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

// adjAxes adjusts negative axes, checks the range, and checks for duplicates.
func adjAxes(ndim int, axes ...int) ([]int, map[int]bool, error) {
	adj, seen := make([]int, len(axes)), make(map[int]bool, len(axes))
	for i, a := range axes {
		ax, err := adjAxis(a, ndim)
		if err != nil {
			return nil, nil, err
		}

		if seen[ax] {
			return nil, nil, fmt.Errorf("duplicate axis=%v", ax)
		}

		seen[ax], adj[i] = true, ax
	}

	return adj, seen, nil
}

// isClose returns true if a and b are close enough.
func isClose(a, b float64, atol, rtol float64) bool {
	return math.Abs(a-b) <= atol+rtol*math.Abs(b)
}
