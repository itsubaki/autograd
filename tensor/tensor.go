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

// Tensor represents a multi-dimensional array.
// Row-major order is used for contiguous tensor storage.
// Non-contiguous views (created via Transpose, BroadcastTo, etc.) may share underlying data but have stride patterns that do not follow row-major order.
// Such operations create views without copying data, so the logical element order may differ from the physical memory layout.
type Tensor[T Number] struct {
	Shape    []int
	Stride   []int
	Data     []T
	ReadOnly bool
}

// New returns a new tensor with the given shape and data.
func New[T Number](shape []int, data []T) *Tensor[T] {
	return &Tensor[T]{
		Shape:  append([]int{}, shape...),
		Stride: stride(shape...),
		Data:   data,
	}
}

// Scalar returns a new tensor with a single element.
func Scalar[T Number](v T) *Tensor[T] {
	return New(nil, []T{v})
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

// Like returns a new tensor with the same shape and stride as v and the given data.
func Like[T, U Number](v *Tensor[T], data []U) *Tensor[U] {
	return &Tensor[U]{
		Shape:  append([]int{}, v.Shape...),
		Stride: append([]int{}, v.Stride...),
		Data:   data,
	}
}

// Arange returns a new tensor with evenly spaced values within a given interval.
func Arange[T Number](start, stop T, step ...T) *Tensor[T] {
	var s T = 1
	if len(step) != 0 {
		s = step[0]
	}

	cond := func() func(i T) bool {
		if s > 0 {
			return func(i T) bool { return i < stop }
		}

		return func(i T) bool { return i > stop }
	}()

	var data []T
	for i := start; cond(i); i += s {
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

// FlatIndex returns the index in the flat data slice for the given multi-dimensional indices.
func FlatIndex[T Number](v *Tensor[T], indices ...int) int {
	if len(indices) == 0 {
		return 0
	}

	if len(indices) != len(v.Shape) {
		panic(fmt.Sprintf("indices length=%v are not equal to ndim=%v", len(indices), len(v.Shape)))
	}

	var idx int
	for i, c := range indices {
		if c < 0 || c >= v.Shape[i] {
			panic(fmt.Sprintf("indices=%v out of range for axis=%v (shape=%v)", c, i, v.Shape))
		}

		idx += c * v.Stride[i]
	}

	return idx
}

// NumDims returns the number of dimensions of the tensor.
func (v *Tensor[T]) NumDims() int {
	return len(v.Shape)
}

// Size returns the number of elements in the tensor.
func (v *Tensor[T]) Size() int {
	return size(v.Shape)
}

// At returns the element at the given index.
// If no indices are given, it returns the first element.
func (v *Tensor[T]) At(indices ...int) T {
	return v.Data[FlatIndex(v, indices...)]
}

// Set sets the element at the given index to the given value.
func (v *Tensor[T]) Set(indices []int, value T) {
	if v.ReadOnly {
		panic("cannot modify read-only tensor")
	}

	v.Data[FlatIndex(v, indices...)] = value
}

// AddAt adds the given value to the element at the given index.
func (v *Tensor[T]) AddAt(indices []int, value T) {
	if v.ReadOnly {
		panic("cannot modify read-only tensor")
	}

	v.Data[FlatIndex(v, indices...)] += value
}

// Seq2 returns a sequence of rows.
func (v *Tensor[T]) Seq2() iter.Seq2[int, []T] {
	// scalar
	ndim := v.NumDims()
	if ndim == 0 {
		return func(yield func(int, []T) bool) {
			yield(0, v.Data)
		}
	}

	seq2 := func(v *Tensor[T]) func(yield func(int, []T) bool) {
		size := v.Shape[ndim-1]
		total := v.Size() / size
		return func(yield func(int, []T) bool) {
			for i := range total {
				start := i * size
				if !yield(i, v.Data[start:start+size]) {
					return
				}
			}
		}
	}

	return seq2(Contiguous(v))
}

// Clone returns a contiguous clone of the tensor.
func Clone[T Number](v *Tensor[T]) *Tensor[T] {
	out := Zeros[T](v.Shape...)
	for i := range out.Size() {
		out.Data[i] = v.At(UnravelIndex(out, i)...)
	}

	return out
}

// Contiguous returns a contiguous tensor.
// If the tensor is already contiguous, it returns the original tensor.
// Otherwise, it returns a clone of the tensor.
func Contiguous[T Number](v *Tensor[T]) *Tensor[T] {
	if IsContiguous(v) {
		return v
	}

	return Clone(v)
}

// Int returns a new tensor with elements casted to int.
func Int[T Number](v *Tensor[T]) *Tensor[int] {
	return F(v, func(a T) int { return int(a) })
}

// Float64 returns a new tensor with elements casted to float64.
func Float64[T Number](v *Tensor[T]) *Tensor[float64] {
	return F(v, func(a T) float64 { return float64(a) })
}

// AddC returns a new tensor with each element in v added to c.
func AddC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c + a })
}

// SubC returns a new tensor with each element in v subtracted from c.
func SubC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c - a })
}

// MulC returns a new tensor with each element in v multiplied by c.
func MulC[T Number](c T, v *Tensor[T]) *Tensor[T] {
	return F(v, func(a T) T { return c * a })
}

// Pow returns a new tensor with each element in v raised to the power of p.
func Pow(p float64, v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Pow(a, p) })
}

// Sqrt returns a new tensor with the square root of each element in v.
func Sqrt(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Sqrt(a) })
}

// Exp returns a new tensor with the exponential of each element in v.
func Exp(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

// Log returns a new tensor with the natural logarithm of each element in v.
func Log(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Log(a) })
}

// Sin returns a new tensor with the sine of each element in v.
func Sin(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

// Cos returns a new tensor with the cosine of each element in v.
func Cos(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

// Tanh returns a new tensor with the hyperbolic tangent of each element in v.
func Tanh(v *Tensor[float64]) *Tensor[float64] {
	return F(v, func(a float64) float64 { return math.Tanh(a) })
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

// Add returns a new tensor with elements v + w for each element in v.
func Add[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a + b })
}

// Sub returns a new tensor with elements v - w for each element in v.
func Sub[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a - b })
}

// Mul returns a new tensor with elements v * w for each element in v.
func Mul[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a * b })
}

// Div returns a new tensor with elements v / w for each element in v.
func Div[T Number](v, w *Tensor[T]) *Tensor[T] {
	return F2(v, w, func(a, b T) T { return a / b })
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
func IsClose(v, w *Tensor[float64], tol ...float64) *Tensor[int] {
	return F2(v, w, func(a, b float64) int {
		if isClose(a, b, tol...) {
			return 1
		}

		return 0
	})
}

// Ravel returns a new tensor containing the same elements as v.
// Ravel returns a view of v when possible; otherwise, it returns a clone.
func Ravel[T Number](v *Tensor[T]) *Tensor[T] {
	return Reshape(v, -1)
}

// Flatten returns a new tensor containing the same elements as v.
// Flatten returns a clone of v.
func Flatten[T Number](v *Tensor[T]) *Tensor[T] {
	return Clone(Ravel(v))
}

// Reshape returns a new tensor with the same data as v with the given shape.
// Reshape returns a view of v when possible; otherwise, it returns a clone.
func Reshape[T Number](v *Tensor[T], shape ...int) *Tensor[T] {
	idx, prod := -1, 1
	for i, s := range shape {
		if s == -1 && idx != -1 {
			panic("duplicate -1 in shape")
		}

		if s == -1 {
			idx = i
			continue
		}

		prod *= s
	}

	s := v.Size()
	if idx != -1 {
		if s%prod != 0 {
			panic("shape with -1 is not divisible")
		}

		shape[idx] = s / prod
	}

	if size(shape) != s {
		panic("invalid shape")
	}

	return New(shape, Contiguous(v).Data)
}

// Transpose returns a new tensor with the axes transposed.
// Transpose returns a view of v.
func Transpose[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim == 0 {
		return Like(v, v.Data)
	}

	if len(axes) == 0 {
		// reverse
		for i := range ndim {
			axes = append(axes, ndim-1-i)
		}
	}

	perm, _, err := adjAxes(ndim, axes...)
	if err != nil {
		panic(err)
	}

	// permute shape and stride
	shape := make([]int, ndim)
	stride := make([]int, ndim)
	for i := range ndim {
		shape[i] = v.Shape[perm[i]]
		stride[i] = v.Stride[perm[i]]
	}

	return &Tensor[T]{
		Shape:  shape,
		Stride: stride,
		Data:   v.Data,
	}
}

// Broadcast returns new tensors by broadcasting v and w to a common shape.
// Broadcast returns views of v and w.
func Broadcast[T Number](v, w *Tensor[T], keepLast ...int) (*Tensor[T], *Tensor[T]) {
	s0, s1, err := broadcast(v.Shape, w.Shape, keepLast...)
	if err != nil {
		panic(err)
	}

	return BroadcastTo(v, s0...), BroadcastTo(w, s1...)
}

// BroadcastTo returns a new tensor with the given shape by broadcasting v to the shape.
func BroadcastTo[T Number](v *Tensor[T], shape ...int) *Tensor[T] {
	ndim, vndim := len(shape), v.NumDims()
	if ndim < vndim {
		panic(fmt.Sprintf("shape %v is smaller than tensor shape %v", shape, v.Shape))
	}

	diff := ndim - vndim
	stride := make([]int, ndim)
	for i := range ndim {
		x := i - diff
		if x < 0 {
			stride[i] = 0
			continue
		}

		s0, s1 := v.Shape[x], shape[i]
		if s0 != s1 && s0 != 1 {
			panic(fmt.Sprintf("shape %v is not compatible with tensor shape %v", shape, v.Shape))
		}

		if s0 == 1 {
			stride[i] = 0
			continue
		}

		stride[i] = v.Stride[x]
	}

	return &Tensor[T]{
		Shape:    append([]int{}, shape...),
		Stride:   stride,
		Data:     v.Data,
		ReadOnly: true,
	}
}

// SumTo returns a new tensor with the given shape by summing v to the shape.
// SumTo returns a view of v when possible; otherwise, it returns a clone.
func SumTo[N Number](v *Tensor[N], shape ...int) *Tensor[N] {
	a, b := shape, v.Shape
	if len(a) < len(b) {
		diff := len(b) - len(a)

		// prepend 1s to a
		adj := make([]int, len(b))
		for i := range diff {
			adj[i] = 1
		}

		// copy a to adj
		copy(adj[diff:], a)
		a = adj
	}

	var axes []int
	for i := range a {
		if a[i] == 1 && b[i] > 1 {
			axes = append(axes, i)
		}
	}

	if len(axes) == 0 {
		return Reshape(v, shape...)
	}

	return Reshape(Sum(v, axes...), shape...)
}

// Squeeze returns a new tensor with the given axes removed.
// If axes is empty, all axes with size 1 are removed.
// Squeeze returns a view of v.
func Squeeze[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	seen, ndim := make(map[int]bool), v.NumDims()
	if len(axes) > 0 {
		for _, a := range axes {
			ax, err := adjAxis(a, ndim)
			if err != nil {
				panic(err)
			}

			if v.Shape[ax] != 1 {
				panic(fmt.Sprintf("axis=%v is not 1 (shape %v)", ax, v.Shape))
			}

			seen[ax] = true
		}
	}

	var shape, stride []int
	for i, s := range v.Shape {
		if len(axes) == 0 && s == 1 {
			continue
		}

		if seen[i] {
			continue
		}

		shape = append(shape, s)
		stride = append(stride, v.Stride[i])
	}

	return &Tensor[T]{
		Shape:  shape,
		Stride: stride,
		Data:   v.Data,
	}
}

// Expand returns a new tensor with a new axis inserted at the given position.
// Expand returns a view of v.
func Expand[T Number](v *Tensor[T], axis int) *Tensor[T] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim+1)
	if err != nil {
		panic(err)
	}

	shape := make([]int, 0, ndim+1)
	stride := make([]int, 0, ndim+1)

	// head
	for i := range ax {
		shape = append(shape, v.Shape[i])
		stride = append(stride, v.Stride[i])
	}

	// insert 1 at axis
	shape = append(shape, 1)
	stride = append(stride, 0)

	// tail
	for i := ax; i < ndim; i++ {
		shape = append(shape, v.Shape[i])
		stride = append(stride, v.Stride[i])
	}

	return &Tensor[T]{
		Shape:  shape,
		Stride: stride,
		Data:   v.Data,
	}
}

// ScatterAdd returns a new tensor with elements added from w at the given indices along the specified axis.
func ScatterAdd[T Number](v, w *Tensor[T], axis int, indices []int) *Tensor[T] {
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
	for i := range w.Size() {
		widx := UnravelIndex(w, i)

		oidx := make([]int, ndim)
		copy(oidx, widx)
		oidx[ax] = idx[widx[ax]]

		out.AddAt(oidx, w.At(widx...))
	}

	return out
}

// Take returns a new tensor with elements selected from the given indices along the specified axis.
func Take[T Number](v *Tensor[T], axis int, indices []int) *Tensor[T] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	idx, err := adjIndices(indices, v.Shape, ax)
	if err != nil {
		panic(err)
	}

	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[ax] = len(indices)

	// take
	out := Zeros[T](shape...)
	for i := range out.Size() {
		oidx := UnravelIndex(out, i)

		vidx := make([]int, ndim)
		copy(vidx, oidx)
		vidx[ax] = idx[oidx[ax]]

		out.Set(oidx, v.At(vidx...))
	}

	return out
}

// Concat returns a new tensor by concatenating the tensors along the given axis.
func Concat[T Number](v []*Tensor[T], axis int) *Tensor[T] {
	ndim := v[0].NumDims()
	for i := range v {
		if v[i].NumDims() != ndim {
			panic("tensors have different number of dimensions")
		}
	}

	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	shape := make([]int, ndim)
	copy(shape, v[0].Shape)
	shape[ax] = 0
	for i := range v {
		shape[ax] += v[i].Shape[ax]
	}

	// concat
	var offset int
	out := Zeros[T](shape...)
	for _, w := range v {
		for j := range w.Size() {
			widx := UnravelIndex(w, j)

			oidx := make([]int, ndim)
			copy(oidx, widx)
			oidx[ax] += offset

			out.Set(oidx, w.At(widx...))
		}

		offset += w.Shape[ax]
	}

	return out
}

// Stack returns a new tensor by stacking the tensors along the given axis.
func Stack[T Number](v []*Tensor[T], axis int) *Tensor[T] {
	ndim := v[0].NumDims()
	ax, err := adjAxis(axis, ndim+1)
	if err != nil {
		panic(err)
	}

	shape := make([]int, ndim+1)
	copy(shape[:ax], v[0].Shape[:ax])
	shape[ax] = len(v)
	copy(shape[ax+1:], v[0].Shape[ax:])

	// stack
	out := Zeros[T](shape...)
	for i, w := range v {
		for j := range w.Size() {
			widx := UnravelIndex(w, j)

			// insert i at axis
			oidx := make([]int, ndim+1)
			copy(oidx[:ax], widx[:ax])
			oidx[ax] = i
			copy(oidx[ax+1:], widx[ax:])

			out.Set(oidx, w.At(widx...))
		}
	}

	return out
}

// Split returns a list of tensors by splitting v into parts with the given size along the given axis.
func Split[T Number](v *Tensor[T], size []int, axis int) []*Tensor[T] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	var sum int
	for _, s := range size {
		sum += s
	}

	if sum != v.Shape[ax] {
		panic("sum of size is not equal to shape at axis")
	}

	var start int
	out := make([]*Tensor[T], len(size))
	for i, s := range size {
		shape := make([]int, ndim)
		copy(shape, v.Shape)
		shape[ax] = s

		// copy
		out[i] = Zeros[T](shape...)
		for j := range out[i].Size() {
			oidx := UnravelIndex(out[i], j)

			vidx := make([]int, ndim)
			copy(vidx, oidx)
			vidx[ax] += start

			out[i].Set(oidx, v.At(vidx...))
		}

		start += s
	}

	return out
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
	for i := range v.Size() {
		vidx := UnravelIndex(v, i)

		// Flip the indices along axis 'a'.
		// This operation reverses the order of elements along that axis.
		//
		// For a given dimension of size `n`, the original index `i` is mapped to:
		//     new_index = n - 1 - i
		//
		// Example:
		//   If the axis length is 5, the indices are [0, 1, 2, 3, 4].
		//   After flipping:
		//       0 → 4
		//       1 → 3
		//       2 → 2
		//       3 → 1
		//       4 → 0
		//
		// So when size = 5 and index = 1, the new index becomes 5 - 1 - 1 = 3.
		// This achieves a mirror-like reflection along the selected axis.
		oidx := make([]int, ndim)
		copy(oidx, vidx)
		for _, a := range ax {
			oidx[a] = v.Shape[a] - 1 - oidx[a]
		}

		out.Set(vidx, v.At(oidx...))
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

	shape := make([]int, ndim)
	copy(shape, v.Shape)
	shape[ax] = v.Shape[ax] * n

	// repeat
	out := Zeros[T](shape...)
	for i := range out.Size() {
		oidx := UnravelIndex(out, i)

		vidx := make([]int, ndim)
		copy(vidx, oidx)
		vidx[ax] = vidx[ax] % v.Shape[ax]

		out.Set(oidx, v.At(vidx...))
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
	for i := range out.Size() {
		oidx := UnravelIndex(out, i)

		vidx := make([]int, ndim)
		copy(vidx, oidx)
		vidx[axis] = oidx[axis] / n

		out.Set(oidx, v.At(vidx...))
	}

	return out
}

// Tril returns a new tensor with the elements below the k-th diagonal zeroed.
func Tril[T Number](v *Tensor[T], k ...int) *Tensor[T] {
	ndim := v.NumDims()
	if ndim < 2 {
		return Clone(v)
	}

	var kk int
	if len(k) != 0 {
		kk = k[0]
	}

	out := ZeroLike(v)
	for i := range v.Size() {
		vidx := UnravelIndex(v, i)
		if vidx[ndim-1] > vidx[ndim-2]+kk {
			continue
		}

		out.Set(vidx, v.At(vidx...))
	}

	return out
}

// Argmax returns the indices of the maximum values along the given axis.
func Argmax[T Number](v *Tensor[T], axis int) *Tensor[int] {
	ndim := v.NumDims()
	ax, err := adjAxis(axis, ndim)
	if err != nil {
		panic(err)
	}

	// axis=1, (2, 3, 4) -> (2, 4, 3)
	perm := make([]int, ndim)
	for i := range ndim {
		perm[i] = i
	}
	perm[ax], perm[ndim-1] = perm[ndim-1], perm[ax]
	vt := Contiguous(Transpose(v, perm...))
	axSize := vt.Shape[ndim-1]

	out := Zeros[int](vt.Shape[:ndim-1]...)
	for i := range out.Size() {
		maxIdx, maxVal := 0, vt.Data[i*axSize]
		for j := 1; j < axSize; j++ {
			val := vt.Data[i*axSize+j]
			if val > maxVal {
				maxIdx, maxVal = j, val
			}
		}

		out.Data[i] = maxIdx
	}

	return out
}

// ReduceAll reduces the tensor v to a scalar by applying the function f to all elements.
func ReduceAll[T Number](v *Tensor[T], acc T, f func(a, b T) T) *Tensor[T] {
	for i := range v.Size() {
		acc = f(acc, v.At(UnravelIndex(v, i)...))
	}

	return Scalar(acc)
}

// Reduce reduces the tensor v to a tensor with fewer dimensions by applying the function f along the given axes.
func Reduce[T Number](v *Tensor[T], acc T, f func(a, b T) T, axes ...int) *Tensor[T] {
	if len(axes) == 0 {
		return ReduceAll(v, acc, f)
	}

	vndim := v.NumDims()
	_, seen, err := adjAxes(vndim, axes...)
	if err != nil {
		panic(err)
	}

	if len(seen) == vndim {
		return ReduceAll(v, acc, f)
	}

	shape := make([]int, 0, vndim-len(seen))
	ndim := make([]int, vndim)

	// Example: v.Shape = [2, 3, 4], axes = [1]
	// seen = [false, true, false]
	// Loop changes:
	//   i=0 → seen[0]=false → ndim[0]=0, pos=1
	//   i=1 → seen[1]=true  → ndim[1]=-1, pos=1
	//   i=2 → seen[2]=false → ndim[2]=1, pos=2
	// Result:
	//   shape = [2, 4]
	//   ndim  = [0, -1, 1]
	// -1 indicates a reduced axis, which will be skipped in the later loop
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

	out := Full(shape, acc)
	for i := range v.Size() {
		vidx := UnravelIndex(v, i)

		var k int
		for j := range vndim {
			idx := ndim[j]
			if idx < 0 {
				// reduced axis
				continue
			}

			k += vidx[j] * out.Stride[idx]
		}

		// set
		out.Data[k] = f(out.Data[k], v.At(vidx...))
	}

	return out
}

// Sum returns a new tensor with the sum of all elements in v.
// If axes is specified, it reduces along the given axes.
func Sum[T Number](v *Tensor[T], axes ...int) *Tensor[T] {
	return Reduce(v, 0, func(acc, x T) T {
		return acc + x
	}, axes...)
}

// Max returns a new tensor with the maximum value among all elements in v.
// If axes is specified, it reduces along the given axes.
func Max(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Reduce(v, -math.MaxFloat64, func(acc, x float64) float64 {
		if x > acc {
			return x
		}

		return acc
	}, axes...)
}

// Min returns a new tensor with the minimum value among all elements in v.
// If axes is specified, it reduces along the given axes.
func Min(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Reduce(v, math.MaxFloat64, func(acc, x float64) float64 {
		if x < acc {
			return x
		}

		return acc
	}, axes...)
}

// Mean returns a new tensor with the mean of elements in v.
// If axes is specified, it reduces along the given axes.
func Mean[T Number](v *Tensor[T], axes ...int) *Tensor[float64] {
	ndim := v.NumDims()
	if ndim == 0 {
		return Float64(Clone(v))
	}

	if len(axes) == 0 {
		// mean all
		return MulC(1/float64(v.Size()), Float64(Sum(v)))
	}

	ax, _, err := adjAxes(ndim, axes...)
	if err != nil {
		panic(err)
	}

	// size
	size := 1
	for _, a := range ax {
		size = size * v.Shape[a]
	}

	// mean
	return MulC(1/float64(size), Float64(Sum(v, ax...)))
}

// Variance returns a new tensor with the variance of elements in v.
func Variance(v *Tensor[float64], axes ...int) *Tensor[float64] {
	ndim := v.NumDims()
	if ndim == 0 {
		return Scalar(0.0)
	}

	if len(axes) == 0 {
		mu := Mean(v)      // mean
		xc := Sub(v, mu)   // x - mean
		xc2 := Mul(xc, xc) // (x - mean)**2
		return Mean(xc2)   // mean((x - mean)**2)
	}

	shape := KeepDims(v.Shape, axes)
	mu := Reshape(Mean(v, axes...), shape...) // mean
	xc := Sub(v, mu)                          // x - mean
	xc2 := Mul(xc, xc)                        // (x - mean)**2
	return Mean(xc2, axes...)                 // mean((x - mean)**2)
}

// StdDev returns a new tensor with the standard deviation of elements in v.
func StdDev(v *Tensor[float64], axes ...int) *Tensor[float64] {
	return Sqrt(Variance(v, axes...))
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
	offset := func(k int, shape, stride []int) int {
		var v int
		for i := len(shape) - 1; i >= 0; i-- {
			idx := k % shape[i]
			k /= shape[i]

			v += idx * stride[i]
		}

		return v
	}

	batch := a.Shape[:ndim-2]
	shape := append(batch, []int{arows, bcols}...)
	o := Zeros[T](shape...)

	// Determine the number of rows each goroutine will handle.
	// We use "ceiling division" to make sure all rows are covered,
	// even if rows is not divisible by workers.
	//
	// Example:
	//   rows = 10, workers = 3
	//   chunk = (10 + 3 - 1) / 3 = 12 / 3 = 4
	//   Goroutine row ranges:
	//     Worker 0: rows 0, 1, 2, 3
	//     Worker 1: rows 4, 5, 6, 7
	//     Worker 2: rows 8, 9
	//   Notice that the last worker handles the remaining 2 rows.
	//
	// If we simply divided by workers using integer division (rows / workers),
	// the rows might not be distributed evenly.
	//
	// Example:
	//   rows = 10, workers = 3
	//   chunk: 10 / 3 = 3
	//     Worker 0: rows 0, 1, 2
	//     Worker 1: rows 3, 4, 5
	//     Worker 2: rows 6, 7, 8
	//     Row 9 would be left unassigned.
	//
	// Note:
	//   If there is a remainder, the workload balance among workers is uneven.
	//   Some workers may finish earlier and stay idle while others process the extra rows.
	//   Ceiling division helps distribute the workload more evenly.
	//
	workers := runtime.NumCPU()
	chunk := (arows + workers - 1) / workers

	// batch matmul
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
							oij := oi + j*o.Stride[ndim-1]

							o.Data[oij] += aik * bkj
						}
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
	return o
}

// SliceEqual returns true if the two slices are equal.
func SliceEqual(a, b []int) bool {
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

// EqualAll returns true if the two tensors are equal.
func EqualAll(v, w *Tensor[int]) bool {
	if !SliceEqual(v.Shape, w.Shape) {
		return false
	}

	for i := range v.Size() {
		if v.At(UnravelIndex(v, i)...) != w.At(UnravelIndex(w, i)...) {
			return false
		}
	}

	return true
}

// IsCloseAll returns true if the two tensors are close enough.
func IsCloseAll(v, w *Tensor[float64], tol ...float64) bool {
	if !SliceEqual(v.Shape, w.Shape) {
		return false
	}

	for i := range v.Size() {
		a, b := v.At(UnravelIndex(v, i)...), w.At(UnravelIndex(w, i)...)
		if !isClose(a, b, tol...) {
			return false
		}
	}

	return true
}

// IsContiguous returns true if the tensor is stored in contiguous memory.
func IsContiguous[T Number](v *Tensor[T]) bool {
	size := 1
	for i := v.NumDims() - 1; i >= 0; i-- {
		if v.Shape[i] == 1 {
			continue
		}

		if v.Stride[i] != size {
			return false
		}

		size *= v.Shape[i]
	}

	return true
}

// F applies the function f to each element of the tensor v and returns a new tensor.
// The returned tensor has the same shape and layout as v.
// In particular, if v is non-contiguous, the result is also non-contiguous.
func F[T, U Number](v *Tensor[T], f func(a T) U) *Tensor[U] {
	data := make([]U, len(v.Data))
	for i := range data {
		data[i] = f(v.Data[i])
	}

	return Like(v, data)
}

// F2 applies the function f to each element of the tensors v and w and returns a new tensor.
// v and w are broadcasted to a common shape.
func F2[T, U Number](v, w *Tensor[T], f func(a, b T) U) *Tensor[U] {
	a, b := Broadcast(v, w)

	out := Zeros[U](a.Shape...)
	for i := range out.Size() {
		oidx := UnravelIndex(out, i)
		out.Set(oidx, f(a.At(oidx...), b.At(oidx...)))
	}

	return out
}

// UnravelIndex returns the multi-dimensional indices for the given logical index (0 to v.Size()-1) in the tensor.
func UnravelIndex[T Number](v *Tensor[T], index int) []int {
	ndim := v.NumDims()

	out := make([]int, ndim)
	for i := ndim - 1; i >= 0; i-- {
		out[i] = index % v.Shape[i]
		index /= v.Shape[i]
	}

	return out
}

// KeepDims returns a new shape with 1 inserted at the given axes.
func KeepDims(shape []int, axes []int) []int {
	ndim := len(shape)
	for i := range axes {
		if axes[i] < 0 {
			axes[i] += ndim
		}
	}

	ax := make(map[int]struct{}, len(axes))
	for _, a := range axes {
		ax[a] = struct{}{}
	}

	out := make([]int, len(shape))
	for i, s := range shape {
		if _, ok := ax[i]; ok {
			out[i] = 1
			continue
		}

		out[i] = s
	}

	return out
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
func isClose(a, b float64, tol ...float64) bool {
	atol, rtol := func() (float64, float64) {
		if len(tol) == 0 {
			return 1e-8, 1e-5
		}

		if len(tol) == 1 {
			return tol[0], tol[0]
		}

		return tol[0], tol[1]
	}()

	return math.Abs(a-b) <= atol+rtol*math.Max(math.Abs(a), math.Abs(b))
}
