package tensor

import (
	"fmt"
	"math"
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/rand"
)

type Tensor struct {
	Shape  []int
	Stride []int
	Data   []float64
}

// New returns a new tensor with the given shape and data.
func New(shape []int, data []float64) *Tensor {
	return &Tensor{
		Shape:  shape,
		Stride: stride(shape...),
		Data:   data,
	}
}

// Full returns a tensor with elements that are all the given value.
func Full(shape []int, value float64) *Tensor {
	return F(Zero(shape...), func(_ float64) float64 { return value })
}

// Rand returns a tensor with elements that pseudo-random number in the half-open interval [0.0,1.0).
func Rand(shape []int, s ...randv2.Source) *Tensor {
	return F(Zero(shape...), func(_ float64) float64 { return rnd(s...).Float64() })
}

// Randn returns a tensor with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
func Randn(shape []int, s ...randv2.Source) *Tensor {
	return F(Zero(shape...), func(_ float64) float64 { return rnd(s...).NormFloat64() })
}

// rnd returns a pseudo-random number generator.
func rnd(s ...randv2.Source) *randv2.Rand {
	if len(s) == 0 || s[0] == nil {
		return randv2.New(rand.NewSource(rand.MustRead()))
	}

	return randv2.New(s[0])
}

// Zero returns a tensor with elements that are all zero.
func Zero(shape ...int) *Tensor {
	return New(shape, make([]float64, size(shape)))
}

// ZeroLike returns a tensor with the same shape as v and elements that are all zero.
func ZeroLike(v *Tensor) *Tensor {
	return Zero(v.Shape...)
}

// OneLike returns a tensor with the same shape as v and elements that are all one.
func OneLike(v *Tensor) *Tensor {
	return F(ZeroLike(v), func(_ float64) float64 { return 1.0 })
}

// Reshape returns a new tensor with the same data as v with the given shape.
func (v *Tensor) Reshape(shape ...int) *Tensor {
	if size(shape) != v.Size() {
		panic("invalid shape")
	}

	return New(shape, v.Data)
}

// Clone returns a copy of the tensor.
func (v *Tensor) Clone() *Tensor {
	data := make([]float64, len(v.Data))
	copy(data, v.Data)
	return New(v.Shape, data)
}

// NumDims returns the number of dimensions of the tensor.
func (v *Tensor) NumDims() int {
	return len(v.Shape)
}

// Size returns the number of elements in the tensor.
func (v *Tensor) Size() int {
	return size(v.Shape)
}

// At returns the element at the given index.
func (v *Tensor) At(index ...int) float64 {
	return v.Data[Index(v, index...)]
}

// Set sets the element at the given index to the given value.
func (v *Tensor) Set(index []int, value float64) {
	v.Data[Index(v, index...)] = value
}

// AddAt adds the given value to the element at the given index.
func (v *Tensor) AddAt(index []int, value float64) {
	v.Data[Index(v, index...)] += value
}

// EqualShape returns true if the shapes of the two tensors are equal.
func (v *Tensor) EqualShape(w *Tensor) bool {
	if len(v.Shape) != len(w.Shape) {
		return false
	}

	for i := range v.Shape {
		if v.Shape[i] != w.Shape[i] {
			return false
		}
	}

	return true
}

// AddC returns c + v for each element in v.
func AddC(c float64, v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return c + a })
}

// SubC returns c - v for each element in v.
func SubC(c float64, v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return c - a })
}

// MulC returns c * v for each element in v.
func MulC(c float64, v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return c * a })
}

// Pow returns v**p for each element in v.
func Pow(v *Tensor, p float64) *Tensor {
	return F(v, func(a float64) float64 { return math.Pow(a, p) })
}

// Exp returns exp(v) for each element in v.
func Exp(v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return math.Exp(a) })
}

// Log returns log(v) for each element in v.
func Log(v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return math.Log(a) })
}

// Sin returns sin(v) for each element in v.
func Sin(v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return math.Sin(a) })
}

// Cos returns cos(v) for each element in v.
func Cos(v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return math.Cos(a) })
}

// Tanh returns tanh(v) for each element in v.
func Tanh(v *Tensor) *Tensor {
	return F(v, func(a float64) float64 { return math.Tanh(a) })
}

// Add returns v + w for each element in v.
func Add(v, w *Tensor) *Tensor {
	return F2(v, w, func(a, b float64) float64 { return a + b })
}

// Sub returns v - w for each element in v.
func Sub(v, w *Tensor) *Tensor {
	return F2(v, w, func(a, b float64) float64 { return a - b })
}

// Mul returns v * w for each element in v.
func Mul(v, w *Tensor) *Tensor {
	return F2(v, w, func(a, b float64) float64 { return a * b })
}

// Div returns v / w for each element in v.
func Div(v, w *Tensor) *Tensor {
	return F2(v, w, func(a, b float64) float64 { return a / b })
}

// Sum returns the sum of all elements in v. If axis is specified, it reduces along the given axes.
func Sum(v *Tensor, axis ...int) *Tensor {
	return Reduce(v, 0.0, func(acc, x float64) float64 { return acc + x }, axis...)
}

// Max returns the maximum value among all elements in v. If axis is specified, it reduces along the given axes.
func Max(v *Tensor, axis ...int) *Tensor {
	return Reduce(v, -math.MaxFloat64, func(acc, x float64) float64 {
		if x > acc {
			return x
		}

		return acc
	}, axis...)
}

// Min returns the minimum value among all elements in v. If axis is specified, it reduces along the given axes.
func Min(v *Tensor, axis ...int) *Tensor {
	return Reduce(v, math.MaxFloat64, func(acc, x float64) float64 {
		if x < acc {
			return x
		}

		return acc
	}, axis...)
}

// Mean returns the mean of all elements in v. If axis is specified, it reduces along the given axes.
func Mean(v *Tensor, axis ...int) *Tensor {
	if len(axis) == 0 {
		return MulC(1/float64(v.Size()), Sum(v, axis...))
	}

	reduced, _, err := reduce(v, axis...)
	if err != nil {
		panic(err)
	}

	count := 1
	for a := range reduced {
		count = count * v.Shape[a]
	}

	return MulC(1/float64(count), Sum(v, axis...))
}

// Mask returns a tensor with elements that 1 if f() is true and 0 otherwise.
func Mask(v *Tensor, f func(x float64) bool) *Tensor {
	return F(v, func(x float64) float64 {
		if f(x) {
			return 1
		}

		return 0
	})
}

// Clip returns a tensor with elements that are clipped to the interval [min, max].
func Clip(v *Tensor, min, max float64) *Tensor {
	return F(v, func(x float64) float64 {
		if x < min {
			return min
		}

		if x > max {
			return max
		}

		return x
	})
}

// F applies the function f to each element of the tensor v and returns a new tensor.
func F(v *Tensor, f func(a float64) float64) *Tensor {
	out := ZeroLike(v)
	for i := range v.Data {
		out.Data[i] = f(v.Data[i])
	}

	return out
}

// F2 applies the function f to each element of the tensors v and w and returns a new tensor.
func F2(v, w *Tensor, f func(a, b float64) float64) *Tensor {
	if !v.EqualShape(w) {
		panic("shapes are not equal")
	}

	out := ZeroLike(v)
	for i := range v.Data {
		out.Data[i] = f(v.Data[i], w.Data[i])
	}

	return out
}

// Reduce reduces the tensor v to a tensor with fewer dimensions by applying the function f along the given axes.
func Reduce(v *Tensor, acc float64, f func(a, b float64) float64, axis ...int) *Tensor {
	if len(axis) == 0 {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []float64{acc})
	}

	// validate
	reduced, ndim, err := reduce(v, axis...)
	if err != nil {
		panic(err)
	}

	if len(reduced) == ndim {
		// reduce all
		for _, x := range v.Data {
			acc = f(acc, x)
		}

		return New(nil, []float64{acc})
	}

	// reduced layout
	rshape := make([]int, 0, ndim-len(reduced))
	rndim := make([]int, ndim)

	var pos int
	for i := range ndim {
		if _, ok := reduced[i]; ok {
			rndim[i] = -1
			continue
		}

		rshape = append(rshape, v.Shape[i])
		rndim[i] = pos
		pos++
	}

	// output
	out := Full(rshape, acc)
	stride := stride(rshape...)
	for x := range len(v.Data) {
		i, remain := 0, x
		for j := range ndim {
			coord := remain / v.Stride[j]
			remain = remain % v.Stride[j]

			idx := rndim[j]
			if idx < 0 {
				// reduced axis
				continue
			}

			i += coord * stride[idx]
		}

		// set
		out.Data[i] = f(out.Data[i], v.Data[x])
	}

	return out
}

// Index returns the index in the flat data slice for the given multi-dimensional index.
func Index(v *Tensor, index ...int) int {
	if len(index) != len(v.Shape) {
		panic("invalid number of index")
	}

	var idx int
	for i, index := range index {
		if index < 0 || index >= v.Shape[i] {
			panic(fmt.Sprintf("index %q out of bounds for axis %q (shape=%q)", index, i, v.Shape))
		}

		idx += index * v.Stride[i]
	}

	return idx
}

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

func reduce(v *Tensor, axis ...int) (map[int]struct{}, int, error) {
	ndim := v.NumDims()
	if ndim == 0 {
		return nil, 0, fmt.Errorf("axis out of range for scalar tensor")
	}

	out := make(map[int]struct{}, len(axis))
	for _, a := range axis {
		if a < 0 || a >= ndim {
			return nil, ndim, fmt.Errorf("axis %q out of range (ndim=%q)", a, ndim)
		}

		if _, ok := out[a]; ok {
			return nil, ndim, fmt.Errorf("duplicate axis %q", a)
		}

		out[a] = struct{}{}
	}

	return out, ndim, nil
}

func size(shape []int) int {
	size := 1
	for _, s := range shape {
		size *= s
	}

	return size
}
