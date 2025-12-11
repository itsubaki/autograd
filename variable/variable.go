package variable

import (
	"fmt"
	randv2 "math/rand/v2"
	"sort"

	"github.com/itsubaki/autograd/tensor"
)

type Variable struct {
	Name       string
	Data       *tensor.Tensor[float64]
	Grad       *Variable
	Creator    *Function
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: tensor.New([]int{len(v)}, v)}
}

func From(v *tensor.Tensor[float64]) *Variable {
	return &Variable{Data: v}
}

func ZeroLike(v *Variable) *Variable {
	return &Variable{Data: tensor.ZeroLike(v.Data)}
}

func OneLike(v *Variable) *Variable {
	return &Variable{Data: tensor.OneLike(v.Data)}
}

func Zeros(shape ...int) *Variable {
	return &Variable{Data: tensor.Zeros[float64](shape...)}
}

func Rand(shape []int, s ...randv2.Source) *Variable {
	return &Variable{Data: tensor.Rand(shape, s...)}
}

func Randn(shape []int, s ...randv2.Source) *Variable {
	return &Variable{Data: tensor.Randn(shape, s...)}
}

// At returns the value at the given coordinates.
// If no coordinates are given, it returns the first element.
func (v *Variable) At(coord ...int) float64 {
	return v.Data.At(coord...)
}

// NumDims returns the number of dimensions of the variable.
func (v *Variable) NumDims() int {
	return v.Data.NumDims()
}

// Size returns the number of elements in the variable.
func (v *Variable) Size() int {
	return v.Data.Size()
}

// Shape returns the shape of the variable.
func (v *Variable) Shape() []int {
	shape := make([]int, len(v.Data.Shape))
	copy(shape, v.Data.Shape)
	return shape
}

// Reshape changes the shape of the variable.
func (v *Variable) Reshape(shape ...int) *Variable {
	v.Data = tensor.Reshape(v.Data, shape...)
	return v
}

func (v *Variable) Cleargrad() {
	v.Grad = nil
}

func (v *Variable) SetCreator(f *Function) {
	v.Creator = f
	v.Generation = f.Generation + 1
}

func (v *Variable) Unchain() {
	v.Creator = nil
}

func (v *Variable) UnchainBackward() {
	if v.Creator == nil {
		return
	}

	fs := append(make([]*Function, 0), v.Creator)
	for len(fs) > 0 {
		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// unchain
		for _, x := range f.Input {
			if x.Creator == nil {
				continue
			}

			fs = append(fs, x.Creator)
			x.Unchain()
		}
	}
}

func (v *Variable) Backward(opts ...Opts) {
	if v.Grad == nil {
		v.Grad = OneLike(v)
	}

	if v.Creator == nil {
		return
	}

	seen := make(map[*Function]bool)
	fs := addFunc(make([]*Function, 0), v.Creator, seen)
	for len(fs) > 0 {
		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// gys
		gys := grads(f.Output)

		// backward
		func() {
			if NoCreateGraph(opts...) {
				defer Nograd().End()
			}

			// backward
			gxs := f.Backward(gys...)

			// chain
			xs, gxs := zip(f.Input, gxs)
			for i, x := range xs {
				x.Grad = add(x.Grad, gxs[i])

				if x.Creator != nil {
					fs = addFunc(fs, x.Creator, seen)
				}
			}
		}()

		if NoRetainGrad(opts...) {
			cleargrad(f.Output)
		}
	}
}

func (v *Variable) String() string {
	name := "variable"
	if v.Name != "" {
		name = v.Name
	}

	s1 := true
	for _, s := range v.Shape() {
		if s == 1 {
			continue
		}

		s1 = false
		break
	}

	if s1 {
		return fmt.Sprintf("%s(%v)", name, v.At())
	}

	return fmt.Sprintf("%s%v(%v)", name, v.Shape(), tensor.Contiguous(v.Data).Data)
}

func addFunc(fs []*Function, f *Function, seen map[*Function]bool) []*Function {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	sort.Slice(fs, func(i, j int) bool { return fs[i].Generation < fs[j].Generation })
	return fs
}

func zip(xs, gxs []*Variable) ([]*Variable, []*Variable) {
	n := min(len(xs), len(gxs))
	return xs[:n], gxs[:n]
}

func grads(y []*Variable) []*Variable {
	gys := make([]*Variable, len(y))
	for i := range y {
		gys[i] = y[i].Grad
	}

	return gys
}

func add(xgrad, gx *Variable) *Variable {
	if xgrad == nil {
		return gx
	}

	// NOTE: create graph
	return Add(xgrad, gx)
}

func cleargrad(output []*Variable) {
	for _, y := range output {
		y.Cleargrad()
	}
}
