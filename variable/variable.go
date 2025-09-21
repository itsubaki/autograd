package variable

import (
	"fmt"
	randv2 "math/rand/v2"
	"sort"

	"github.com/itsubaki/autograd/matrix"
)

type Variable struct {
	Name       string
	Data       *matrix.Matrix
	Grad       *Variable
	Creator    *Function
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: matrix.New(v)}
}

func From(v *matrix.Matrix) *Variable {
	return &Variable{Data: v}
}

func ZeroLike(v *Variable) *Variable {
	return &Variable{Data: matrix.ZeroLike(v.Data)}
}

func OneLike(v *Variable) *Variable {
	return &Variable{Data: matrix.OneLike(v.Data)}
}

func Zeros(shape []int) *Variable {
	return &Variable{Data: matrix.Zeros(shape[0], shape[1])}
}

func Rand(shape []int, s ...randv2.Source) *Variable {
	return &Variable{Data: matrix.Rand(shape[0], shape[1], s...)}
}

func Randn(shape []int, s ...randv2.Source) *Variable {
	return &Variable{Data: matrix.Randn(shape[0], shape[1], s...)}
}

func (v *Variable) At(coord ...int) float64 {
	return v.Data.At(coord[0], coord[1])
}

func (v *Variable) Shape() []int {
	return matrix.Shape(v.Data)
}

func (v *Variable) Reshape(shape ...int) *Variable {
	v.Data = matrix.Reshape(v.Data, shape...)
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

	for {
		if len(fs) == 0 {
			break
		}

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

	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// gys
		gys := gys(f.Output)

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

	if v.Data.Rows == 1 && v.Data.Cols == 1 {
		return fmt.Sprintf("%s(%v)", name, v.At(0, 0))
	}

	if v.Data.Rows == 1 {
		return fmt.Sprintf("%s%v(%v)", name, v.Shape(), v.Data.Row(0))
	}

	return fmt.Sprintf("%s%v(%v)", name, v.Shape(), v.Data)
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

func gys(y []*Variable) []*Variable {
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
