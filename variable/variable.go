package variable

import (
	"fmt"
	"sort"

	"github.com/itsubaki/autograd/matrix"
)

type Variable struct {
	Name       string
	Data       matrix.Matrix
	Grad       *Variable
	Creator    *Function
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: matrix.New(v)}
}

func NewOf(v ...[]float64) *Variable {
	return &Variable{Data: v}
}

func Const(c float64) *Variable {
	return &Variable{Name: "const", Data: matrix.Const(c)}
}

func OneLike(v *Variable) *Variable {
	return NewOf(matrix.OneLike(v.Data)...)
}

func (v *Variable) Cleargrad() {
	v.Grad = nil
}

func (v *Variable) SetCreator(f *Function) {
	v.Creator = f
	v.Generation = f.Generation + 1
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
			if !HasCreateGraph(opts...) {
				defer Nograd().End()
			}

			gxs := f.Backward(gys...)
			for i, x := range f.Input {
				x.Grad = add(x.Grad, gxs[i])

				if x.Creator != nil {
					fs = addFunc(fs, x.Creator, seen)
				}
			}
		}()

		if !HasRetainGrad(opts...) {
			cleargrad(f.Output)
		}
	}
}

func (v Variable) String() string {
	name := "variable"
	if v.Name != "" {
		name = v.Name
	}

	if len(v.Data) == 1 {
		return fmt.Sprintf("%s(%v)", name, v.Data[0])
	}

	return fmt.Sprintf("%s(%v)", name, v.Data)
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
