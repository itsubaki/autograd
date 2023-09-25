package variable

import (
	"fmt"
	"sort"

	"github.com/itsubaki/autograd/vector"
)

type Data = []float64

type Variable struct {
	Name       string
	Data       Data
	Grad       *Variable
	Creator    *Function
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: v}
}

func Const(c float64) *Variable {
	return &Variable{Name: "const", Data: vector.Const(c)}
}

func OneLike(v *Variable) *Variable {
	return New(vector.OneLike(v.Data)...)
}

func (v *Variable) Cleargrad() {
	v.Grad = nil
}

func (v *Variable) SetCreator(f *Function) {
	v.Creator = f
	v.Generation = f.Generation() + 1
}

func (v *Variable) Backward() {
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

		// backward
		x, y := f.Input(), f.Output()
		gxs := f.Backward(gys(y)...)

		for i := range x {
			x[i].Grad = gx(x[i].Grad, gxs[i])

			if x[i].Creator != nil {
				fs = addFunc(fs, x[i].Creator, seen)
			}
		}
	}
}

func (v Variable) String() string {
	if v.Name == "" {
		return fmt.Sprintf("variable%v", v.Data)
	}

	return fmt.Sprintf("%v%v", v.Name, v.Data)
}

func addFunc(fs []*Function, f *Function, seen map[*Function]bool) []*Function {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	sort.Slice(fs, func(i, j int) bool { return fs[i].Generation() < fs[j].Generation() })
	return fs
}

func gys(y []*Variable) []*Variable {
	gys := make([]*Variable, len(y))
	for i := range y {
		gys[i] = y[i].Grad
	}

	return gys
}

func gx(xgrad, gx *Variable) *Variable {
	if xgrad == nil {
		return gx
	}

	// NOTE: creating graph
	return Add(xgrad, gx)
}
