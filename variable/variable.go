package variable

import (
	"fmt"
	"sort"

	"github.com/itsubaki/autograd/vector"
)

type Data = []float64

type Function interface {
	Input() []*Variable
	Output() []*Variable
	Generation() int
	Backward(gy ...Data) []Data
}

type Variable struct {
	Data, Grad Data
	Creator    Function
	Generation int
}

func New(v ...float64) *Variable {
	return &Variable{Data: v}
}

func NewLikeWith(c float64, v *Variable) *Variable {
	return &Variable{Data: vector.NewLikeWith(c, v.Data)}
}

func OneLike(v *Variable) *Variable {
	return NewLikeWith(1.0, v)
}

func Copy(v *Variable) *Variable {
	w := vector.NewLike(v.Data)
	copy(w, v.Data)

	return &Variable{Data: w}
}

func (v *Variable) Cleargrad() {
	v.Grad = nil
}

func (v *Variable) SetCreator(f Function) {
	v.Creator = f
	v.Generation = f.Generation() + 1
}

func (v *Variable) Backward(retain ...bool) {
	if len(v.Grad) == 0 {
		v.Grad = vector.OneLike(v.Data)
	}

	if v.Creator == nil {
		return
	}

	seen := make(map[Function]bool)
	fs := addfunc(make([]Function, 0), v.Creator, seen)

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
			x[i].Grad = gx(gxs[i], x[i].Grad)

			if x[i].Creator != nil {
				fs = addfunc(fs, x[i].Creator, seen)
			}

			// clear unnecessary grad
			cleargrad(f.Output(), retain...)
		}
	}
}

func (v Variable) String() string {
	return fmt.Sprintf("variable(%v)", v.Data)
}

func addfunc(fs []Function, f Function, seen map[Function]bool) []Function {
	if _, ok := seen[f]; ok {
		return fs
	}

	seen[f] = true
	fs = append(fs, f)
	sort.Slice(fs, func(i, j int) bool { return fs[i].Generation() < fs[j].Generation() })
	return fs
}

func cleargrad(output []*Variable, retain ...bool) {
	if len(retain) > 0 && retain[0] {
		return
	}

	for i := range output {
		output[i].Cleargrad()
	}
}

func gx(gxs, xgrad []float64) []float64 {
	if len(xgrad) > 0 {
		gxs = vector.Add(gxs, xgrad)
	}

	return gxs
}

func gys(y []*Variable) []Data {
	gys := make([]Data, len(y))
	for i := range gys {
		gys[i] = y[i].Grad
	}

	return gys
}
