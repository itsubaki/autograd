package variable

import (
	"fmt"

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

func OneLike(v *Variable) *Variable {
	return &Variable{Data: vector.OneLike(v.Data)}
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

	fs := make([]Function, 0)
	seen := make(map[Function]bool)
	add := func(f Function) {
		if _, ok := seen[f]; ok {
			return
		}

		seen[f] = true
		fs = append(fs, f)
		sort(fs)
	}

	add(v.Creator)

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
				add(x[i].Creator)
			}

			// clear unnecessary grad
			cleargrad(f.Output(), retain...)
		}
	}
}

func (v Variable) String() string {
	return fmt.Sprintf("variable(%v)", v.Data)
}

func sort(fs []Function) {
	for i := range fs {
		for j := range fs {
			if fs[i].Generation() < fs[j].Generation() {
				fs[i], fs[j] = fs[j], fs[i]
			}
		}
	}
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
