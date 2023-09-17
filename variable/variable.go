package variable

import (
	"fmt"

	"github.com/itsubaki/autograd/vector"
)

type Data = []float64

type Function interface {
	Input() []*Variable
	Output() []*Variable
	Backward(gy []Data) []Data
}

type Variable struct {
	Data, Grad Data
	Creator    Function
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

func (v *Variable) Backward() {
	if len(v.Grad) == 0 {
		v.Grad = vector.OneLike(v.Data)
	}

	if v.Creator == nil {
		return
	}

	fs := []Function{v.Creator}
	for {
		if len(fs) == 0 {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// backward
		x, y := f.Input(), f.Output()
		gys := make([]Data, len(y))
		for i := range gys {
			gys[i] = y[i].Grad
		}
		gxs := f.Backward(gys)

		for i := range x {
			x[i].Grad = gxs[i]

			if x[i].Creator != nil {
				fs = append(fs, x[i].Creator)
			}
		}
	}
}

func (v Variable) String() string {
	return fmt.Sprintf("variable(%v)", v.Data)
}
